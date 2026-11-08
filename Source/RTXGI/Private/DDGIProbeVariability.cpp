// RTXGI GI SDK Probe Variability
// See DDGIProbeVariability.h.

#include "DDGIProbeVariability.h"

#if RHI_RAYTRACING

#include "DDGIVolumeComponent.h"
#include "Algo/Sort.h"
#include "GlobalShader.h"
#include "LightSceneProxy.h"
#include "ShaderParameterStruct.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RenderUtils.h"
#include "RHIGPUReadback.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/Crc.h"
#include "SceneManagement.h"
#include "SceneProxies/SkyLightSceneProxy.h"

// UE private interfaces
#include "ScenePrivate.h"
#include "LightSceneInfo.h"

static TAutoConsoleVariable<int32> CVarDDGIVariabilityPauseConverged(
	TEXT("r.RTXGI.DDGI.Variability.PauseConverged"),
	1,
	TEXT("1: volumes with Probe Variability enabled stop updating once converged, apart from periodic checks.\n")
	TEXT("0: they keep updating; variability is still measured (stats, probe visualization)."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<float> CVarDDGIVariabilityMinSweeps(
	TEXT("r.RTXGI.DDGI.Variability.MinSweeps"),
	16.0f,
	TEXT("Full probe sweeps a volume must blend after a reset before it may count as converged. Filters out the low readings\n")
	TEXT("of freshly cleared probes (they take new values directly) and gives probe history time to settle."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarDDGIVariabilityCheckIntervalFrames(
	TEXT("r.RTXGI.DDGI.Variability.CheckIntervalFrames"),
	60,
	TEXT("Frames between the measure-only checks of a converged volume (trace, irradiance measurement and classification,\n")
	TEXT("nothing written). A check that measures variability above the threshold, or a probe whose classification would\n")
	TEXT("change, wakes the volume. At most one volume is checked per frame. 0 disables checks: converged volumes then only\n")
	TEXT("wake on volume, light or sky light changes."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarDDGIVariabilityAllowRealTimeSkyCapture(
	TEXT("r.RTXGI.DDGI.Variability.AllowRealTimeSkyCapture"),
	0,
	TEXT("0: volumes never converge while the sky light uses real-time capture (the capture can change every frame without\n")
	TEXT("any light or sky light property changing). 1: allow it and rely on the periodic checks."),
	ECVF_RenderThreadSafe);

// Portable reduction of the variability atlas
class FDDGIProbeVariabilityReduction : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FDDGIProbeVariabilityReduction)
	SHADER_USE_PARAMETER_STRUCT(FDDGIProbeVariabilityReduction, FGlobalShader)

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, InputTexture)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, OutputTexture)
		SHADER_PARAMETER(FIntPoint, InputSize)
		SHADER_PARAMETER(FIntPoint, OutputSize)
	END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FDDGIProbeVariabilityReduction, "/Plugin/RTXGI/Private/SDK/ddgi/ProbeVariabilityReductionCS.usf", "DDGIProbeVariabilityReductionCS", SF_Compute);

namespace DDGIProbeVariability
{
	namespace
	{
		constexpr int32 ReductionFootprint = 8;

		// Readback texel format. RGBA32F is a storage format on every RHI (RG32F is optional on Vulkan).
		constexpr EPixelFormat ReductionFormat = PF_A32B32G32R32F;

		struct FLightRecord
		{
			uint32 Hash = 0;
			FSphere Bounds = FSphere(ForceInit);
			bool bDirectional = false;
			uint64 LastSeenFrame = 0;
		};

		struct FSceneLightingRecord
		{
			TMap<const FLightSceneInfo*, FLightRecord> Lights;
			uint32 SkyLightHash = 0;
			bool bInitialized = false;
		};

		TMap<const void*, FSceneLightingRecord> GSceneLightingRecords;

		struct FLightingChanges
		{
			bool bAllVolumes = false;
			TArray<FSphere, TInlineAllocator<8>> ChangedBounds;

			bool Affects(const FBox& VolumeInfluenceBox) const
			{
				if (bAllVolumes)
				{
					return true;
				}
				for (const FSphere& Bounds : ChangedBounds)
				{
					if (FMath::SphereAABBIntersection(Bounds, VolumeInfluenceBox))
					{
						return true;
					}
				}
				return false;
			}
		};

		void ResetState(FDDGIProbeVariabilityState& State)
		{
			++State.Generation;
			State.SweepsSinceReset = 0.0f;
			State.bHasReadingForGeneration = false;
			State.bConverged = false;
		}

		uint32 HashLight(const FLightSceneProxy& Proxy)
		{
			// Unset fields must hash the same every frame
			FLightRenderParameters Params;
			FMemory::Memzero(Params);
			Proxy.GetLightShaderParameters(Params);

			struct FHashData
			{
				FVector WorldPosition;
				FVector3f Direction;
				FVector3f Tangent;
				FLinearColor Color;
				FVector2f SpotAngles;
				float InvRadius;
				float FalloffExponent;
				float DiffuseScale;
				float SourceRadius;
				float SoftSourceRadius;
				float SourceLength;
				float RectLightBarnCosAngle;
				float RectLightBarnLength;
				float IndirectLightingScale;
				const void* IESTexture;
				const void* LightFunctionMaterial;
				uint8 LightType;
				uint8 LightingChannelMask;
				uint8 bCastDynamicShadow;
			};

			FHashData Data;
			FMemory::Memzero(Data);
			Data.WorldPosition = Params.WorldPosition;
			Data.Direction = Params.Direction;
			Data.Tangent = Params.Tangent;
			Data.Color = Params.Color;
			Data.SpotAngles = Params.SpotAngles;
			Data.InvRadius = Params.InvRadius;
			Data.FalloffExponent = Params.FalloffExponent;
			Data.DiffuseScale = Params.DiffuseScale;
			Data.SourceRadius = Params.SourceRadius;
			Data.SoftSourceRadius = Params.SoftSourceRadius;
			Data.SourceLength = Params.SourceLength;
			Data.RectLightBarnCosAngle = Params.RectLightBarnCosAngle;
			Data.RectLightBarnLength = Params.RectLightBarnLength;
			Data.IndirectLightingScale = Proxy.GetIndirectLightingScale();
			Data.IESTexture = Proxy.GetIESTexture();
			Data.LightFunctionMaterial = Proxy.GetLightFunctionMaterial();
			Data.LightType = Proxy.GetLightType();
			Data.LightingChannelMask = Proxy.GetLightingChannelMask();
			Data.bCastDynamicShadow = Proxy.CastsDynamicShadow() ? 1 : 0;

			return FCrc::MemCrc32(&Data, sizeof(Data));
		}

		uint32 HashSkyLight(const FSkyLightSceneProxy* SkyLight)
		{
			if (!SkyLight)
			{
				return 0;
			}

			struct FHashData
			{
				FLinearColor EffectiveColor;
				FLinearColor LowerHemisphereColor;
				const void* ProcessedTexture;
				const void* BlendDestinationProcessedTexture;
				float BlendFraction;
				float AverageBrightness;
				float IndirectLightingIntensity;
				float SkyDistanceThreshold;
				uint8 bLowerHemisphereIsSolidColor;
				uint8 bCastShadows;
				uint8 bAffectGlobalIllumination;
				uint8 bRealTimeCaptureEnabled;
			};

			FHashData Data;
			FMemory::Memzero(Data);
			Data.EffectiveColor = SkyLight->GetEffectiveLightColor();
			Data.LowerHemisphereColor = SkyLight->LowerHemisphereColor;
			Data.ProcessedTexture = SkyLight->ProcessedTexture;
			Data.BlendDestinationProcessedTexture = SkyLight->BlendDestinationProcessedTexture;
			Data.BlendFraction = SkyLight->BlendFraction;
			Data.AverageBrightness = SkyLight->AverageBrightness;
			Data.IndirectLightingIntensity = SkyLight->IndirectLightingIntensity;
			Data.SkyDistanceThreshold = SkyLight->SkyDistanceThreshold;
			Data.bLowerHemisphereIsSolidColor = SkyLight->bLowerHemisphereIsSolidColor ? 1 : 0;
			Data.bCastShadows = SkyLight->bCastShadows ? 1 : 0;
			Data.bAffectGlobalIllumination = SkyLight->bAffectGlobalIllumination ? 1 : 0;
			Data.bRealTimeCaptureEnabled = SkyLight->bRealTimeCaptureEnabled ? 1 : 0;

			// The SH also changes on a recapture that keeps the same cubemap texture
			const uint32 PropertiesHash = FCrc::MemCrc32(&Data, sizeof(Data));
			return FCrc::MemCrc32(&SkyLight->IrradianceEnvironmentMap, sizeof(SkyLight->IrradianceEnvironmentMap), PropertiesHash);
		}

		FLightingChanges DetectLightingChanges(const FScene& Scene, uint64 FrameCounter)
		{
			FLightingChanges Changes;
			FSceneLightingRecord& Record = GSceneLightingRecords.FindOrAdd(Scene.GetWorld());

			// Sky light: its irradiance reaches every probe that sees the sky
			const uint32 SkyLightHash = HashSkyLight(Scene.SkyLight);
			if (Record.bInitialized && SkyLightHash != Record.SkyLightHash)
			{
				Changes.bAllVolumes = true;
			}
			Record.SkyLightHash = SkyLightHash;

			for (const FLightSceneInfoCompact& LightCompact : Scene.Lights)
			{
				const FLightSceneInfo* LightSceneInfo = LightCompact.LightSceneInfo;
				if (!LightSceneInfo || !LightSceneInfo->Proxy)
				{
					continue;
				}

				const FLightSceneProxy& LightProxy = *LightSceneInfo->Proxy;
				const bool bDirectional = (LightProxy.GetLightType() == LightType_Directional);
				const uint32 Hash = HashLight(LightProxy);
				const FSphere Bounds = bDirectional ? FSphere(ForceInit) : LightProxy.GetBoundingSphere();

				if (FLightRecord* Existing = Record.Lights.Find(LightSceneInfo))
				{
					if (Existing->Hash != Hash)
					{
						if (bDirectional || Existing->bDirectional)
						{
							Changes.bAllVolumes = true;
						}
						else
						{
							Changes.ChangedBounds.Add(Existing->Bounds);
							Changes.ChangedBounds.Add(Bounds);
						}
						Existing->Hash = Hash;
						Existing->Bounds = Bounds;
						Existing->bDirectional = bDirectional;
					}
					Existing->LastSeenFrame = FrameCounter;
				}
				else
				{
					if (Record.bInitialized)
					{
						if (bDirectional)
						{
							Changes.bAllVolumes = true;
						}
						else
						{
							Changes.ChangedBounds.Add(Bounds);
						}
					}

					FLightRecord& NewRecord = Record.Lights.Add(LightSceneInfo);
					NewRecord.Hash = Hash;
					NewRecord.Bounds = Bounds;
					NewRecord.bDirectional = bDirectional;
					NewRecord.LastSeenFrame = FrameCounter;
				}
			}

			// Removed lights
			for (auto It = Record.Lights.CreateIterator(); It; ++It)
			{
				if (It.Value().LastSeenFrame != FrameCounter)
				{
					if (It.Value().bDirectional)
					{
						Changes.bAllVolumes = true;
					}
					else
					{
						Changes.ChangedBounds.Add(It.Value().Bounds);
					}
					It.RemoveCurrent();
				}
			}

			Record.bInitialized = true;
			return Changes;
		}

		
		FBox GetVolumeInfluenceBox(const FDDGIVolumeSceneProxy* VolProxy)
		{
			const FDDGIVolumeSceneProxy::FComponentData& Data = VolProxy->ComponentData;
			const FVector Extent = FVector(Data.Transform.GetScale3D()) * 100.0;
			const FTransform VolumeTransform(FQuat(Data.Transform.GetRotation()), FVector(Data.Origin));
			const FBox VolumeBox = FBox(-Extent, Extent).TransformBy(VolumeTransform);
			return VolumeBox.ExpandBy(FMath::Min(double(Data.ProbeMaxRayDistance), 1.0e7));
		}

		void PollReadbacks(FDDGIVolumeSceneProxy* VolProxy, uint64 FrameCounter, bool bAllowConvergence)
		{
			FDDGIProbeVariabilityState& State = VolProxy->VariabilityState;

			if (!FRDGBuilder::GetAsyncExecuteTask().IsCompleted())
			{
				return;
			}

			//FIFO
			int32 SlotOrder[FDDGIProbeVariabilityState::NumReadbackSlots];
			int32 NumPending = 0;
			for (int32 SlotIndex = 0; SlotIndex < FDDGIProbeVariabilityState::NumReadbackSlots; ++SlotIndex)
			{
				if (State.Slots[SlotIndex].bPending)
				{
					SlotOrder[NumPending++] = SlotIndex;
				}
			}
			Algo::Sort(MakeArrayView(SlotOrder, NumPending), [&State](int32 A, int32 B)
			{
				return State.Slots[A].EnqueueFrame < State.Slots[B].EnqueueFrame;
			});

			const float Threshold = VolProxy->ComponentData.VariabilityThreshold;
			const float MinSweeps = FMath::Max(0.0f, CVarDDGIVariabilityMinSweeps.GetValueOnRenderThread());

			for (int32 OrderIndex = 0; OrderIndex < NumPending; ++OrderIndex)
			{
				FDDGIProbeVariabilityState::FReadbackSlot& Slot = State.Slots[SlotOrder[OrderIndex]];

				if (FrameCounter < Slot.EnqueueFrame + 2 || !Slot.Readback->IsReady())
				{
					break;
				}

				float Sum = 0.0f;
				float Weight = 0.0f;
				int32 RowPitchInPixels = 0;
				const float* Texel = static_cast<const float*>(Slot.Readback->Lock(RowPitchInPixels));
				const bool bHasData = (Texel != nullptr);
				if (bHasData)
				{
					Sum = Texel[0];
					Weight = Texel[1];
					Slot.Readback->Unlock();
				}
				Slot.bPending = false;

				if (!bHasData || Slot.Generation != State.Generation)
				{
					continue;
				}

				State.LastVariability = (Weight > 0.0f) ? (Sum / Weight) : TNumericLimits<float>::Max();
				State.LastWeight = Weight;
				State.bHasReadingForGeneration = true;

				// G == nothing was measured
				// so the probe was either rejected or early-outted
				const bool bBelowThreshold = (Weight > 0.0f) && (State.LastVariability < Threshold);

				if (State.bConverged)
				{
					if (!bBelowThreshold)
					{
						
						ResetState(State);
					}
				}
				else if (bBelowThreshold && bAllowConvergence && Slot.SweepsAtEnqueue >= MinSweeps)
				{
					State.bConverged = true;
					State.LastCheckFrame = FrameCounter;
				}
			}
		}
	}

	bool IsEnabledForVolume(const FDDGIVolumeSceneProxy* VolProxy)
	{
		return VolProxy->ComponentData.EnableProbeVariability && VolProxy->ProbesVariability.IsValid();
	}

	bool IsConverged(const FDDGIVolumeSceneProxy* VolProxy)
	{
		return IsEnabledForVolume(VolProxy) && VolProxy->VariabilityState.bConverged;
	}

	void BeginFrame_RenderThread(const FScene& Scene, uint64 FrameCounter, TArrayView<FDDGIVolumeSceneProxy* const> SceneVolumes)
	{
		bool bAnyEnabled = false;
		for (FDDGIVolumeSceneProxy* VolProxy : SceneVolumes)
		{
			if (IsEnabledForVolume(VolProxy))
			{
				bAnyEnabled = true;
			}
			else
			{
				// Enabling variability requests a reset through the component data change
				VolProxy->bVariabilityResetRequested = false;
				if (VolProxy->VariabilityState.bConverged)
				{
					ResetState(VolProxy->VariabilityState);
				}
			}
		}

		if (!bAnyEnabled)
		{
			return;
		}

		const FLightingChanges LightingChanges = DetectLightingChanges(Scene, FrameCounter);

		const bool bRealTimeSkyCaptureBlocks = Scene.SkyLight && Scene.SkyLight->bRealTimeCaptureEnabled && CVarDDGIVariabilityAllowRealTimeSkyCapture.GetValueOnRenderThread() == 0;
		const bool bAllowConvergence = (CVarDDGIVariabilityPauseConverged.GetValueOnRenderThread() != 0) && !bRealTimeSkyCaptureBlocks;

		for (FDDGIVolumeSceneProxy* VolProxy : SceneVolumes)
		{
			if (!IsEnabledForVolume(VolProxy))
			{
				continue;
			}

			FDDGIProbeVariabilityState& State = VolProxy->VariabilityState;

			const bool bReset =
				VolProxy->bVariabilityResetRequested ||
				(State.bConverged && !bAllowConvergence) ||
				LightingChanges.Affects(GetVolumeInfluenceBox(VolProxy));

			VolProxy->bVariabilityResetRequested = false;
			if (bReset)
			{
				ResetState(State);
			}

			PollReadbacks(VolProxy, FrameCounter, bAllowConvergence);
		}
	}

	FDDGIVolumeSceneProxy* SelectVolumeToCheck_RenderThread(uint64 FrameCounter, TArrayView<FDDGIVolumeSceneProxy* const> SceneVolumes)
	{
		const int32 CheckIntervalFrames = CVarDDGIVariabilityCheckIntervalFrames.GetValueOnRenderThread();
		if (CheckIntervalFrames <= 0)
		{
			return nullptr;
		}

		FDDGIVolumeSceneProxy* VolumeToCheck = nullptr;
		uint64 LongestWait = 0;
		for (FDDGIVolumeSceneProxy* VolProxy : SceneVolumes)
		{
			if (!IsConverged(VolProxy))
			{
				continue;
			}

			const uint64 FramesSinceCheck = FrameCounter - VolProxy->VariabilityState.LastCheckFrame;
			if (FramesSinceCheck >= uint64(CheckIntervalFrames) && FramesSinceCheck > LongestWait)
			{
				VolumeToCheck = VolProxy;
				LongestWait = FramesSinceCheck;
			}
		}

		if (VolumeToCheck)
		{
			VolumeToCheck->VariabilityState.LastCheckFrame = FrameCounter;
		}
		return VolumeToCheck;
	}

	void OnVolumeUpdated_RenderThread(FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, uint64 FrameCounter, bool bMeasureOnly)
	{
		check(IsEnabledForVolume(VolProxy));
		FDDGIProbeVariabilityState& State = VolProxy->VariabilityState;

		if (!bMeasureOnly)
		{
			const int32 ProbeCount = GetProbeCount(VolProxy->ComponentData.ProbeCounts);
			if (ProbeCount > 0)
			{
				State.SweepsSinceReset += float(VolProxy->ProbeIndexCount) / float(ProbeCount);
			}
		}

		FDDGIProbeVariabilityState::FReadbackSlot* FreeSlot = nullptr;
		for (FDDGIProbeVariabilityState::FReadbackSlot& Slot : State.Slots)
		{
			if (!Slot.bPending)
			{
				FreeSlot = &Slot;
				break;
			}
		}
		if (!FreeSlot)
		{
			return;
		}

		RDG_EVENT_SCOPE(GraphBuilder, "DDGI Probe Variability");

		FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GMaxRHIFeatureLevel);
		TShaderMapRef<FDDGIProbeVariabilityReduction> ComputeShader(ShaderMap);

		FRDGTextureRef Input = GraphBuilder.RegisterExternalTexture(VolProxy->ProbesVariability);
		FIntPoint InputSize = Input->Desc.Extent;
		do
		{
			const FIntPoint OutputSize = FIntPoint::DivideAndRoundUp(InputSize, ReductionFootprint);

			FRDGTextureDesc OutputDesc = FRDGTextureDesc::Create2D(OutputSize, ReductionFormat, FClearValueBinding::None, TexCreate_ShaderResource | TexCreate_UAV);
			FRDGTextureRef Output = GraphBuilder.CreateTexture(OutputDesc, TEXT("DDGIVariabilityReduction"));

			FDDGIProbeVariabilityReduction::FParameters* PassParameters = GraphBuilder.AllocParameters<FDDGIProbeVariabilityReduction::FParameters>();
			PassParameters->InputTexture = Input;
			PassParameters->OutputTexture = GraphBuilder.CreateUAV(Output);
			PassParameters->InputSize = InputSize;
			PassParameters->OutputSize = OutputSize;

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("DDGI Variability Reduction %dx%d", OutputSize.X, OutputSize.Y),
				ComputeShader,
				PassParameters,
				FComputeShaderUtils::GetGroupCount(OutputSize, 8));

			Input = Output;
			InputSize = OutputSize;
		}
		while (InputSize.X > 1 || InputSize.Y > 1);

		if (!FreeSlot->Readback)
		{
			FreeSlot->Readback = MakeUnique<FRHIGPUTextureReadback>(TEXT("DDGIProbeVariabilityReadback"));
		}

		AddEnqueueCopyPass(GraphBuilder, FreeSlot->Readback.Get(), Input);

		FreeSlot->bPending = true;
		FreeSlot->EnqueueFrame = FrameCounter;
		FreeSlot->Generation = State.Generation;
		FreeSlot->SweepsAtEnqueue = State.SweepsSinceReset;
	}

	void RemoveScene_RenderThread(const void* SceneKey)
	{
		GSceneLightingRecords.Remove(SceneKey);
	}
}

#endif // RHI_RAYTRACING
