#pragma once

#include "CoreMinimal.h"

class FRDGBuilder;
class FScene;
class FDDGIVolumeSceneProxy;

#if RHI_RAYTRACING

namespace DDGIProbeVariability
{
	// The volume has variability enabled and its atlas allocated
	bool IsEnabledForVolume(const FDDGIVolumeSceneProxy* VolProxy);

	// The volume's regular updates are paused
	bool IsConverged(const FDDGIVolumeSceneProxy* VolProxy);

	void BeginFrame_RenderThread(const FScene& Scene, uint64 FrameCounter, TArrayView<FDDGIVolumeSceneProxy* const> SceneVolumes);

	FDDGIVolumeSceneProxy* SelectVolumeToCheck_RenderThread(uint64 FrameCounter, TArrayView<FDDGIVolumeSceneProxy* const> SceneVolumes);

	void OnVolumeUpdated_RenderThread(FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, uint64 FrameCounter, bool bMeasureOnly);

	// World teardown
	void RemoveScene_RenderThread(const void* SceneKey);
}

#endif // RHI_RAYTRACING
