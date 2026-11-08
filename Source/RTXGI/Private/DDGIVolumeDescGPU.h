/*
* Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef RTXGI_DDGI_VOLUME_DESC_GPU_H
#define RTXGI_DDGI_VOLUME_DESC_GPU_H

#include "LegacyEngineCompat.h"


/**
 * DDGIVolumeDescGPU
 * The condensed DDGIVolume descriptor for use on the GPU.
 */
BEGIN_UNIFORM_BUFFER_STRUCT(FDDGIVolumeDescGPU, )
	SHADER_PARAMETER(FVector3f,       origin)
	SHADER_PARAMETER(int,           numRaysPerProbe)
	SHADER_PARAMETER(FVector4f,      rotation)
	SHADER_PARAMETER(FVector3f,       probeGridSpacing)
	SHADER_PARAMETER(float,         probeMaxRayDistance)
	SHADER_PARAMETER(FIntVector,    probeGridCounts)
	SHADER_PARAMETER(float,         probeDistanceExponent)
	SHADER_PARAMETER(float,         probeHysteresis)
	SHADER_PARAMETER(float,         probeChangeThreshold)
	SHADER_PARAMETER(float,         probeBrightnessThreshold)
	SHADER_PARAMETER(float,         probeIrradianceEncodingGamma)
	SHADER_PARAMETER(float,         probeInverseIrradianceEncodingGamma)
	SHADER_PARAMETER(int,           probeNumIrradianceTexels)
	SHADER_PARAMETER(int,           probeNumDistanceTexels)
	SHADER_PARAMETER(float,         normalBias)
	SHADER_PARAMETER(float,         viewBias)
	SHADER_PARAMETER(FMatrix44f,       probeRayRotationTransform)

	SHADER_PARAMETER(int,           volumeMovementType)
	SHADER_PARAMETER(FIntVector,    probeScrollOffsets)

	// Must list the same members as the HLSL struct in Shaders/Private/SDK/DDGIVolumeDescGPU.ush.
	SHADER_PARAMETER(float,         probeRandomRayBackfaceThreshold)
	SHADER_PARAMETER(float,         probeFixedRayBackfaceThreshold)
	SHADER_PARAMETER(float,         probeMinFrontfaceDistance)

	SHADER_PARAMETER(int,           probeRelocationEnabled)     // 1: probe offsets texture is valid and used
	SHADER_PARAMETER(int,           probeRayDataFormat)         // 0: F32x2 (radiance packed R10G10B10), 1: F32x4
	SHADER_PARAMETER(int,           probeIrradianceFormat)      // 0: R10G10B10A2 UNORM, 1: float (16 or 32 bit)
	SHADER_PARAMETER(int,           probeVariabilityEnabled)    // 1: the irradiance blend writes the coefficient of variation
END_UNIFORM_BUFFER_STRUCT()

#endif /* RTXGI_DDGI_VOLUME_DESC_GPU_H */
