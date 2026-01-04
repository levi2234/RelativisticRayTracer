#ifndef POST_PROCESSING_H
#define POST_PROCESSING_H

#include <cuda_runtime.h>
#include "math_utils.h"
#include "camera_effects/camera_settings.h"

// Simple hash for film grain noise
__device__ inline float grain_hash(float2 p) {
    return fmodf(sinf(dot(make_float3(p.x, p.y, 0), make_float3(12.9898f, 78.233f, 0))) * 43758.5453f, 1.0f);
}

__device__ inline float3 apply_vignette(float3 color, float2 uv, float intensity) {
    float d = length(sub(make_float3(uv.x, uv.y, 0), make_float3(0.5f, 0.5f, 0)));
    float v = smoothstep(0.8f, 0.2f, d * intensity);
    return mul(color, v);
}

__device__ inline float2 apply_lens_distortion(float2 uv, float k) {
    float2 t_uv = make_float2(uv.x - 0.5f, uv.y - 0.5f);
    float r2 = t_uv.x * t_uv.x + t_uv.y * t_uv.y;
    float f = 1.0f + r2 * k;
    return make_float2(t_uv.x * f + 0.5f, t_uv.y * f + 0.5f);
}

// Simple Bloom Threshold
__device__ inline float3 get_bloom_contribution(float3 color, float threshold) {
    float brightness = dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
    if (brightness > threshold) return color;
    return make_float3(0, 0, 0);
}

#endif

