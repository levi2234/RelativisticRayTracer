#ifndef DENSITIES_H
#define DENSITIES_H

#include <cuda_runtime.h>
#include "config.h"
#include "math_utils.h"

/**
 * Temperature profile (Shakura-Sunyaev)
 * T(r) = T_ref * (r/ISCO)^-0.75
 */
__device__ __forceinline__ float getDiskTemperature(float r) {
    if (r < ISCO_RADIUS) return 0.0f;
    return DISK_TEMP_REF * powf(r / ISCO_RADIUS, -0.75f);
}

/**
 * Calculates local gas density with rotating high-fidelity clouds.
 */
__device__ __forceinline__ float getAccretionDensity(float3 p, float time) {
    float r = length(make_float3(p.x, 0.0f, p.z));
    if (r < ISCO_RADIUS || r > DISK_OUT_M) return 0.0f;

    // 1. Base Envelope (Tapering)
    float edge_falloff = 1.0f;
    float edge_start = DISK_OUT_M * 0.85f;
    if (r > edge_start) {
        edge_falloff = 1.0f - (r - edge_start) / (DISK_OUT_M - edge_start);
        edge_falloff *= edge_falloff;
    }

    float local_h = DISK_H_M * powf(ISCO_RADIUS / r, 0.5f);
    float vertical_density = expf(-(p.y * p.y) / (2.0f * local_h * local_h + 1e-7f));
    float radial_density = powf(ISCO_RADIUS / r, 0.4f);
    float base_envelope = vertical_density * radial_density * edge_falloff;

    // 2. High-Fidelity Multi-Octave Clouds
    float phi = atan2f(p.z, p.x);
    
    // Differential rotation: Inner parts move faster (Keplerian)
    float omega = 3.5f * powf(ISCO_RADIUS / r, 1.5f); 
    float angle_rotated = phi - time * omega;

    float3 rot_p = make_float3(
        r * cosf(angle_rotated),
        p.y * 4.0f, // Stretch vertical sampling to make filaments flatter
        r * sinf(angle_rotated)
    );
    
    float evolution = time * 0.35f;
    float3 noise_coords = add(mul(rot_p, 0.45f), make_float3(0, evolution, 0));

    // Sample high-fidelity noise
    float n = fbm(noise_coords, 5); 

    // --- HIGH CONTRAST STREAKS ---
    float cloud = fmaxf(0.0f, n - 0.32f); 
    cloud = powf(cloud * 2.8f, 1.6f);
    cloud = fminf(6.0f, cloud);

    return base_envelope * (0.02f + 5.0f * cloud); 
}

/**
 * Calculates local density for the large-scale "Dust Cloud" layer.
 */
__device__ __forceinline__ float getDustCloudDensity(float3 p, float time) {
    float r = length(make_float3(p.x, 0.0f, p.z));
    if (r < ISCO_RADIUS || r > DISK_OUT_M) return 0.0f;

    float edge_falloff = 1.0f;
    float edge_start = DISK_OUT_M * 0.7f;
    if (r > edge_start) {
        edge_falloff = 1.0f - (r - edge_start) / (DISK_OUT_M - edge_start);
        edge_falloff = smoothstep(0.0f, 1.0f, edge_falloff);
    }

    float boundary_jitter = fbm(add(mul(p, 0.15f), make_float3(0, time * 0.1f, 0)), 2);
    edge_falloff *= smoothstep(0.3f, 0.7f, boundary_jitter + 0.2f);

    float local_h = CLOUD_H_M * powf(ISCO_RADIUS / r, 0.5f); 
    float vertical_profile = expf(-(p.y * p.y) / (2.0f * local_h * local_h + 1e-7f));
    vertical_profile = smoothstep(0.0f, 1.0f, vertical_profile);
    
    float base = vertical_profile * edge_falloff;

    float phi = atan2f(p.z, p.x);
    const float CONSTANT_OMEGA = 0.5f; 
    float angle_rot = phi - time * CONSTANT_OMEGA; 

    float3 sampling_p = make_float3(r * cosf(angle_rot), p.y, r * sinf(angle_rot));
    float3 noise_coords = add(mul(sampling_p, 0.4f), make_float3(0, time * 0.05f, 0));
    
    float3 warp_q = make_float3(
        fbm(add(noise_coords, make_float3(0.0, 0.0, 0.0)), 2),
        fbm(add(noise_coords, make_float3(2.2, 1.3, 0.0)), 2),
        fbm(add(noise_coords, make_float3(1.1, 4.4, 3.1)), 2)
    );
    
    float3 final_coords = add(noise_coords, mul(warp_q, 1.0f)); 
    float n = fbm_billow(final_coords, 5); 
    
    float cloud = smoothstep(0.42f, 0.58f, n);
    cloud = powf(cloud, 1.5f);

    return base * cloud * 12.0f;
}

#endif
