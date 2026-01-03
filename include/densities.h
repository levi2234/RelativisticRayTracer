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
 * Optimized for the Blue/Wispy reference: 
 * Fluid, turbulent filaments without digital banding.
 */
__device__ __forceinline__ float getDustCloudDensity(float3 p, float time) {
    float r = length(make_float3(p.x, 0.0f, p.z));
    if (r < ISCO_RADIUS || r > DISK_OUT_M) return 0.0f;

    // 1. Base Envelope
    float edge_falloff = smoothstep(DISK_OUT_M, DISK_OUT_M * 0.8f, r);
    
    // INNER TAPER: Soften the density near the ISCO so we can see the horizon/lensing
    float inner_taper = smoothstep(ISCO_RADIUS, ISCO_RADIUS + 5.0f, r);
    
    // Thinner disk overall, especially near the center
    float local_h = CLOUD_H_M * 0.5f * powf(ISCO_RADIUS / r, 0.2f); 
    float vertical_profile = expf(-(p.y * p.y) / (2.0f * local_h * local_h + 1e-7f));
    
    float base = vertical_profile * edge_falloff * inner_taper;
    
    if (base < 0.001f) return 0.0f;

    // 2. DIFFERENTIAL SHEARING
    float phi = atan2f(p.z, p.x);
    float omega = 1.0f * powf(ISCO_RADIUS / r, 1.5f); 
    float angle_rot = phi - time * omega; 

    // 3. FLUID DOMAIN WARPING
    float3 coords = make_float3(r * 0.8f, p.y * 15.0f, angle_rot * 10.0f);
    
    float3 w1 = make_float3(
        fbm(mul(coords, 0.15f), 2),
        fbm(add(mul(coords, 0.15f), make_float3(1,2,3)), 2),
        fbm(add(mul(coords, 0.15f), make_float3(4,5,6)), 2)
    );
    
    float3 w2_coords = add(coords, mul(w1, 3.0f));
    float3 w2 = make_float3(
        fbm(mul(w2_coords, 0.4f), 2),
        fbm(add(mul(w2_coords, 0.4f), make_float3(2,1,0)), 2),
        fbm(add(mul(w2_coords, 0.4f), make_float3(0,3,1)), 2)
    );

    float3 final_coords = add(coords, mul(w2, 1.5f));

    // 4. MULTI-OCTAVE WISPS
    float n = 0.0f;
    float amp = 1.0f;
    float freq = 1.0f;
    for(int i = 0; i < 5; i++) {
        float noise_val = noise3D(mul(final_coords, freq));
        float wisp = 1.0f - fabsf(noise_val * 2.0f - 1.0f);
        n += wisp * amp;
        amp *= 0.5f;
        freq *= 2.1f;
    }

    // 5. CONTRAST & TRANSPARENCY
    // Lower the floor and increase the contrast to allow more light through
    float strands = smoothstep(0.4f, 0.8f, n * 0.55f);
    strands = powf(strands, 4.0f); 
    
    float detail = fbm(add(mul(final_coords, 4.0f), make_float3(0, time * 0.5f, 0)), 2);
    strands *= (0.6f + 0.4f * detail);

    // Reduced density multiplier (from 20.0 to 12.0) for more transparency
    return base * strands * 12.0f;
}

#endif
