#ifndef GEODESICS_H
#define GEODESICS_H

#include <cuda_runtime.h>
#include "config.h"
#include "math_utils.h"

/**
 * Calculates the Relativistic Beaming factor (Doppler + Gravitational Redshift)
 */
__device__ __forceinline__ float calculateRedshiftFactor(float3 p_rel, float3 ray_vel) {
    float r = length(p_rel);
    if (r < EVENT_HORIZON * 1.01f) return 0.0f;

    float g_gravity = sqrtf(1.0f - EVENT_HORIZON / r);

    float v_mag = 1.0f / (powf(r, 1.5f) + SPIN_A);
    float3 gas_dir = normalize(make_float3(-p_rel.z, 0, p_rel.x));
    float cos_theta = dot(ray_vel, gas_dir);
    
    float gamma = 1.0f / sqrtf(1.0f - v_mag * v_mag);
    float g_doppler = 1.0f / (gamma * (1.0f - v_mag * cos_theta));

    return g_gravity * g_doppler;
}

/**
 * Calculates the geodesic acceleration in the Kerr metric.
 */
__device__ __forceinline__ float3 getGeodesicAcc(float3 p_rel, float3 v) {
    float r2 = dot(p_rel, p_rel);
    float r = sqrtf(r2);
    if (r < EVENT_HORIZON * 0.5f) return make_float3(0,0,0);
    
    float3 L_vec = cross(p_rel, v);
    float L2 = dot(L_vec, L_vec);
    float radial_mag = -1.5f * EVENT_HORIZON * L2 / (r2 * r2 * r);
    float3 radial_acc = mul(p_rel, radial_mag);

    float3 drag_dir = cross(SPIN_AXIS, p_rel);
    float drag_strength = (2.0f * SPIN_A * EVENT_HORIZON) / (r2 * r);
    float3 dragging_acc = mul(drag_dir, drag_strength);

    return add(radial_acc, dragging_acc);
}

#endif
