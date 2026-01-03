#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <cuda_runtime.h>
#include "math_utils.h"
#include "geodesics.h"
#include "config.h"

/**
 * Euler Method (1st Order)
 */
__device__ __forceinline__ void integrate_euler(float3 &p, float3 &v, float h) {
    float3 rel_p = sub(p, MASS_POS);
    float3 acc = getGeodesicAcc(rel_p, v);
    
    p = add(p, mul(v, h));
    v = add(v, mul(acc, h));
}

/**
 * Runge-Kutta 4 (4th Order)
 */
__device__ __forceinline__ void integrate_rk4(float3 &p, float3 &v, float h) {
    float3 p0 = p;
    float3 v0 = v;

    // k1
    float3 p1 = sub(p0, MASS_POS);
    float3 kv1 = getGeodesicAcc(p1, v0);
    float3 kp1 = v0;

    // k2
    float3 v2 = add(v0, mul(kv1, h * 0.5f));
    float3 p2_w = add(p0, mul(kp1, h * 0.5f));
    float3 p2 = sub(p2_w, MASS_POS);
    float3 kv2 = getGeodesicAcc(p2, v2);
    float3 kp2 = v2;

    // k3
    float3 v3 = add(v0, mul(kv2, h * 0.5f));
    float3 p3_w = add(p0, mul(kp2, h * 0.5f));
    float3 p3 = sub(p3_w, MASS_POS);
    float3 kv3 = getGeodesicAcc(p3, v3);
    float3 kp3 = v3;

    // k4
    float3 v4 = add(v0, mul(kv3, h));
    float3 p4_w = add(p0, mul(kp3, h));
    float3 p4 = sub(p4_w, MASS_POS);
    float3 kv4 = getGeodesicAcc(p4, v4);
    float3 kp4 = v4;

    // Final combination
    float3 kv_sum = add(kv1, add(mul(kv2, 2.0f), add(mul(kv3, 2.0f), kv4)));
    float3 kp_sum = add(kp1, add(mul(kp2, 2.0f), add(mul(kp3, 2.0f), kp4)));

    v = add(v, mul(kv_sum, h / 6.0f));
    p = add(p, mul(kp_sum, h / 6.0f));
}

#endif
