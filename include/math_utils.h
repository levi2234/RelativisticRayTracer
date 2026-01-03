#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cuda_runtime.h>
#include <math.h>

#define PI 3.1415926535f

// --- MATH HELPERS ---

__device__ inline float dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ inline float length(float3 v) {
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

__device__ inline float3 normalize(float3 v) {
    float mag = length(v);
    if (mag < 1e-6f) return make_float3(0,0,0);
    return make_float3(v.x / mag, v.y / mag, v.z / mag);
}

__device__ inline float3 sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 mul(float3 v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ inline float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// --- ROTATION HELPERS ---

__device__ inline float3 rotate_3d(float3 p, float3 axis, float angle) {
    float s = sinf(angle);
    float c = cosf(angle);
    float oc = 1.0f - c;
    return make_float3(
        (oc * axis.x * axis.x + c) * p.x + (oc * axis.x * axis.y - axis.z * s) * p.y + (oc * axis.z * axis.x + axis.y * s) * p.z,
        (oc * axis.x * axis.y + axis.z * s) * p.x + (oc * axis.y * axis.y + c) * p.y + (oc * axis.y * axis.z - axis.x * s) * p.z,
        (oc * axis.z * axis.x - axis.y * s) * p.x + (oc * axis.y * axis.z + axis.x * s) * p.y + (oc * axis.z * axis.z + c) * p.z
    );
}

// --- NOISE HELPERS ---

__device__ inline float3 hash33(float3 p) {
    p = make_float3(fmodf(p.x * 0.1031f, 1.0f), fmodf(p.y * 0.1031f, 1.0f), fmodf(p.z * 0.1031f, 1.0f));
    p.x += dot(p, make_float3(p.y + 33.33f, p.z + 33.33f, p.x + 33.33f));
    p.y += dot(p, make_float3(p.x + 33.33f, p.z + 33.33f, p.y + 33.33f));
    p.z += dot(p, make_float3(p.x + 33.33f, p.y + 33.33f, p.z + 33.33f));
    return make_float3(fmodf((p.x + p.y) * p.z, 1.0f), fmodf((p.x + p.z) * p.y, 1.0f), fmodf((p.y + p.z) * p.x, 1.0f));
}

__device__ inline float worley3D(float3 p) {
    float3 i = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    float3 f = make_float3(p.x - i.x, p.y - i.y, p.z - i.z);
    float minDist = 1.0f;
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                float3 neighbor = make_float3((float)x, (float)y, (float)z);
                float3 point = hash33(add(i, neighbor));
                float3 diff = sub(add(neighbor, point), f);
                float dist = length(diff);
                minDist = fminf(minDist, dist);
            }
        }
    }
    return minDist;
}

__device__ inline float hash31(float3 p) {
    float3 p3 = make_float3(fmodf(p.x * 0.1031f, 1.0f), fmodf(p.y * 0.1031f, 1.0f), fmodf(p.z * 0.1031f, 1.0f));
    float d = p3.x * (p3.y + 33.33f) + p3.y * (p3.z + 33.33f) + p3.z * (p3.x + 33.33f);
    p3.x += d; p3.y += d; p3.z += d;
    return fmodf((p3.x + p3.y) * p3.z, 1.0f);
}

__device__ inline float noise3D(float3 p) {
    float3 i = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    float3 f = make_float3(p.x - i.x, p.y - i.y, p.z - i.z);
    
    float3 u = make_float3(f.x * f.x * (3.0f - 2.0f * f.x),
                          f.y * f.y * (3.0f - 2.0f * f.y),
                          f.z * f.z * (3.0f - 2.0f * f.z));
    
    return lerp(lerp(lerp(hash31(add(i, make_float3(0, 0, 0))), hash31(add(i, make_float3(1, 0, 0))), u.x),
                    lerp(hash31(add(i, make_float3(0, 1, 0))), hash31(add(i, make_float3(1, 1, 0))), u.x), u.y),
               lerp(lerp(hash31(add(i, make_float3(0, 0, 1))), hash31(add(i, make_float3(1, 0, 1))), u.x),
                    lerp(hash31(add(i, make_float3(0, 1, 1))), hash31(add(i, make_float3(1, 1, 1))), u.x), u.y), u.z);
}

__device__ inline float fbm(float3 p, int octaves) {
    float v = 0.0f;
    float a = 0.5f;
    for (int i = 0; i < octaves; ++i) {
        v += a * noise3D(p);
        p = make_float3(p.x * 2.05f + 10.0f, p.y * 2.05f + 10.0f, p.z * 2.05f + 10.0f);
        a *= 0.5f;
    }
    return v;
}

__device__ inline float fbm_billow(float3 p, int octaves) {
    float v = 0.0f;
    float a = 0.5f;
    for (int i = 0; i < octaves; ++i) {
        float n = noise3D(p);
        v += a * (1.0f - fabsf(n * 2.0f - 1.0f));
        p = make_float3(p.x * 2.05f + 10.0f, p.y * 2.05f + 10.0f, p.z * 2.05f + 10.0f);
        a *= 0.5f;
    }
    return v;
}

#endif

