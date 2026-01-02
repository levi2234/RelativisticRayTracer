#include "raymarcher.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * --- RELATIVISTIC ACCRETION DISK (SDF VERSION) ---
 * 
 * This engine simulates light bending around a black hole.
 * The accretion disk is defined as an SDF in the scene.
 */

// --- GLOBAL CONSTANTS ---
#define MAX_STEPS 1200      
#define STEP_SIZE 0.04f     
#define SURF_DIST 0.015f    
#define EVENT_HORIZON 1.0f  // Schwarzschild radius
#define MASS_POS make_float3(0.0f, 0.0f, 10.0f) 

// --- MATH HELPERS ---

__device__ float dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ float length(float3 v) {
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

__device__ float3 normalize(float3 v) {
    float mag = length(v);
    if (mag < 1e-6f) return make_float3(0,0,0);
    return make_float3(v.x / mag, v.y / mag, v.z / mag);
}

__device__ float3 sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 mul(float3 v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

// --- SDF PRIMITIVES ---

__device__ float sdTorus(float3 p, float2 t) {
    float2 q = make_float2(length(make_float3(p.x, 0, p.z)) - t.x, p.y);
    return sqrtf(q.x * q.x + q.y * q.y) - t.y;
}

// --- SCENE DEFINITION ---

__device__ float sceneSDF(float3 p, float time) {
    float3 p_rel = sub(p, MASS_POS);
    return sdTorus(p_rel, make_float2(4.5f, 0.05f));
}

// --- RENDER KERNEL ---

__global__ void raymarch_kernel(uchar4* output, int width, int height, float time, CameraState cam) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Viewport normalization [-1, 1]
    float u = (float)x / width * 2.0f - 1.0f;
    float v = (float)y / height * 2.0f - 1.0f;
    float aspect = (float)width / height;
    u *= aspect;

    // Initial Path using Camera state
    float3 p = cam.pos;
    // Ray direction = forward + u*right + v*up
    float3 rd = normalize(add(cam.forward, add(mul(cam.right, u), mul(cam.up, v))));
    float3 vel = rd;
    
    bool hit_horizon = false;
    bool hit_scene = false;
    float total_dist = 0.0f;

    for(int i = 0; i < MAX_STEPS; i++) {
        float3 rel_p = sub(p, MASS_POS);
        float r2 = dot(rel_p, rel_p);
        float r = sqrtf(r2);

        // 1. Black Hole Horizon
        if (r < EVENT_HORIZON * 1.01f) {
            hit_horizon = true;
            break;
        }

        // --- SCHWARZSCHILD BENDING ---
        float3 L_vec = cross(rel_p, vel);
        float L2 = dot(L_vec, L_vec);
        float acc_mag = -1.5f * EVENT_HORIZON * L2 / (r2 * r2 * r);
        float3 acc = mul(rel_p, acc_mag);

        // Physics step
        vel = add(vel, mul(acc, STEP_SIZE));
        p = add(p, mul(vel, STEP_SIZE));
        total_dist += STEP_SIZE;
        
        // 2. Scene SDF Collision
        float dS = sceneSDF(p, time);
        if (dS < SURF_DIST) {
            hit_scene = true;
            break;
        }

        // 3. Escape check
        if (r > 60.0f && dot(rel_p, vel) > 0) break;
    }

    // --- SHADING ---
    unsigned char r_col = 0, g_col = 0, b_col = 0;
    
    if (hit_horizon) {
        // Black Hole
    } else if (hit_scene) {
        float3 rel_p = sub(p, MASS_POS);
        float r = length(rel_p);
        float norm_r = fabsf(r - 4.5f) / 2.0f;
        float heat = powf(fmaxf(0.0f, 1.0f - norm_r), 2.0f);
        
        r_col = 255;
        g_col = (unsigned char)(heat * 180 + 50);
        b_col = (unsigned char)(heat * heat * 120);
        
        float angle = atan2f(rel_p.z, rel_p.x);
        float noise = 0.9f + 0.1f * sinf(angle * 30.0f);
        r_col = (unsigned char)(r_col * noise);
        g_col = (unsigned char)(g_col * noise);
        b_col = (unsigned char)(b_col * noise);
    } else {
        float3 dir = normalize(vel);
        float stars = powf(fmaxf(0.0f, sinf(dir.x * 50.0f) * sinf(dir.y * 50.0f) * sinf(dir.z * 50.0f)), 25.0f);
        unsigned char s = stars > 0.4f ? 255 : 20;
        r_col = s; g_col = s; b_col = s + 40;
    }

    output[(height - 1 - y) * width + x] = make_uchar4(r_col, g_col, b_col, 255);
}

void launch_raymarch(uchar4* d_out, int w, int h, float time, CameraState cam) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    raymarch_kernel<<<grid, block>>>(d_out, w, h, time, cam);
}
