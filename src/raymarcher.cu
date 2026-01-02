#include "raymarcher.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * --- RELATIVISTIC SKYBOX TRAVERSE ---
 * 
 * This engine simulates light bending around a black hole with a texture skybox.
 * Light rays that escape to infinity sample an equirectangular texture.
 */

// --- GLOBAL CONSTANTS ---
#define MAX_STEPS 1200      
#define STEP_SIZE 0.04f     
#define SURF_DIST 0.015f    
#define EVENT_HORIZON 1.0f  
#define MASS_POS make_float3(0.0f, 0.0f, 10.0f) 

#define PI 3.1415926535f

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

__device__ float sdAccretionDiskAdvanced(float3 p, float innerR, float outerR, float thickness) {
    float d2 = length(make_float3(p.x, 0, p.z));
    
    // Determine how far we are across the disk (0.0 to 1.0)
    // Fix: CUDA doesn't have a built-in 'clamp' for floats, use fminf/fmaxf
    float val = (d2 - innerR) / (outerR - innerR);
    float factor = fminf(fmaxf(val, 0.0f), 1.0f);
    
    // Use a smoothstep or power function to define the "slope"
    // This makes the disk very thin at the edges and fat at the center
    float currentThickness = thickness * powf(1.0f - factor, 2.0f); 
    
    // Distance to the flat plane
    float distToPlane = fabsf(p.y) - currentThickness;
    
    // Distance to the radial bounds (the ring shape)
    float distToRing = fmaxf(innerR - d2, d2 - outerR);
    
    // Combine them (standard box-style composition)
    return fmaxf(distToPlane, distToRing);
}


// --- SCENE DEFINITION ---

__device__ float sceneSDF(float3 p, float time) {
    float3 p_rel = sub(p, MASS_POS);
    return sdAccretionDiskAdvanced(p_rel, 3.0f, 5.5f, 0.05f);
}

// --- RENDER KERNEL ---

__global__ void raymarch_kernel(uchar4* output, int width, int height, float time, CameraState cam, cudaTextureObject_t skyboxTex) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u_coord = (float)x / width * 2.0f - 1.0f;
    float v_coord = (float)y / height * 2.0f - 1.0f;
    float aspect = (float)width / height;
    u_coord *= aspect;

    float3 p = cam.pos;
    float3 rd = normalize(add(cam.forward, add(mul(cam.right, u_coord), mul(cam.up, v_coord))));
    float3 vel = rd;
    
    bool hit_horizon = false;
    bool hit_scene = false;

    // Relativistic integration
    for(int i = 0; i < MAX_STEPS; i++) {
        float3 rel_p = sub(p, MASS_POS);
        float r2 = dot(rel_p, rel_p);
        float r = sqrtf(r2);

        if (r < EVENT_HORIZON * 1.01f) {
            hit_horizon = true;
            break;
        }

        float3 L_vec = cross(rel_p, vel);
        float L2 = dot(L_vec, L_vec);
        float acc_mag = -1.5f * EVENT_HORIZON * L2 / (r2 * r2 * r);
        float3 acc = mul(rel_p, acc_mag);

        vel = add(vel, mul(acc, STEP_SIZE));
        p = add(p, mul(vel, STEP_SIZE));
        
        float dS = sceneSDF(p, time);
        if (dS < SURF_DIST) {
            hit_scene = true;
            break;
        }

        if (r > 60.0f && dot(rel_p, vel) > 0) break;
    }

    // --- SHADING ---
    uchar4 color;
    
    if (hit_horizon) {
        color = make_uchar4(0, 0, 0, 255);
    } else if (hit_scene) {
        float3 rel_p = sub(p, MASS_POS);
        float r = length(rel_p);
        float norm_r = fabsf(r - 4.5f) / 2.0f;
        float heat = powf(fmaxf(0.0f, 1.0f - norm_r), 2.0f);
        
        unsigned char r_c = 255;
        unsigned char g_c = (unsigned char)(heat * 180 + 50);
        unsigned char b_c = (unsigned char)(heat * heat * 120);
        
        float angle = atan2f(rel_p.z, rel_p.x);
        float noise = 0.9f + 0.1f * sinf(angle * 30.0f);
        color = make_uchar4((unsigned char)(r_c * noise), (unsigned char)(g_c * noise), (unsigned char)(b_c * noise), 255);
    } else {
        // Sample Skybox Texture
        float3 d = normalize(vel);
        
        // Equirectangular mapping
        float phi = atan2f(d.z, d.x);
        float theta = asinf(d.y);
        
        float tx = 0.5f + phi / (2.0f * PI);
        float ty = 0.5f - theta / PI;
        
        float4 texColor = tex2D<float4>(skyboxTex, tx, ty);
        color = make_uchar4((unsigned char)(texColor.x * 255), (unsigned char)(texColor.y * 255), (unsigned char)(texColor.z * 255), 255);
    }

    output[(height - 1 - y) * width + x] = color;
}

void launch_raymarch(uchar4* d_out, int w, int h, float time, CameraState cam, cudaTextureObject_t skyboxTex) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    raymarch_kernel<<<grid, block>>>(d_out, w, h, time, cam, skyboxTex);
}
