#include "raymarcher.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

// --- CONSTANTS ---
#define MAX_STEPS 100
#define SURF_DIST 0.001f
#define MAX_DIST 100.0f

// --- MATH HELPERS ---

__device__ float length(float3 v) {
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

__device__ float3 normalize(float3 v) {
    float mag = length(v);
    return make_float3(v.x / mag, v.y / mag, v.z / mag);
}

__device__ float3 sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// --- SDF primitives ---

__device__ float sdSphere(float3 p, float s) {
    return length(p) - s;
}

__device__ float sdBox(float3 p, float3 b) {
    float3 q = make_float3(fabsf(p.x) - b.x, fabsf(p.y) - b.y, fabsf(p.z) - b.z);
    float outside = length(make_float3(fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f)));
    float inside = fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
    return outside + inside;
}

// --- SCENE DEFINITION ---

__device__ float sceneSDF(float3 p, float time) {
    // Rotating sphere
    float3 spherePos = make_float3(sinf(time) * 1.5f, cosf(time * 0.5f) * 0.5f, 5.0f);
    float sphere = sdSphere(sub(p, spherePos), 1.0f);
    
    // Stationary box
    float3 boxPos = make_float3(0.0f, -1.5f, 5.0f);
    float box = sdBox(sub(p, boxPos), make_float3(0.7f, 0.7f, 0.7f));
    
    // Combine (Union)
    return fminf(sphere, box);
}

// --- RENDER KERNEL ---

__global__ void raymarch_kernel(uchar4* output, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Viewport coordinates [-1, 1]
    float u = (float)x / width * 2.0f - 1.0f;
    float v = (float)y / height * 2.0f - 1.0f;
    float aspect = (float)width / height;
    u *= aspect;

    // Ray Setup
    float3 ro = make_float3(0.0f, 0.0f, 0.0f);
    float3 rd = normalize(make_float3(u, v, 1.0f));

    // Raymarching Loop
    float dO = 0.0f; // Distance from Origin
    bool hit = false;
    for(int i = 0; i < MAX_STEPS; i++) {
        float3 p = make_float3(ro.x + rd.x * dO, ro.y + rd.y * dO, ro.z + rd.z * dO);
        float dS = sceneSDF(p, time); // Distance to Scene
        dO += dS;
        if(dS < SURF_DIST || dO > MAX_DIST) {
            if(dS < SURF_DIST) hit = true;
            break;
        }
    }

    // Shading
    unsigned char r, g, b;
    if(hit) {
        float brightness = 1.0f / (1.0f + dO * 0.1f);
        r = (unsigned char)(brightness * 180 + 30);
        g = (unsigned char)(brightness * 120 + 20);
        b = (unsigned char)(brightness * 230 + 25);
    } else {
        // Sky Gradient
        float bg = 0.5f * (v + 1.0f);
        r = (unsigned char)(10 + bg * 20);
        g = (unsigned char)(10 + bg * 20);
        b = (unsigned char)(30 + bg * 50);
    }

    // Flip Y for OpenGL texture coordinates if needed (OpenGL is bottom-up)
    output[(height - 1 - y) * width + x] = make_uchar4(r, g, b, 255);
}

// --- HOST WRAPPERS ---

void launch_raymarch(uchar4* d_out, int w, int h, float time) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    raymarch_kernel<<<grid, block>>>(d_out, w, h, time);
    
    // Safety check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Silence is gold for now, but a diagnostic could go here
    }
}
