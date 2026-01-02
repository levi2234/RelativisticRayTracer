#include "raymarcher.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * --- RAYMARCHING CORE ---
 * 
 * This file contains the GPU kernels that perform the actual raymarching logic.
 * Raymarching (specifically Sphere Tracing) works by "stepping" along a ray
 * by the distance to the nearest object in the scene.
 */

// --- CONSTANTS ---
#define MAX_STEPS 100      // Maximum number of steps per ray
#define SURF_DIST 0.001f   // Distance considered a "hit" on a surface
#define MAX_DIST 100.0f    // Distance at which we give up (sky)

// --- MATH HELPERS ---

// Calculates the magnitude of a 3D vector
__device__ float length(float3 v) {
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

// Returns a unit-length version of the input vector
__device__ float3 normalize(float3 v) {
    float mag = length(v);
    return make_float3(v.x / mag, v.y / mag, v.z / mag);
}

// Vector subtraction: a - b
__device__ float3 sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// --- SDF (Signed Distance Function) primitives ---
// These return the distance from point 'p' to the surface of the shape.
// Positive = outside, Negative = inside, Zero = on surface.

// Distance to a sphere of radius 's' centered at origin
__device__ float sdSphere(float3 p, float s) {
    return length(p) - s;
}

// Distance to a box of size 'b' centered at origin
__device__ float sdBox(float3 p, float3 b) {
    float3 q = make_float3(fabsf(p.x) - b.x, fabsf(p.y) - b.y, fabsf(p.z) - b.z);
    float outside = length(make_float3(fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f)));
    float inside = fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
    return outside + inside;
}

// --- SCENE DEFINITION ---
// This function combines all objects into a single mathematical field.
__device__ float sceneSDF(float3 p, float time) {
    // 1. Moving Sphere: Animated position using sine/cosine
    float3 spherePos = make_float3(sinf(time) * 1.5f, cosf(time * 0.5f) * 0.5f, 5.0f);
    float sphere = sdSphere(sub(p, spherePos), 1.0f);
    
    // 2. Stationary Box: Fixed at center-bottom
    float3 boxPos = make_float3(0.0f, -1.5f, 5.0f);
    float box = sdBox(sub(p, boxPos), make_float3(0.7f, 0.7f, 0.7f));
    
    // 3. Boolean Union: fmin returns the distance to the closest object
    return fminf(sphere, box);
}

// --- RENDER KERNEL ---
// This runs once for every single pixel on the screen.
__global__ void raymarch_kernel(uchar4* output, int width, int height, float time) {
    // Determine which pixel this thread is responsible for
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bounds check
    if (x >= width || y >= height) return;

    // Normalize coordinates: Convert pixel (0..width) to (-1..1) range
    float u = (float)x / width * 2.0f - 1.0f;
    float v = (float)y / height * 2.0f - 1.0f;
    
    // Primary aspect ratio correction
    float aspect = (float)width / height;
    u *= aspect;

    // Ray Setup: Camera is at (0,0,0) looking forward along Z axis
    float3 ro = make_float3(0.0f, 0.0f, 0.0f);           // Ray Origin
    float3 rd = normalize(make_float3(u, v, 1.0f));     // Ray Direction

    // Raymarching Loop: "Sphere Tracing"
    float dO = 0.0f; // Total distance traveled
    bool hit = false;
    for(int i = 0; i < MAX_STEPS; i++) {



        // Integrate the ray
        float3 p = make_float3(ro.x + rd.x * dO, ro.y + rd.y * dO, ro.z + rd.z * dO);
        
        // Get distance to nearest object in the scene
        float dS = sceneSDF(p, time); 
        
        dO += dS; // Integrate the ray
        
        // Check for intersection or "sky" limit
        if(dS < SURF_DIST || dO > MAX_DIST) {
            if(dS < SURF_DIST) hit = true;
            break;
        }
    }

    // Shading: Determine the pixel color based on whether we hit anything
    unsigned char r, g, b;
    if(hit) {
        // Simple distance-based lighting (darker as it gets further away)
        float brightness = 1.0f / (1.0f + dO * 0.1f);
        r = (unsigned char)(brightness * 180 + 30);
        g = (unsigned char)(brightness * 120 + 20);
        b = (unsigned char)(brightness * 230 + 25);
    } else {
        // Procedural Sky Backdrop
        float bg = 0.5f * (v + 1.0f); // Gradient base
        r = (unsigned char)(10 + bg * 20);
        g = (unsigned char)(10 + bg * 20);
        b = (unsigned char)(30 + bg * 50);
    }

    // Write to Output: 
    // We flip Y because OpenGL's texture coordinate (0,0) is bottom-left,
    // but our kernel's y index typically starts from the top.
    output[(height - 1 - y) * width + x] = make_uchar4(r, g, b, 255);
}

// --- HOST WRAPPER ---
// This part runs on the CPU and launches the GPU kernel.
void launch_raymarch(uchar4* d_out, int w, int h, float time) {
    // Define thread block size (16x16 = 256 threads)
    dim3 block(16, 16);
    
    // Calculate grid size to cover the entire window
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    // Launch the kernel on the GPU
    raymarch_kernel<<<grid, block>>>(d_out, w, h, time);
    
    // Post-launch error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Diagnostic printing could be added here if needed
    }
}
