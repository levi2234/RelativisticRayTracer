#include "raymarcher.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * --- RELATIVISTIC ACCRETION DISK INTEGRATOR ---
 * 
 * This kernel simulates light paths near a Schwarzschild black hole.
 * Instead of simple spheres, we now render a "Thin Accretion Disk" 
 * to visualize the extreme gravitational lensing (the "Interstellar" look).
 */

// --- CONSTANTS ---
#define MAX_STEPS 1000      // More steps for higher precision in the curves
#define STEP_SIZE 0.04f     // Smaller steps for better accuracy
#define EVENT_HORIZON 1.0f  // Schwarzschild radius (rs)
#define MASS_POS make_float3(0.0f, 0.0f, 10.0f) 

// Disk dimensions (multiples of rs)
#define DISK_INNER 2.2f     
#define DISK_OUTER 7.0f

// --- MATH HELPERS ---

__device__ float dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

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

__device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 mul(float3 v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

// --- RENDER KERNEL ---

__global__ void raymarch_kernel(uchar4* output, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Viewport [-1, 1] with aspect correction
    float u = (float)x / width * 2.0f - 1.0f;
    float v = (float)y / height * 2.0f - 1.0f;
    float aspect = (float)width / height;
    u *= aspect;

    // Camera setup: tilting slightly to see the disk better
    float3 ro = make_float3(0.0f, 1.5f, 0.0f);           
    float3 rd = normalize(make_float3(u, v - 0.15f, 1.0f)); 
    
    float3 p = ro;
    float3 vel = rd;

    bool hit_horizon = false;
    bool hit_disk = false;
    float disk_r = 0.0f; // Stores radius from center when hitting disk

    // Geodesic Integration Loop
    for(int i = 0; i < MAX_STEPS; i++) {
        float3 rel_p = sub(p, MASS_POS);
        float r2 = dot(rel_p, rel_p);
        float r = sqrtf(r2);

        // 1. Event Horizon Check
        if (r < EVENT_HORIZON * 1.01f) {
            hit_horizon = true;
            break;
        }

        // 2. Accretion Disk Check (Thin plane at Y=0 relative to MASS_POS)
        // Check if ray crossed the Y=0 plane between this step and previous
        float prev_y = rel_p.y;
        
        // Relativistic Deflection Force
        float3 L_vec = cross(rel_p, vel);
        float L2 = dot(L_vec, L_vec);
        float acc_mag = -1.5f * EVENT_HORIZON * L2 / (r2 * r2 * r);
        float3 acc = mul(rel_p, acc_mag);

        // Step
        vel = add(vel, mul(acc, STEP_SIZE));
        p = add(p, mul(vel, STEP_SIZE));
        
        float3 next_rel_p = sub(p, MASS_POS);
        
        // If we crossed the plane (y sign change)
        if ((prev_y > 0 && next_rel_p.y < 0) || (prev_y < 0 && next_rel_p.y > 0)) {
            // Roughly estimate the intersection radius
            float t = fabsf(prev_y) / (fabsf(prev_y) + fabsf(next_rel_p.y));
            float3 intersect = add(rel_p, mul(sub(next_rel_p, rel_p), t));
            float ir2 = dot(intersect, intersect);
            float ir = sqrtf(ir2);
            
            if (ir > DISK_INNER && ir < DISK_OUTER) {
                hit_disk = true;
                disk_r = ir;
                break;
            }
        }

        // 3. Escape check
        if (r > 60.0f && dot(rel_p, vel) > 0) break;
    }

    // --- SHADING ---
    unsigned char r_col, g_col, b_col;
    
    if (hit_horizon) {
        r_col = 0; g_col = 0; b_col = 0;
    } else if (hit_disk) {
        // Glowing hot disk shading
        // Hotter (brighter/whiter) near the center, redder/dimmer further out
        float norm_r = (disk_r - DISK_INNER) / (DISK_OUTER - DISK_INNER);
        float heat = powf(1.0f - norm_r, 2.0f);
        
        r_col = (unsigned char)(255);
        g_col = (unsigned char)(heat * 200 + 50);
        b_col = (unsigned char)(heat * heat * 150);
        
        // Add some "texture" to the disk based on angle (very simple)
        float3 rel_p = sub(p, MASS_POS);
        float angle = atan2f(rel_p.z, rel_p.x);
        float noise = 0.8f + 0.2f * sinf(angle * 20.0f + norm_r * 50.0f);
        r_col = (unsigned char)(r_col * noise);
        g_col = (unsigned char)(g_col * noise);
        b_col = (unsigned char)(b_col * noise);
        
    } else {
        // Starfield
        float3 dir = normalize(vel);
        float stars = powf(fmaxf(0.0f, sinf(dir.x * 60.0f) * sinf(dir.y * 60.0f) * sinf(dir.z * 60.0f)), 25.0f);
        unsigned char s = stars > 0.4f ? 255 : 0;
        
        r_col = s; g_col = s; b_col = s + 30;
    }

    output[(height - 1 - y) * width + x] = make_uchar4(r_col, g_col, b_col, 255);
}

void launch_raymarch(uchar4* d_out, int w, int h, float time) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    raymarch_kernel<<<grid, block>>>(d_out, w, h, time);
}
