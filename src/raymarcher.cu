#include "raymarcher.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * --- RELATIVISTIC RADIATIVE TRANSFER ENGINE ---
 * 
 * This engine simulates light passing THROUGH a gas accretion disk.
 * Instead of a solid surface, we accumulate emission and absorption
 * while accounting for Doppler boosting and Gravitational redshift.
 */

// --- GLOBAL CONSTANTS ---
#define MAX_STEPS 3000      
#define STEP_SIZE 0.08f     
#define EVENT_HORIZON 1.0f  // rs
#define MASS_POS make_float3(0.0f, 0.0f, 30.0f) 

// Disk physical dimensions
#define DISK_INNER 3.0f     
#define DISK_OUTER 16.0f
#define DISK_HEIGHT 0.15f   // Thinner disk for a more elegant appearance

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
\


// --- PHYSICS FUNCTIONS ---

/**
 * Calculates the local gas density at position p.
 * Model: Torus with radial and vertical Gaussian falloff.
 */
__device__ float getDiskDensity(float3 p_rel) {
    float r = sqrtf(p_rel.x*p_rel.x + p_rel.z*p_rel.z);
    
    // Radial bounds check
    if (r < DISK_INNER || r > DISK_OUTER) return 0.0f;
    
    // Radial density profile (peak in the middle)
    float radial_factor = expf(-powf((r - 4.5f) / 1.5f, 2.0f));
    
    // Vertical falloff (disk thickness)
    float vertical_factor = expf(-powf(p_rel.y / DISK_HEIGHT, 2.0f));
    
    return radial_factor * vertical_factor;
}

__device__ float getAccretionDensity(float3 p, float innerR, float outerR, float maxHeight) {
    float r = length(make_float3(p.x, 0, p.z));
    
    if (r < innerR || r > outerR) return 0.0f;

    float falloff = (r - innerR) / (outerR - innerR);
    float localMaxHeight = maxHeight * powf(1.0f - falloff, 1.5f);

    // --- 1. Base Density ---
    float density = expf(-(p.y * p.y) / (2.0f * localMaxHeight * localMaxHeight + 1e-7f));
    density *= (1.0f - falloff) * 8.0f; 

    return fmaxf(density, 0.0f);
}


/**
 * Calculates the Relativistic Beaming factor (Doppler + Gravitational Redshift)
 * g = observed_frequency / emitted_frequency
 */
__device__ float calculateRedshiftFactor(float3 p_rel, float3 ray_vel) {
    float r = length(p_rel);
    if (r < EVENT_HORIZON * 1.01f) return 0.0f;

    // 1. Gravitational Redshift: z_g = 1 / sqrt(1 - rs/r)
    float g_gravity = sqrtf(1.0f - EVENT_HORIZON / r);

    // 2. Doppler Shift: Light emitted by gas in circular Keplerian orbit
    // Orbit velocity v = sqrt(rs / (2r)) for Schwarzschild
    float v_mag = sqrtf(EVENT_HORIZON / (2.0f * r));
    
    // Unity vector in the direction of gas motion (tangential)
    float3 gas_dir = normalize(make_float3(-p_rel.z, 0, p_rel.x));
    
    // Cosine of angle between ray and gas motion
    float cos_theta = dot(ray_vel, gas_dir);
    
    // Doppler factor (Special Relativity)
    float gamma = 1.0f / sqrtf(1.0f - v_mag * v_mag);
    float g_doppler = 1.0f / (gamma * (1.0f - v_mag * cos_theta));

    // Combined g-factor
    return g_gravity * g_doppler;
}

// --- PHYSICS ENGINE ---

__device__ float3 getGeodesicAcc(float3 p_rel, float3 v) {
    float r2 = dot(p_rel, p_rel);
    float r = sqrtf(r2);
    if (r < EVENT_HORIZON * 0.5f) return make_float3(0,0,0);
    
    float3 L_vec = cross(p_rel, v);
    float L2 = dot(L_vec, L_vec);
    float acc_mag = -1.5f * EVENT_HORIZON * L2 / (r2 * r2 * r);
    return mul(p_rel, acc_mag);
}


// ----- INTEGRATORS -----

/**
 * Euler Method (1st Order)
 */
__device__ void integrate_euler(float3 &p, float3 &v, float h) {
    float3 rel_p = sub(p, MASS_POS);
    float3 acc = getGeodesicAcc(rel_p, v);
    
    p = add(p, mul(v, h));
    v = add(v, mul(acc, h));
}


/**
 * Runge-Kutta 4 (4th Order)
 */
__device__ void integrate_rk4(float3 &p, float3 &v, float h) {
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
    
    float intensity_r = 0, intensity_g = 0, intensity_b = 0;
    float transmittance = 1.0f;
    bool hit_horizon = false;

    // Relativistic Integration
    for(int i = 0; i < MAX_STEPS; i++) {
        float3 rel_p = sub(p, MASS_POS);
        float r2 = dot(rel_p, rel_p);
        float r = sqrtf(r2);

        // 1. Black hole horizon (terminate rays)
        if (r < EVENT_HORIZON * 1.01f) {
            hit_horizon = true;
            transmittance = 0.0f;
            break;
        }

        // --- GEODESIC STEP ---
        // Adaptive check: slows down inside the disk zone
        float current_h = STEP_SIZE;
        bool in_disk_zone = (fabsf(rel_p.y) < DISK_HEIGHT * 2.5f && r > DISK_INNER - 1.0f && r < DISK_OUTER + 1.0f);
        if (in_disk_zone) current_h *= 0.4f;

        integrate_rk4(p, vel, current_h);


        // --- RADIATIVE TRANSFER (GAS DISK) ---
        // Quick AABB-style check to optimize out gas calculations
        if (fabsf(rel_p.y) < DISK_HEIGHT * 3.0f && r > DISK_INNER && r < DISK_OUTER) {
            float density = getAccretionDensity(rel_p, DISK_INNER, DISK_OUTER, DISK_HEIGHT);
            
            if (density > 0.01f) {
                // Calculate local g-factor (redshift)
                float g = calculateRedshiftFactor(rel_p, vel);
                
                // Emission: I_obs = g^4 * I_emit
                // Model gas as a blackbody (red-orange)
                float emission = powf(g, 4.0f) * density;
                
                // Colors based on heat (vaguely blackbody)
                float r_emit = 1.3f;
                float g_emit = 0.6f + 0.4f * (g - 0.5f); // Shifted by Doppler
                float b_emit = 0.2f * g;

                // Simple integration (Step-wise solution to RTE)
                // Use current_h instead of STEP_SIZE for correct volume sampling
                float d_tau = density * 1.5f * current_h; 
                float step_trans = expf(-d_tau);

                
                intensity_r += r_emit * emission * (1.0f - step_trans) * transmittance;
                intensity_g += g_emit * emission * (1.0f - step_trans) * transmittance;
                intensity_b += b_emit * emission * (1.0f - step_trans) * transmittance;
                
                transmittance *= step_trans;
                
                // Early exit if the disk is fully opaque
                if (transmittance < 0.01f) break;
            }
        }

        // 3. Escape to infinity
        if (r > 64.0f && dot(rel_p, vel) > 0) break;
    }

    // --- FINAL COLOR ASSEMBLY ---
    uchar4 final_color;
    
    if (hit_horizon) {
        final_color = make_uchar4(0, 0, 0, 255);
    } else {
        // Sample Skybox and combine with disk intensity
        float3 d = normalize(vel);
        float phi = atan2f(d.z, d.x);
        float theta = asinf(d.y);
        float tx = 0.5f + phi / (2.0f * PI);
        float ty = 0.5f - theta / PI;
        
        float4 skyColor = tex2D<float4>(skyboxTex, tx, ty);
        
        // Final pixel colors
        float out_r = intensity_r + skyColor.x * transmittance;
        float out_g = intensity_g + skyColor.y * transmittance;
        float out_b = intensity_b + skyColor.z * transmittance;
        
        // Simple tone mapping (prevent over-saturation)
        out_r = 1.0f - expf(-out_r);
        out_g = 1.0f - expf(-out_g);
        out_b = 1.0f - expf(-out_b);

        final_color = make_uchar4((unsigned char)(out_r * 255), (unsigned char)(out_g * 255), (unsigned char)(out_b * 255), 255);
    }

    output[(height - 1 - y) * width + x] = final_color;
}

void launch_raymarch(uchar4* d_out, int w, int h, float time, CameraState cam, cudaTextureObject_t skyboxTex) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    raymarch_kernel<<<grid, block>>>(d_out, w, h, time, cam, skyboxTex);
}
