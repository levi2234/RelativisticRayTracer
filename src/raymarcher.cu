#include "raymarcher.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * --- RELATIVISTIC RADIATIVE TRANSFER ENGINE (PHYSICS TOOL VERSION) ---
 * 
 * Target: Sagittarius A* (Supermassive Black Hole)
 * Units: Geometric (G=c=1), scaled to Mass M. 
 * Rs (Schwarzschild radius) = 2.0M.
 */

// --- PHYSICAL CONSTANTS (SI Units) ---
#define C_LIGHT 299792458.0f             // [m/s] Speed of light
#define G_CONSTANT 6.67430e-11f          // [m^3 kg^-1 s^-2] Gravitational constant
#define SOLAR_MASS 1.98847e30f           // [kg] Mass of the Sun

// --- TARGET OBJECT: SAGITTARIUS A* ---
#define BH_MASS_SOLAR 4.154e6f           // [M_sun] Mass in solar masses
#define DISK_TEMP_REF 1.5e7f             // [K] Reference temperature of gas

// --- NEW KERR PARAMETERS ---
#define SPIN_A 0.4f                     // [0.0 to 1.0] Dimensionless spin parameter
#define SPIN_AXIS make_float3(0, 1, 0)   // Rotation around Y-axis

// --- SIMULATION SCALING (Geometric Units G=c=1) ---
// Mass in meters: M = G*Mass/c^2
#define M_UNIT (G_CONSTANT * (BH_MASS_SOLAR * SOLAR_MASS) / (C_LIGHT * C_LIGHT)) // [m]

// Simulation Units: 1.0 = M (The mass of the BH)
#define EVENT_HORIZON 2.0f               // [M] Schwarzschild radius Rs = 2M
#define MASS_POS make_float3(0.0f, 0.0f, 60.0f) // [M] Black hole position

// Physics & Aesthetic Tuning
#define ISCO_RADIUS 10.0f                 // [M] Innermost stable orbit (6M for non-spinning)
#define DISK_OUT_M 25.0f                // [M] Outer radius of disk
#define DISK_H_M 0.8f                    // [M] Maximum disk thickness
#define DISK_LUMINOSITY 6.0f            // [Dimensionless] Emission gain factor
#define DISK_OPACITY 0.4f                // [1/M] Absorption coefficient
#define EXPOSURE 0.8f                    // [Dimensionless] Tone mapping exposure

// --- DUST CLOUD LAYER PARAMS ---
#define CLOUD_H_M 1.2f                  // Much closer to disk thickness (was 5.0)
#define CLOUD_OUT_M 25.0f                // Perfectly matches DISK_OUT_M (was 28.0)
#define CLOUD_OPACITY 0.45f              // Higher opacity for dense dust streaks
#define CLOUD_LUMINOSITY 2.0f            // Highlights for the wisps

// Integration Quality
#define STEP_SIZE_M 0.3f                // [M] Integration step size in vacuum
#define MAX_STEPS 2000                   // [Steps] Max ray steps


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

// --- NOISE HELPERS ---

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float hash31(float3 p) {
    float3 p3 = make_float3(fmodf(p.x * 0.1031f, 1.0f), fmodf(p.y * 0.1031f, 1.0f), fmodf(p.z * 0.1031f, 1.0f));
    float d = p3.x * (p3.y + 33.33f) + p3.y * (p3.z + 33.33f) + p3.z * (p3.x + 33.33f);
    p3.x += d; p3.y += d; p3.z += d;
    return fmodf((p3.x + p3.y) * p3.z, 1.0f);
}

__device__ float noise3D(float3 p) {
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

__device__ float fbm(float3 p, int octaves) {
    float v = 0.0f;
    float a = 0.5f;
    for (int i = 0; i < octaves; ++i) {
        v += a * noise3D(p);
        p = make_float3(p.x * 2.0f + 100.0f, p.y * 2.0f + 100.0f, p.z * 2.0f + 100.0f);
        a *= 0.5f;
    }
    return v;
}

// --- PHYSICS FUNCTIONS ---

/**
 * Temperature profile (Shakura-Sunyaev)
 * T(r) = T_ref * (r/ISCO)^-0.75
 */
__device__ float getDiskTemperature(float r) {
    if (r < ISCO_RADIUS) return 0.0f;
    return DISK_TEMP_REF * powf(r / ISCO_RADIUS, -0.75f);
}

/**
 * Calculates local gas density with rotating high-fidelity clouds.
 */
__device__ float getAccretionDensity(float3 p, float time) {
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
    // direction matches the gas velocity used in calculateRedshiftFactor (CCW)
    float omega = 3.5f * powf(ISCO_RADIUS / r, 1.5f); 
    float angle_rotated = phi - time * omega;

    // Use a rotating coordinate system to sampling noise.
    // This ensures clouds follow circular orbits perfectly.
    float3 rot_p = make_float3(
        r * cosf(angle_rotated),
        p.y * 4.0f, // Stretch vertical sampling to make filaments flatter
        r * sinf(angle_rotated)
    );
    
    // Add "boiling" / internal evolution by shifting the noise in the revolving frame
    float evolution = time * 0.35f;
    float3 noise_coords = add(mul(rot_p, 0.45f), make_float3(0, evolution, 0));

    // Sample high-fidelity noise
    float n = fbm(noise_coords, 5); 

    // --- HIGH CONTRAST STREAKS ---
    // 1. Bias: Higher value clears out the "gas" between clouds (more vacuum)
    float cloud = fmaxf(0.0f, n - 0.32f); 
    
    // 2. Gain/Power: Increases the "sharpness" of the filament edges
    cloud = powf(cloud * 2.8f, 1.6f);
    
    // 3. Clamping: Prevent extreme density spikes
    cloud = fminf(6.0f, cloud);

    // Return density with high contrast between clouds and vacuum
    // Reduced base level to 0.02 for cleaner gaps
    return base_envelope * (0.02f + 5.0f * cloud); 
}

__device__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/**
 * Calculates local density for the large-scale "Dust Cloud" layer.
 * This layer is puffier, thicker, and slower-moving.
 */
__device__ float getDustCloudDensity(float3 p, float time) {
    float r = length(make_float3(p.x, 0.0f, p.z));
    if (r < ISCO_RADIUS * 1.0f || r > CLOUD_OUT_M*1.1) return 0.0f;

    // 1. Base Envelope: Uniform density and height throughout, with radial falloff at edge
    float local_h = CLOUD_H_M; 
    float vertical_profile = expf(-(p.y * p.y) / (2.0f * local_h * local_h + 1e-7f));
    
    // Smooth radial falloff at the outer boundary
    float radial_falloff = smoothstep(CLOUD_OUT_M * 1.1f, CLOUD_OUT_M * 0.85f, r);
    float base = vertical_profile * radial_falloff;

    // 2. Large Scale Puffiness
    float phi = atan2f(p.z, p.x);
    // Synced rotation: Match the omega and direction of the accretion disk
    float omega = 3.5f * powf(ISCO_RADIUS / fmaxf(r, ISCO_RADIUS), 1.5f);
    float angle_rot = phi - time * omega; 

    float3 rot_p = make_float3(
        r * cosf(angle_rot),
        p.y * 2.0f, // Reduced stretch for much chunkier, rougher puffiness
        r * sinf(angle_rot)
    );

    // Increase frequency for finer, rougher detail
    float3 noise_coords = add(mul(rot_p, 1.25f), make_float3(0, time * 0.25f, 0));
    
    // Domain warping: Offset the sampling coordinates by another noise layer
    // This creates the "rough", turbulent look from the reference image
    float3 warp = make_float3(
        fbm(add(noise_coords, make_float3(0.0, 0.0, 0.0)), 2),
        fbm(add(noise_coords, make_float3(5.2, 1.3, 0.0)), 2),
        fbm(add(noise_coords, make_float3(0.0, 9.4, 3.1)), 2)
    );
    
    float3 final_coords = add(noise_coords, mul(warp, 0.65f));
    float n = fbm(final_coords, 6); // High fidelity (6 octaves)
    
    // Sharpen the cloud structures into rough clusters
    float cloud = fmaxf(0.0f, n - 0.42f);
    cloud = powf(cloud * 3.5f, 2.2f); // High power factor for sharp "cloudy" peaks

    return base * fminf(7.0f, cloud);
}
/**
 * Calculates the Relativistic Beaming factor (Doppler + Gravitational Redshift)
 * g = observed_frequency / emitted_frequency
 */
__device__ float calculateRedshiftFactor(float3 p_rel, float3 ray_vel) {
    float r = length(p_rel);
    if (r < EVENT_HORIZON * 1.01f) return 0.0f;

    // Gravitational Redshift (Standard)
    float g_gravity = sqrtf(1.0f - EVENT_HORIZON / r);

    // 2. Kerr Keplerian Velocity
    // In a rotating BH, the orbital velocity is modified: v = 1 / (r^1.5 + a)
    // This prevents the ISCO from reaching v=c too early.
    float v_mag = 1.0f / (powf(r, 1.5f) + SPIN_A);
    
    // Tangential gas direction (CCW around Y)
    float3 gas_dir = normalize(make_float3(-p_rel.z, 0, p_rel.x));
    float cos_theta = dot(ray_vel, gas_dir);
    
    float gamma = 1.0f / sqrtf(1.0f - v_mag * v_mag);
    float g_doppler = 1.0f / (gamma * (1.0f - v_mag * cos_theta));

    return g_gravity * g_doppler;
}

// --- PHYSICS ENGINE ---

__device__ float3 getGeodesicAcc(float3 p_rel, float3 v) {
    float r2 = dot(p_rel, p_rel);
    float r = sqrtf(r2);
    if (r < EVENT_HORIZON * 0.5f) return make_float3(0,0,0);
    
    // 1. Schwarzschild-like Radial Bending
    // a = -1.5 * Rs * L^2 / r^5
    float3 L_vec = cross(p_rel, v);
    float L2 = dot(L_vec, L_vec);
    float radial_mag = -1.5f * EVENT_HORIZON * L2 / (r2 * r2 * r);
    float3 radial_acc = mul(p_rel, radial_mag);

    // 2. Kerr Frame Dragging Acceleration
    // This simulates the "twist" of space. In absolute time, we treat this
    // as a Coriolis-like force pulling the ray in the direction of spin.
    float3 drag_dir = cross(SPIN_AXIS, p_rel);
    // Dragging strength falls off with 1/r^3
    float drag_strength = (2.0f * SPIN_A * EVENT_HORIZON) / (r2 * r);
    float3 dragging_acc = mul(drag_dir, drag_strength);

    return add(radial_acc, dragging_acc);
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
        float current_h = STEP_SIZE_M;
        // High quality sampling near horizon and in the gas disk zone
        bool near_bh = (r < 18.0f);
        bool in_disk_zone = (fabsf(rel_p.y) < DISK_H_M * 5.0f && r < DISK_OUT_M + 5.0f);
        bool in_cloud_zone = (fabsf(rel_p.y) < CLOUD_H_M * 1.5f && r < CLOUD_OUT_M);
        
        if (near_bh) current_h *= 0.1f; 
        else if (in_disk_zone) current_h *= 0.3f;
        else if (in_cloud_zone) current_h *= 0.5f;

        integrate_rk4(p, vel, current_h);




        // --- RADIATIVE TRANSFER (INTEGRATED MEDIA) ---
        if (in_disk_zone || in_cloud_zone) {
            float d_disk = in_disk_zone ? getAccretionDensity(rel_p, time) : 0.0f; //TOGGLE TO ENABLE OR DISABLE
            float d_cloud = in_cloud_zone ? getDustCloudDensity(rel_p, time) : 0.0f; //TOGGLE TO ENABLE OR DISABLE

            if (d_disk > 0.001f || d_cloud > 0.001f) {
                float3 step_emit = make_float3(0, 0, 0);
                float step_opacity = 0;

                // 1. Accretion Disk Component
                if (d_disk > 0.001f) {
                    float g = calculateRedshiftFactor(rel_p, vel);
                    float T = getDiskTemperature(r);
                    float T_norm = powf(T / DISK_TEMP_REF, 0.5f);
                    float bol_I = powf(g, 4.0f) * T_norm * d_disk * DISK_LUMINOSITY;
                    
                    float color_t = g * powf(T / DISK_TEMP_REF, 0.4f) * 2.5f;
                    step_emit.x += 1.0f * bol_I;
                    step_emit.y += fminf(0.25f, 0.12f * color_t) * bol_I;
                    step_emit.z += fmaxf(0.0f, 0.01f * (color_t - 2.0f)) * bol_I;
                    
                    step_opacity += d_disk * DISK_OPACITY;
                }

                // 2. Dust Cloud Component (Interleaved)
                if (d_cloud > 0.001f) {
                    // Proximal lighting: clouds near the disk glow whiter/brighter
                    float lighting = 0.5f + 3.0f * powf(ISCO_RADIUS / fmaxf(r, ISCO_RADIUS), 1.2f);
                    float cloud_I = d_cloud * CLOUD_LUMINOSITY * lighting;

                    // Reference: Pinkish-white highlights with dark purple base
                    step_emit.x += 0.95f * cloud_I;
                    step_emit.y += 0.75f * cloud_I;
                    step_emit.z += 0.85f * cloud_I;

                    step_opacity += d_cloud * CLOUD_OPACITY;
                }

                // Unified physical integration for current ray step
                float d_tau = step_opacity * current_h;
                float step_trans = expf(-d_tau);
                float factor = (1.0f - step_trans) * transmittance;

                intensity_r += step_emit.x * factor;
                intensity_g += step_emit.y * factor;
                intensity_b += step_emit.z * factor;
                
                transmittance *= step_trans;
            }
        }

        // 3. Escape to infinity
        if (r > 250.0f && dot(rel_p, vel) > 0) break;

    }

    // --- FINAL COLOR ASSEMBLY ---
    uchar4 final_color;
    
    // Background light (Skybox or Black Hole)
    float3 bg_color = make_float3(0,0,0);
    if (!hit_horizon) {
        float3 d = normalize(vel);
        float phi = atan2f(d.z, d.x);
        float theta = asinf(d.y);
        float tx = 0.5f + phi / (2.0f * PI);
        float ty = 0.5f - theta / PI;
        float4 skyColor = tex2D<float4>(skyboxTex, tx, ty);
        bg_color = make_float3(skyColor.x, skyColor.y, skyColor.z);
    }

    // Final pixel colors
    float out_r = intensity_r + bg_color.x * transmittance;
    float out_g = intensity_g + bg_color.y * transmittance;
    float out_b = intensity_b + bg_color.z * transmittance;

        
        // Tone mapping (prevent over-saturation)
        out_r = 1.0f - expf(-out_r * EXPOSURE);
        out_g = 1.0f - expf(-out_g * EXPOSURE);
        out_b = 1.0f - expf(-out_b * EXPOSURE);


        final_color = make_uchar4((unsigned char)(out_r * 255), (unsigned char)(out_g * 255), (unsigned char)(out_b * 255), 255);


    output[(height - 1 - y) * width + x] = final_color;
}

void launch_raymarch(uchar4* d_out, int w, int h, float time, CameraState cam, cudaTextureObject_t skyboxTex) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    raymarch_kernel<<<grid, block>>>(d_out, w, h, time, cam, skyboxTex);
}
