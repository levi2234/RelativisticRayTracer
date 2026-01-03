#include "raymarcher.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

#include "config.h"
#include "math_utils.h"
#include "densities.h"
#include "geodesics.h"
#include "integrators.h"

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
            float d_disk = in_disk_zone ? getAccretionDensity(rel_p, time) : 0.0f; 
            float d_cloud = in_cloud_zone ? getDustCloudDensity(rel_p, time) : 0.0f; 

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
                    float g = calculateRedshiftFactor(rel_p, vel);
                    float lighting = 0.5f + 3.0f * powf(ISCO_RADIUS / fmaxf(r, ISCO_RADIUS), 1.2f);
                    float cloud_I = d_cloud * CLOUD_LUMINOSITY * lighting;

                    // --- REDSHIFT COLOR GRADING ---
                    // g > 1.0: Blue-shift (approaching), g < 1.0: Red-shift (receding)
                    // We map g to a color shift to make the disk less uniform
                    float shift = smoothstep(0.7f, 1.3f, g);
                    
                    // Base color is greyish blue
                    float3 base_color = make_float3(0.60f, 0.65f, 0.80f);
                    
                    // Shift: Approaching (Blue/White) vs Receding (Deep Red/Grey)
                    step_emit.x += base_color.x * cloud_I * lerp(1.2f, 0.8f, shift);
                    step_emit.y += base_color.y * cloud_I * lerp(0.8f, 1.1f, shift);
                    step_emit.z += base_color.z * cloud_I * lerp(0.6f, 1.4f, shift);

                    step_opacity += d_cloud * CLOUD_OPACITY;
                }

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

    float out_r = intensity_r + bg_color.x * transmittance;
    float out_g = intensity_g + bg_color.y * transmittance;
    float out_b = intensity_b + bg_color.z * transmittance;

    // Tone mapping
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
