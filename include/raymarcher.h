#ifndef RAYMARCHER_H
#define RAYMARCHER_H

#include <vector_types.h>
#include <texture_types.h>

/**
 * Camera state to be passed from C++ (Host) to CUDA (Device).
 */
struct CameraState {
    float3 pos;
    float3 forward;
    float3 right;
    float3 up;
};

// Wrapper function to launch the CUDA kernel
// Added skyboxTex parameter for the texture skybox
void launch_raymarch(uchar4* d_out, int w, int h, float time, CameraState cam, cudaTextureObject_t skyboxTex);

#endif
