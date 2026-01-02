#ifndef RAYMARCHER_H
#define RAYMARCHER_H

#include <vector_types.h> // For uchar4, float3

/**
 * Camera state to be passed from C++ (Host) to CUDA (Device).
 * Defines the camera position and its orthonormal basis vectors.
 */
struct CameraState {
    float3 pos;
    float3 forward;
    float3 right;
    float3 up;
};

// Wrapper function to launch the CUDA kernel
void launch_raymarch(uchar4* d_out, int w, int h, float time, CameraState cam);

#endif
