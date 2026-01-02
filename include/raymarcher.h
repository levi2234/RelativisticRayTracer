#ifndef RAYMARCHER_H
#define RAYMARCHER_H

#include <vector_types.h> // For uchar4

// This function will be called by main.cpp to trigger the GPU work
void launch_raymarch(uchar4* d_out, int w, int h, float time);

#endif
