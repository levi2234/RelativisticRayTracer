#include "camera_paths.h"
#include <cmath>
#include <cuda_runtime.h>

// Catmull-Rom spline interpolation for smooth movement
float3 catmull_rom(float3 p0, float3 p1, float3 p2, float3 p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    auto f = [t, t2, t3](float a, float b, float c, float d) {
        return 0.5f * ((2.0f * b) +
                       (-a + c) * t +
                       (2.0f * a - 5.0f * b + 4.0f * c - d) * t2 +
                       (-a + 3.0f * b - 3.0f * c + d) * t3);
    };

    float3 result;
    result.x = f(p0.x, p1.x, p2.x, p3.x);
    result.y = f(p0.y, p1.y, p2.y, p3.y);
    result.z = f(p0.z, p1.z, p2.z, p3.z);
    return result;
}

// Seamless angle interpolation
float lerp_angle(float a, float b, float t) {
    float diff = fmodf(b - a + 180.0f, 360.0f) - 180.0f;
    if (diff < -180.0f) diff += 360.0f;
    return a + diff * t;
}

void initDefaultPaths() {
    // --- PATH 1: THE GARGANTUA FLY-BY ---
    CameraPath gargantua;
    gargantua.name = "Gargantua Fly-By";
    // Cinematic path that dives toward the disk and shears past the horizon
    gargantua.keyframes = {
        {0.0f,  {0.0f, 15.0f, -80.0f},  0.0f,    -10.6f}, // High approach
        {6.0f,  {15.0f, 3.0f, -30.0f},  -26.6f,  -5.1f},  // Entering disk zone
        {12.0f, {35.0f, 0.8f, 10.0f},   -106.0f, -1.2f},  // Side shear pass
        {18.0f, {5.0f, 1.5f, 50.0f},    -174.3f, -1.7f},  // Looking back
        {25.0f, {-20.0f, 12.0f, 70.0f}, -196.0f, -9.3f}   // Pulling away
    };
    PathManager::instance().registerPath(gargantua);

    // --- PATH 2: EVENT HORIZON FOCUS ---
    CameraPath orbit;
    orbit.name = "Event Horizon Focus";
    // A tight, slow orbit to showcase lensing
    orbit.keyframes = {
        {0.0f,  {40.0f, 2.0f, 0.0f}, -90.0f, 0.0f},
        {8.0f,  {0.0f, 5.0f, 40.0f}, -180.0f, -5.0f},
        {16.0f, {-40.0f, 2.0f, 0.0f}, -270.0f, 0.0f},
        {24.0f, {0.0f, -5.0f, -40.0f}, -360.0f, 5.0f},
        {32.0f, {40.0f, 2.0f, 0.0f}, -450.0f, 0.0f}
    };
    PathManager::instance().registerPath(orbit);

    // --- PATH 3: HORIZON SKIMMER (Disk Exploration) ---
    CameraPath skimmer;
    skimmer.name = "Horizon Skimmer";
    // Tight, low-altitude pass over the disk into the high-lensing inner region
    // Now with mathematically calculated look-at angles to keep the BH centered
    skimmer.keyframes = {
        {0.0f,  {0.0f, 3.0f, -40.0f},   0.0f,    -4.3f},  // Approach low over the "receding" side
        {7.0f,  {25.0f, 1.0f, -15.0f},  -59.0f,  -2.0f},  // Skimming the outer disk filaments
        {14.0f, {12.0f, 0.4f, 5.0f},    -112.4f, -1.7f},  // High-speed pass near the ISCO
        {21.0f, {3.5f, 0.2f, 0.0f},     -90.0f,  -3.3f},  // Extreme lensing: skimming just above photon sphere
        {28.0f, {-5.0f, 1.0f, -8.0f},   32.0f,   -6.0f},  // Slingshot around and looking back at BH
        {35.0f, {-20.0f, 5.0f, -30.0f}, 33.7f,   -8.0f}   // Ascending out of the gravity well
    };
    PathManager::instance().registerPath(skimmer);
}

