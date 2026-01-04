#ifndef CAMERA_PATHS_H
#define CAMERA_PATHS_H

#include <vector>
#include <string>
#include "raymarcher.h"

struct Keyframe {
    float time;
    float3 pos;
    float yaw;
    float pitch;
};

struct CameraPath {
    std::string name;
    std::vector<Keyframe> keyframes;
};

class PathManager {
public:
    static PathManager& instance() {
        static PathManager instance;
        return instance;
    }

    void registerPath(const CameraPath& path) {
        paths.push_back(path);
    }

    const std::vector<CameraPath>& getPaths() const {
        return paths;
    }

    const CameraPath* getPath(int index) const {
        if (index >= 0 && index < (int)paths.size()) return &paths[index];
        return nullptr;
    }

private:
    std::vector<CameraPath> paths;
};

// Helper to interpolate smoothly using Catmull-Rom splines
float3 catmull_rom(float3 p0, float3 p1, float3 p2, float3 p3, float t);
float lerp_angle(float a, float b, float t);

void initDefaultPaths();

#endif

