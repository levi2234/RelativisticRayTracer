#ifndef CAMERA_SETTINGS_H
#define CAMERA_SETTINGS_H

struct CameraEffects {
    bool useBloom = true;
    float bloomThreshold = 0.8f;
    float bloomIntensity = 0.5f;
    
    bool useVignette = true;
    float vignetteIntensity = 0.4f;
    
    bool useChromaticAberration = false;
    float caAmount = 0.005f;

    bool useLensDistortion = true;
    float distortionAmount = 0.15f;
};

#endif

