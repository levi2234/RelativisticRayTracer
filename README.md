# Relativistic Ray Tracer

A high-performance, real-time GPU-accelerated raytracer for visualizing black holes using General Relativity. This project simulates the extreme gravitational lensing, Doppler beaming, and volumetric accretion disk physics around a Kerr (spinning) black hole.

![Black Hole Simulation](assets/skyboxes/skybox2.jpg) *Note: Add a recording thumbnail here*

## Features

### 🌌 Relativistic Physics
- **Kerr Metric**: Simulates frame-dragging and the unique light paths around a rotating singularity.
- **Gravitational Lensing**: Accurate light bending from the photon sphere to the shadow.
- **Radiative Transfer**: Simulates Doppler beaming and gravitational redshift, shifting colors based on the gas velocity and proximity to the event horizon.

### 💨 Volumetric Rendering
- **Accretion Disk**: High-fidelity gas simulation using multi-scale Ridge noise and domain warping for a "shredded" cinematic look.
- **Dust Clouds**: Wispy, turbulent filaments with nested swirled domain warping for organic detail.
- **Protrusion Masking**: Organic 3D "puffs" that break the uniform disk profile.

### 🎬 Cinematic Tools
- **Fixed-Step Simulation Clock**: Ensures perfectly smooth video output even if the GPU framerate fluctuates.
- **Keyframe Path System**: Modular camera paths using Catmull-Rom splines for smooth, drone-like movement.
- **Screen Recording**: Direct integration with FFmpeg for high-quality H.264 MP4 output.

### 📸 Camera Effects
- **Bloom**: Soft radiance glow from white-hot gas.
- **Lens Distortion**: Barrel distortion to simulate wide-angle cinematic lenses.
- **Chromatic Aberration**: Lens color fringing based on gravitational intensity.
- **Vignette**: Artistic edge-darkening for focus.

## Controls

### Camera
- **W/A/S/D**: Fly movement
- **Space/Shift**: Move Up/Down
- **Mouse**: Look around
- **ESCAPE**: Exit

### Path & Recording
- **R**: Start/Stop Screen Recording (saves to `.mp4`)
- **P**: Toggle Camera Path playback
- **N**: Cycle to Next camera path

### Visual Effects
- **B**: Toggle Bloom
- **L**: Toggle Lens Distortion
- **C**: Toggle Chromatic Aberration
- **V**: Toggle Vignette

## Installation & Building

### Prerequisites
- **NVIDIA GPU** with CUDA support.
- **CUDA Toolkit** (v11.0 or higher recommended).
- **FFmpeg** (installed and added to your System PATH for recording).
- **CMake** (v3.18 or higher).

### Build Instructions
1. Clone the repository.
2. Open the project in Cursor or your favorite IDE.
3. Configure and build using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```
4. Run the executable found in `build/Release/`.

## Project Structure
- `include/`: C++ and CUDA header files.
- `src/`: Core implementation files (`main.cpp`, `raymarcher.cu`, etc.).
- `camera_effects/`: Post-processing logic and settings.
- `assets/`: Skybox textures and environment maps.
- `external/`: GLAD and other small dependencies.

## Acknowledgements
Inspired by the physics of *Interstellar* (Gargantua) and based on the Kerr metric for rotating black holes.

