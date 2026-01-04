#ifndef CONFIG_H
#define CONFIG_H

#include <cuda_runtime.h>

// --- WINDOW SETTINGS ---
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080
#define RECORDING_FPS 24

// --- PHYSICAL CONSTANTS (SI Units) ---
#define C_LIGHT 299792458.0f             // [m/s] Speed of light
#define G_CONSTANT 6.67430e-11f          // [m^3 kg^-1 s^-2] Gravitational constant
#define SOLAR_MASS 1.98847e30f           // [kg] Mass of the Sun

// --- TARGET OBJECT: SAGITTARIUS A* ---
#define BH_MASS_SOLAR 4.154e6f           // [M_sun] Mass in solar masses
#define DISK_TEMP_REF 1.5e7f             // [K] Reference temperature of gas

// --- NEW KERR PARAMETERS ---
#define SPIN_A 0.0f                     // [0.0 to 1.0] Dimensionless spin parameter
#define SPIN_AXIS make_float3(0, 1, 0)   // Rotation around Y-axis

// --- SIMULATION SCALING (Geometric Units G=c=1) ---
// Mass in meters: M = G*Mass/c^2
#define M_UNIT (G_CONSTANT * (BH_MASS_SOLAR * SOLAR_MASS) / (C_LIGHT * C_LIGHT)) // [m]

// Simulation Units: 1.0 = M (The mass of the BH)
#define EVENT_HORIZON 2.0f               // [M] Schwarzschild radius Rs = 2M
#define MASS_POS make_float3(0.0f, 0.0f, 0.0f) // [M] Black hole at origin

// Physics & Aesthetic Tuning
#define ISCO_RADIUS 10.0f                 // [M] Innermost stable orbit (6M for non-spinning)
#define DISK_OUT_M 25.0f                // [M] Outer radius of disk
#define DISK_H_M 0.8f                    // [M] Maximum disk thickness
#define DISK_LUMINOSITY 6.0f            // [Dimensionless] Emission gain factor
#define DISK_OPACITY 0.4f                // [1/M] Absorption coefficient
#define EXPOSURE 0.8f                    // [Dimensionless] Tone mapping exposure

// --- DUST CLOUD LAYER PARAMS ---
#define CLOUD_H_M 0.5f                  // Much closer to disk thickness (was 5.0)
#define CLOUD_OUT_M 25.0f                // Perfectly matches DISK_OUT_M (was 28.0)
#define CLOUD_OPACITY 0.3f              // Higher opacity for dense thundercloud look
#define CLOUD_LUMINOSITY 0.4f            // Highlights for the wisps

// Integration Quality
#define STEP_SIZE_M 0.05f                // [M] Integration step size in vacuum
#define MAX_STEPS 13000                   // [Steps] Max ray steps

#endif

