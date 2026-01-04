#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "raymarcher.h"
#include "config.h"
#include "camera_paths.h"

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

// --- SCREEN RECORDER ---
struct ScreenRecorder {
    FILE* ffmpeg = nullptr;
    std::vector<unsigned char> frameBuffer;
    bool isRecording = false;
    int frameCount = 0;
    std::string currentFilename;

    std::string generateFilename() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
#ifdef _WIN32
        localtime_s(&tm, &time);
#else
        localtime_r(&time, &tm);
#endif
        std::ostringstream oss;
        oss << "recording_" 
            << std::put_time(&tm, "%Y%m%d_%H%M%S") 
            << ".mp4";
        return oss.str();
    }

    bool startRecording(int width, int height) {
        if (isRecording) return false;

        frameBuffer.resize(width * height * 4);
        currentFilename = generateFilename();
        frameCount = 0;

        // Build FFmpeg command for raw RGBA input -> H.264 MP4 output
        std::ostringstream cmd;
        cmd << "ffmpeg -y "                           // Overwrite output
            << "-f rawvideo "                         // Input format
            << "-pix_fmt rgba "                       // Pixel format
            << "-s " << width << "x" << height << " " // Frame size
            << "-r " << RECORDING_FPS << " "          // Input framerate
            << "-i - "                                // Read from stdin
            << "-vf vflip "                           // Flip vertically (OpenGL origin is bottom-left)
            << "-c:v libx264 "                        // H.264 codec
            << "-preset fast "                        // Encoding speed/quality tradeoff
            << "-crf 18 "                             // Quality (lower = better, 18 is visually lossless)
            << "-pix_fmt yuv420p "                    // Output pixel format for compatibility
            << "\"" << currentFilename << "\"";

        ffmpeg = popen(cmd.str().c_str(), "wb");
        if (!ffmpeg) {
            std::cerr << "Failed to start FFmpeg. Make sure FFmpeg is installed and in PATH." << std::endl;
            return false;
        }

        isRecording = true;
        std::cout << "Recording started: " << currentFilename << std::endl;
        return true;
    }

    void captureFrame(int width, int height) {
        if (!isRecording || !ffmpeg) return;

        // Read pixels from OpenGL framebuffer
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, frameBuffer.data());
        
        // Write frame to FFmpeg
        size_t written = fwrite(frameBuffer.data(), 1, frameBuffer.size(), ffmpeg);
        if (written != frameBuffer.size()) {
            std::cerr << "Warning: Frame write incomplete" << std::endl;
        }
        frameCount++;
    }

    void stopRecording() {
        if (!isRecording) return;

        if (ffmpeg) {
            pclose(ffmpeg);
            ffmpeg = nullptr;
        }

        isRecording = false;
        std::cout << "Recording stopped: " << currentFilename 
                  << " (" << frameCount << " frames, ~" 
                  << (frameCount / RECORDING_FPS) << "s)" << std::endl;
    }

    void toggle(int width, int height) {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording(width, height);
        }
    }

    ~ScreenRecorder() {
        stopRecording();
    }
} g_Recorder;

// --- CAMERA CONTROLLER ---
struct CameraController {
    float3 pos = {0.0f, 10.0f, -60.0f}; 
    float yaw = 0.0f;               
    float pitch = -10.0f;             
    float lastX = WINDOW_WIDTH / 2.0f;
    float lastY = WINDOW_HEIGHT / 2.0f;
    bool firstMouse = true;
    float moveSpeed = 0.3f;
    float mouseSensitivity = 0.1f;

    CameraState getCUDAState() {
        return getCUDAStateFrom(pos, yaw, pitch);
    }

    static CameraState getCUDAStateFrom(float3 pos, float yaw, float pitch) {
        float radYaw = yaw * 3.14159f / 180.0f;
        float radPitch = pitch * 3.14159f / 180.0f;

        float3 forward;
        forward.x = std::sin(radYaw) * std::cos(radPitch);
        forward.y = std::sin(radPitch);
        forward.z = std::cos(radYaw) * std::cos(radPitch);
        
        float mag = std::sqrt(forward.x*forward.x + forward.y*forward.y + forward.z*forward.z);
        forward.x /= mag; forward.y /= mag; forward.z /= mag;

        float3 worldUp = {0.0f, 1.0f, 0.0f};
        float3 right;
        right.x = worldUp.y * forward.z - worldUp.z * forward.y;
        right.y = worldUp.z * forward.x - worldUp.x * forward.z;
        right.z = worldUp.x * forward.y - worldUp.y * forward.x;
        float rMag = std::sqrt(right.x*right.x + right.y*right.y + right.z*right.z);
        right.x /= rMag; right.y /= rMag; right.z /= rMag;

        float3 up;
        up.x = forward.y * right.z - forward.z * right.y;
        up.y = forward.z * right.x - forward.x * right.z;
        up.z = forward.x * right.y - forward.y * right.x;

        return {pos, forward, right, up};
    }
} g_Camera;

// --- KEYFRAMING SYSTEM ---
struct PathController {
    int currentPathIndex = 0;
    bool active = false;
    float pathTime = 0.0f; // Fixed simulation clock

    CameraState getInterpolatedState() {
        const CameraPath* path = PathManager::instance().getPath(currentPathIndex);
        if (!path || path->keyframes.empty()) return g_Camera.getCUDAState();

        float t = pathTime;
        const auto& keys = path->keyframes;

        if (t <= keys.front().time) return CameraController::getCUDAStateFrom(keys.front().pos, keys.front().yaw, keys.front().pitch);
        if (t >= keys.back().time) return CameraController::getCUDAStateFrom(keys.back().pos, keys.back().yaw, keys.back().pitch);

        for (size_t i = 0; i < keys.size() - 1; ++i) {
            if (t >= keys[i].time && t <= keys[i+1].time) {
                float factor = (t - keys[i].time) / (keys[i+1].time - keys[i].time);
                
                int i0 = std::max(0, (int)i - 1);
                int i1 = (int)i;
                int i2 = (int)i + 1;
                int i3 = std::min((int)keys.size() - 1, (int)i + 2);

                float3 p = catmull_rom(keys[i0].pos, keys[i1].pos, keys[i2].pos, keys[i3].pos, factor);
                float y = lerp_angle(keys[i1].yaw, keys[i2].yaw, factor);
                float pi = lerp_angle(keys[i1].pitch, keys[i2].pitch, factor);
                
                return CameraController::getCUDAStateFrom(p, y, pi);
            }
        }
        return g_Camera.getCUDAState();
    }

    void start() {
        active = true;
        pathTime = 0.0f;
    }

    void update(float dt) {
        if (active) pathTime += dt;
    }

    void stop() { active = false; }
    
    void nextPath() {
        if (PathManager::instance().getPaths().empty()) return;
        currentPathIndex = (currentPathIndex + 1) % PathManager::instance().getPaths().size();
    }
} g_Path;

// --- GLOBAL STATE ---
struct AppState {
    GLuint pbo = 0;
    GLuint tex = 0;
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLuint shaderProgram = 0;
    cudaGraphicsResource_t cudaResource = nullptr;
    
    // Skybox Texture
    cudaArray* skyboxArray = nullptr;
    cudaTextureObject_t skyboxTexObj = 0;
} g_State;

// --- SKYBOX LOADING ---

void loadSkybox(const char* filename) {
    int width, height, channels;
    // Load as float for high-quality mapping later, though PNG is 8-bit
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 4);
    if (!data) {
        std::cerr << "Failed to load skybox: " << filename << std::endl;
        return;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&g_State.skyboxArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(g_State.skyboxArray, 0, 0, data, width * 4, width * 4, height, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = g_State.skyboxArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat; // Map 0-255 to 0.0-1.0
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&g_State.skyboxTexObj, &resDesc, &texDesc, NULL);
    stbi_image_free(data);
    std::cout << "Skybox loaded: " << width << "x" << height << std::endl;
}

// --- CALLBACKS ---

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode; (void)mods; // Unused parameters
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        g_Recorder.toggle(WINDOW_WIDTH, WINDOW_HEIGHT);
    }
    if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        if (g_Path.active) g_Path.stop();
        else g_Path.start();
    }
    if (key == GLFW_KEY_N && action == GLFW_PRESS) {
        g_Path.nextPath();
        std::cout << "Switched to path: " << PathManager::instance().getPath(g_Path.currentPathIndex)->name << std::endl;
    }
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    if (g_Camera.firstMouse) {
        g_Camera.lastX = xpos;
        g_Camera.lastY = ypos;
        g_Camera.firstMouse = false;
    }
    float xoffset = g_Camera.lastX - xpos;
    float yoffset = g_Camera.lastY - ypos;
    g_Camera.lastX = xpos;
    g_Camera.lastY = ypos;

    xoffset *= g_Camera.mouseSensitivity;
    yoffset *= g_Camera.mouseSensitivity;
    g_Camera.yaw   -= xoffset;
    g_Camera.pitch -= yoffset;
    if (g_Camera.pitch > 89.0f)  g_Camera.pitch = 89.0f;
    if (g_Camera.pitch < -89.0f) g_Camera.pitch = -89.0f;
}

void processInput(GLFWwindow* window) {
    CameraState cs = g_Camera.getCUDAState();
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        g_Camera.pos.x += cs.forward.x * g_Camera.moveSpeed;
        g_Camera.pos.y += cs.forward.y * g_Camera.moveSpeed;
        g_Camera.pos.z += cs.forward.z * g_Camera.moveSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        g_Camera.pos.x -= cs.forward.x * g_Camera.moveSpeed;
        g_Camera.pos.y -= cs.forward.y * g_Camera.moveSpeed;
        g_Camera.pos.z -= cs.forward.z * g_Camera.moveSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        g_Camera.pos.x -= cs.right.x * g_Camera.moveSpeed;
        g_Camera.pos.y -= cs.right.y * g_Camera.moveSpeed;
        g_Camera.pos.z -= cs.right.z * g_Camera.moveSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        g_Camera.pos.x += cs.right.x * g_Camera.moveSpeed;
        g_Camera.pos.y += cs.right.y * g_Camera.moveSpeed;
        g_Camera.pos.z += cs.right.z * g_Camera.moveSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        g_Camera.pos.y -= g_Camera.moveSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        g_Camera.pos.y += g_Camera.moveSpeed;
    }
}

// --- OPENGL SETUP ---

const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D ourTexture;
    void main() {
        FragColor = texture(ourTexture, TexCoord);
    }
)";

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader Error: " << infoLog << std::endl;
    }
    return shader;
}

void initGLResources() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    g_State.shaderProgram = glCreateProgram();
    glAttachShader(g_State.shaderProgram, vs);
    glAttachShader(g_State.shaderProgram, fs);
    glLinkProgram(g_State.shaderProgram);

    float vertices[] = {
         1.0f,  1.0f, 1.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };
    unsigned int indices[] = { 0, 1, 3, 1, 2, 3 };

    glGenVertexArrays(1, &g_State.vao);
    glGenBuffers(1, &g_State.vbo);
    glGenBuffers(1, &g_State.ebo);
    glBindVertexArray(g_State.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_State.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_State.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenTextures(1, &g_State.tex);
    glBindTexture(GL_TEXTURE_2D, g_State.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glGenBuffers(1, &g_State.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_State.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    
    cudaGraphicsGLRegisterBuffer(&g_State.cudaResource, g_State.pbo, cudaGraphicsRegisterFlagsWriteDiscard);
}

void updateFPS(GLFWwindow* window) {
    static double lastTime = 0.0;
    static int nbFrames = 0;
    double currentTime = glfwGetTime();
    double delta = currentTime - lastTime;
    nbFrames++;
    if (delta >= 1.0) {
        double fps = double(nbFrames) / delta;
        char title[256];
        const char* rec = g_Recorder.isRecording ? " [REC]" : "";
        const char* path_active = g_Path.active ? " [PATH]" : "";
        const CameraPath* path = PathManager::instance().getPath(g_Path.currentPathIndex);
        const char* path_name = path ? path->name.c_str() : "None";

        sprintf(title, "Relativistic Ray Tracer | FPS: %.1f | %s%s %s | R:Rec P:Path N:Next", 
                fps, rec, path_active, path_name);
        glfwSetWindowTitle(window, title);
        nbFrames = 0;
        lastTime = currentTime;
    }
}

void renderFrame(float simTime) {
    uchar4* d_out;
    size_t size;
    cudaGraphicsMapResources(1, &g_State.cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, &size, g_State.cudaResource);

    CameraState camState = g_Path.active ? g_Path.getInterpolatedState() : g_Camera.getCUDAState();
    launch_raymarch(d_out, WINDOW_WIDTH, WINDOW_HEIGHT, simTime, camState, g_State.skyboxTexObj);

    cudaGraphicsUnmapResources(1, &g_State.cudaResource, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_State.pbo);
    glBindTexture(GL_TEXTURE_2D, g_State.tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glUseProgram(g_State.shaderProgram);
    glBindVertexArray(g_State.vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Relativistic Ray Tracer", NULL, NULL);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    initGLResources();
    loadSkybox("assets/skyboxes/skybox2.jpg");

    // Initialize cinematic paths
    initDefaultPaths();

    float simTime = 0.0f;
    double lastFrameTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double currentFrameTime = glfwGetTime();
        float dt = (float)(currentFrameTime - lastFrameTime);
        lastFrameTime = currentFrameTime;

        // If recording, force a fixed time step to avoid path skips during lag
        if (g_Recorder.isRecording) {
            dt = 1.0f / RECORDING_FPS;
        }

        simTime += dt;
        g_Path.update(dt);

        updateFPS(window);
        processInput(window);
        renderFrame(simTime);
        
        // Capture frame for recording (after render, before swap)
        if (g_Recorder.isRecording) {
            g_Recorder.captureFrame(WINDOW_WIDTH, WINDOW_HEIGHT);
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    g_Recorder.stopRecording();  // Ensure recording is stopped and file is finalized
    
    if (g_State.skyboxTexObj) cudaDestroyTextureObject(g_State.skyboxTexObj);
    if (g_State.skyboxArray) cudaFreeArray(g_State.skyboxArray);
    
    glfwTerminate();
    return 0;
}
