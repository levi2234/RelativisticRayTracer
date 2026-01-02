#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include "raymarcher.h"

// --- SETTINGS ---
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;

// --- STATE ---
struct AppState {
    GLuint pbo = 0;
    GLuint tex = 0;
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLuint shaderProgram = 0;
    cudaGraphicsResource_t cudaResource = nullptr;
} g_State;

// --- SHADERS ---
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

// --- UTILITIES ---

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader Compilation Error:\n" << infoLog << std::endl;
    }
    return shader;
}

void initGLResources() {
    // 1. Shaders
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    g_State.shaderProgram = glCreateProgram();
    glAttachShader(g_State.shaderProgram, vs);
    glAttachShader(g_State.shaderProgram, fs);
    glLinkProgram(g_State.shaderProgram);
    glDeleteShader(vs);
    glDeleteShader(fs);

    // 2. Geometry (Full screen quad)
    float vertices[] = {
         1.0f,  1.0f, 1.0f, 1.0f, // TR
         1.0f, -1.0f, 1.0f, 0.0f, // BR
        -1.0f, -1.0f, 0.0f, 0.0f, // BL
        -1.0f,  1.0f, 0.0f, 1.0f  // TL
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

    // 3. Texture & PBO
    glGenTextures(1, &g_State.tex);
    glBindTexture(GL_TEXTURE_2D, g_State.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glGenBuffers(1, &g_State.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_State.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // 4. CUDA Interop hand-shake
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&g_State.cudaResource, g_State.pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "CUDA-GL Register failed: " << cudaGetErrorString(err) << std::endl;
    }
}

// --- MAIN LOOP ---

void renderFrame() {
    static float time = 0.0f;
    time += 0.016f; // Approx 60fps dt

    // 1. Map for CUDA
    uchar4* d_out;
    size_t size;
    cudaGraphicsMapResources(1, &g_State.cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, &size, g_State.cudaResource);

    // 2. Launch Kernel
    launch_raymarch(d_out, WINDOW_WIDTH, WINDOW_HEIGHT, time);

    // 3. Unmap
    cudaGraphicsUnmapResources(1, &g_State.cudaResource, 0);

    // 4. Draw to Screen
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
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    initGLResources();

    std::cout << "Rendering started..." << std::endl;
    while (!glfwWindowShouldClose(window)) {
        renderFrame();
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(g_State.cudaResource);
    glDeleteBuffers(1, &g_State.pbo);
    glDeleteTextures(1, &g_State.tex);
    glDeleteProgram(g_State.shaderProgram);
    glDeleteVertexArrays(1, &g_State.vao);
    glDeleteBuffers(1, &g_State.vbo);
    glDeleteBuffers(1, &g_State.ebo);
    
    glfwTerminate();
    return 0;
}
