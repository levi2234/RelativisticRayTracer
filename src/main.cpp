#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include "raymarcher.h"

/**
 * --- RELATIVISTIC RAY TRACER (CUDA-OPENGL INTEROP) ---
 * 
 * This application uses CUDA for high-performance raymarching calculations
 * and OpenGL for displaying the results. The "bridge" between them is a 
 * Pixel Buffer Object (PBO).
 * 
 * Workflow:
 * 1. CUDA "Maps" the PBO, getting a pointer to its memory.
 * 2. CUDA Kernel writes pixel data (RGBA) directly into that memory.
 * 3. CUDA "Unmaps" the PBO, returning control to OpenGL.
 * 4. OpenGL copies the PBO data into a Texture.
 * 5. OpenGL renders a full-screen quad with that Texture.
 */

// --- SETTINGS ---
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;

// --- STATE ---
// Standardized global state for the application
struct AppState {
    GLuint pbo = 0;             // Pixel Buffer Object: Stores raw pixel data
    GLuint tex = 0;             // Texture ID: OpenGL uses this to show the PBO
    GLuint vao = 0, vbo = 0, ebo = 0; // Geometry for the full-screen quad
    GLuint shaderProgram = 0;   // Simple shader to display the texture
    cudaGraphicsResource_t cudaResource = nullptr; // Handle for CUDA-GL interop
} g_State;

// --- SHADERS ---
// Minimal shaders to draw a texture on a quad
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

// Compiles a shader and checks for errors
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

// Initialized PBO, Texture, Shaders, and CUDA interop
void initGLResources() {
    // 1. Shaders: Compile and Link
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    g_State.shaderProgram = glCreateProgram();
    glAttachShader(g_State.shaderProgram, vs);
    glAttachShader(g_State.shaderProgram, fs);
    glLinkProgram(g_State.shaderProgram);
    glDeleteShader(vs);
    glDeleteShader(fs);

    // 2. Geometry: Define a quad that covers the whole screen (-1 to 1)
    float vertices[] = {
        // Positions (x, y)   // TexCoords (u, v)
         1.0f,  1.0f,         1.0f, 1.0f, // Top Right
         1.0f, -1.0f,         1.0f, 0.0f, // Bottom Right
        -1.0f, -1.0f,         0.0f, 0.0f, // Bottom Left
        -1.0f,  1.0f,         0.0f, 1.0f  // Top Left
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

    // 3. Texture & PBO Setup
    // Texture will hold the final image for the shader to sample
    glGenTextures(1, &g_State.tex);
    glBindTexture(GL_TEXTURE_2D, g_State.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // PBO: Memory buffer shared with CUDA
    glGenBuffers(1, &g_State.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_State.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // 4. CUDA Interop: Connect the PBO to CUDA
    // cudaGraphicsRegisterFlagsWriteDiscard means CUDA will overwrite the buffer
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&g_State.cudaResource, g_State.pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "CUDA-GL Register failed: " << cudaGetErrorString(err) << std::endl;
    }
}

// --- MAIN LOOP ---

// Called every frame to calculate and draw
void renderFrame() {
    static float time = 0.0f;
    time += 0.016f; // Approximately 60 FPS update

    // 1. Map: CUDA takes control of the OpenGL buffer
    uchar4* d_out;
    size_t size;
    cudaGraphicsMapResources(1, &g_State.cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, &size, g_State.cudaResource);

    // 2. Launch Kernel: CUDA writes the raymarched image to d_out
    launch_raymarch(d_out, WINDOW_WIDTH, WINDOW_HEIGHT, time);

    // 3. Unmap: Hand control back to OpenGL
    cudaGraphicsUnmapResources(1, &g_State.cudaResource, 0);

    // 4. Update Texture: Copy raw pixel data from PBO to the GPU Texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_State.pbo);
    glBindTexture(GL_TEXTURE_2D, g_State.tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // 5. Render: Draw the quad with the new texture
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(g_State.shaderProgram);
    glBindVertexArray(g_State.vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) return -1;
    
    // OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Relativistic Ray Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Load OpenGL extensions via GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Prepare GPU resources
    initGLResources();

    std::cout << "Rendering started..." << std::endl;
    while (!glfwWindowShouldClose(window)) {
        renderFrame();
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        // Quick exit with Escape key
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }

    // Cleanup resources
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
