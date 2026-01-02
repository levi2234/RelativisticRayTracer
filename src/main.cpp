#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "raymarcher.h"

// --- SETTINGS ---
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;

// --- CAMERA CONTROLLER ---
struct CameraController {
    float3 pos = {0.0f, 1.2f, 0.0f}; // Camera starting position
    float yaw = 0.0f;               // Horizontal rotation
    float pitch = 0.0f;             // Vertical rotation
    float lastX = WINDOW_WIDTH / 2.0f;
    float lastY = WINDOW_HEIGHT / 2.0f;
    bool firstMouse = true;
    float moveSpeed = 0.15f;
    float mouseSensitivity = 0.1f;

    CameraState getCUDAState() {
        float radYaw = yaw * 3.14159f / 180.0f;
        float radPitch = pitch * 3.14159f / 180.0f;

        // Calculate Forward vector
        float3 forward;
        forward.x = std::sin(radYaw) * std::cos(radPitch);
        forward.y = std::sin(radPitch);
        forward.z = std::cos(radYaw) * std::cos(radPitch);
        
        // Normalize forward
        float mag = std::sqrt(forward.x*forward.x + forward.y*forward.y + forward.z*forward.z);
        forward.x /= mag; forward.y /= mag; forward.z /= mag;

        // Calculate Right vector (Cross of World Up and Forward)
        float3 worldUp = {0.0f, 1.0f, 0.0f};
        float3 right;
        right.x = worldUp.y * forward.z - worldUp.z * forward.y;
        right.y = worldUp.z * forward.x - worldUp.x * forward.z;
        right.z = worldUp.x * forward.y - worldUp.y * forward.x;
        float rMag = std::sqrt(right.x*right.x + right.y*right.y + right.z*right.z);
        right.x /= rMag; right.y /= rMag; right.z /= rMag;

        // Calculate Up vector (Cross of Forward and Right)
        float3 up;
        up.x = forward.y * right.z - forward.z * right.y;
        up.y = forward.z * right.x - forward.x * right.z;
        up.z = forward.x * right.y - forward.y * right.x;


        return {pos, forward, right, up};
    }
} g_Camera;

// --- GLOBAL STATE ---
struct AppState {
    GLuint pbo = 0;
    GLuint tex = 0;
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLuint shaderProgram = 0;
    cudaGraphicsResource_t cudaResource = nullptr;
} g_State;

// --- CALLBACKS ---

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

    // Constrain pitch to avoid flipping
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

void renderFrame() {
    static float time = 0.0f;
    time += 0.016f;

    uchar4* d_out;
    size_t size;
    cudaGraphicsMapResources(1, &g_State.cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, &size, g_State.cudaResource);

    launch_raymarch(d_out, WINDOW_WIDTH, WINDOW_HEIGHT, time, g_Camera.getCUDAState());

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
    
    // Mouse Capture
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    initGLResources();

    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        renderFrame();
        glfwSwapBuffers(window);
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }

    glfwTerminate();
    return 0;
}
