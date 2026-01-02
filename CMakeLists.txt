cmake_minimum_required(VERSION 3.18)
project(RelativisticRayTracer VERSION 0.1.0 LANGUAGES CXX CUDA)

# Standard C++ and CUDA versions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Directory definitions
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# External Dependencies (GLFW)
include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

message(STATUS "Fetching GLFW...")
FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/releases/download/3.3.8/glfw-3.3.8.zip
    TLS_VERIFY OFF
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Source files
add_executable(${PROJECT_NAME} 
    ${SRC_DIR}/main.cpp 
    ${SRC_DIR}/glad.c 
    ${SRC_DIR}/raymarcher.cu
)

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIR})

# Link libraries
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    glfw 
    OpenGL::GL 
    CUDA::cudart
)
