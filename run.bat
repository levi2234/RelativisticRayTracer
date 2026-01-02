@echo off
setlocal

REM --- BUILD AND RUN SCRIPT ---
REM This script automates the 3-step process of building and executing the project:
REM 1. Configuration (CMake)
REM 2. Compilation (Build)
REM 3. Execution (Run)

set BUILD_DIR=build
set CONFIG=Debug

echo [1/3] Configuring project...
REM Note: On Windows, we use the Visual Studio generator for full CUDA support.
cmake -G "Visual Studio 17 2022" -A x64 -B %BUILD_DIR% -S .
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Configuration failed.
    exit /b %ERRORLEVEL%
)

echo [2/3] Building project...
REM Trigger the MSBuild system via CMake's unified interface.
cmake --build %BUILD_DIR% --config %CONFIG%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed.
    exit /b %ERRORLEVEL%
)

echo [3/3] Running executable...
REM Find result in the platform-specific subfolder (VS puts it in Debug/ or Release/)
set EXE_PATH=%BUILD_DIR%\%CONFIG%\RelativisticRayTracer.exe
if not exist "%EXE_PATH%" (
    set EXE_PATH=%BUILD_DIR%\%CONFIG%\RelativisticRayTracer.exe
)

if exist "%EXE_PATH%" (
    echo [SUCCESS] Running %EXE_PATH%...
    "%EXE_PATH%"
) else (
    echo [ERROR] Could not find executable at %EXE_PATH%
    exit /b 1
)

endlocal
