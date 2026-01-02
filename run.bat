@echo off
setlocal

set BUILD_DIR=build
set CONFIG=Debug

echo [1/3] Configuring project...
cmake -G "Visual Studio 17 2022" -A x64 -B %BUILD_DIR% -S .
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Configuration failed.
    exit /b %ERRORLEVEL%
)

echo [2/3] Building project...
cmake --build %BUILD_DIR% --config %CONFIG%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed.
    exit /b %ERRORLEVEL%
)

echo [3/3] Running executable...
set EXE_PATH=%BUILD_DIR%\%CONFIG%\RelativisticRayTracer.exe
if not exist "%EXE_PATH%" (
    set EXE_PATH=%BUILD_DIR%\RelativisticRayTracer.exe
)

if exist "%EXE_PATH%" (
    echo [SUCCESS] Running %EXE_PATH%...
    "%EXE_PATH%"
) else (
    echo [ERROR] Could not find executable at %EXE_PATH%
    exit /b 1
)

endlocal
