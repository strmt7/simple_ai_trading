@echo off
call "C:\trader\simple_ai_trading\.tmp\lightgbm-opencl-4.7.0\msvc-setup.cmd"
if errorlevel 1 exit /b %errorlevel%
set MSBUILDDISABLENODEREUSE=1
"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -S "C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/cmake-build/_deps/opencl-icd-loader-src" -B "C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/loader-explicit-build" -G "Visual Studio 18 2026" -A x64 -T "v143,version=14.42.34433" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_SHARED_LIBS=ON -DOPENCL_ICD_LOADER_HEADERS_DIR=C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/cmake-build/_deps/opencl-headers-src
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build "C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/loader-explicit-build" --config Release --target OpenCL --parallel 8
exit /b %errorlevel%
