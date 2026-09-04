@echo off
call "C:\trader\simple_ai_trading\.tmp\lightgbm-opencl-4.7.0\msvc-setup.cmd"
if errorlevel 1 exit /b %errorlevel%
set MSBUILDDISABLENODEREUSE=1
set CMAKE_GENERATOR=Visual Studio 18 2026
set CMAKE_GENERATOR_PLATFORM=x64
set CMAKE_GENERATOR_TOOLSET=v143,version=14.42.34433
set CMAKE_BUILD_PARALLEL_LEVEL=8
set "PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"
"C:\Users\strat\AppData\Local\hermes\bin\uv.exe" build "C:\Users\strat\AppData\Local\uv\cache\sdists-v9\pypi\lightgbm\4.7.0\KAceDCsqSF6rJGy0\src" --wheel --out-dir C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/wheels --config-setting=cmake.define.USE_GPU=ON --config-setting=cmake.define.CMAKE_POLICY_DEFAULT_CMP0167=OLD --config-setting=cmake.define.Boost_NO_BOOST_CMAKE=ON --config-setting=cmake.define.Boost_COMPILER=-vc143 --config-setting=cmake.define.Boost_ARCHITECTURE=-x64 --config-setting=cmake.define.Boost_INCLUDE_DIR=C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/cmake-build/Boost/source --config-setting=cmake.define.BOOST_LIBRARYDIR=C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/boost-explicit-stage/lib --config-setting=cmake.define.OpenCL_INCLUDE_DIR=C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/cmake-build/_deps/opencl-headers-src --config-setting=cmake.define.OpenCL_LIBRARY=C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/loader-explicit-build/Release/OpenCL.lib --config-setting=build-dir=C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/gpu-wheel-build
exit /b %errorlevel%
