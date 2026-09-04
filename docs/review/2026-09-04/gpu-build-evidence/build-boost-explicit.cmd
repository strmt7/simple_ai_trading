@echo off
call "C:\trader\simple_ai_trading\.tmp\lightgbm-opencl-4.7.0\msvc-setup.cmd"
if errorlevel 1 exit /b %errorlevel%
cd /d "C:\trader\simple_ai_trading\.tmp\lightgbm-opencl-4.7.0\cmake-build\Boost\source"
b2.exe --ignore-site-config --user-config= --project-config=C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/explicit-project-config.jam --build-dir=C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/boost-explicit-build --stagedir=C:/trader/simple_ai_trading/.tmp/lightgbm-opencl-4.7.0/boost-explicit-stage toolset=msvc-14.3 architecture=x86 address-model=64 -j8 -q --with-headers --with-chrono --with-filesystem --with-system link=static runtime-link=shared variant=release threading=multi
exit /b %errorlevel%
