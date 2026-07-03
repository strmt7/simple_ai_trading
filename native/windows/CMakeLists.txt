cmake_minimum_required(VERSION 3.24)

project(SimpleAITradingNative VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(SimpleAITrading WIN32
    src/main.cpp
)

target_include_directories(SimpleAITrading PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/generated"
)

target_compile_definitions(SimpleAITrading PRIVATE
    UNICODE
    _UNICODE
    NOMINMAX
)

target_link_libraries(SimpleAITrading PRIVATE
    user32
    gdi32
    dwmapi
    shell32
)
