cmake_minimum_required(VERSION 3.16)
project(moeboost_tools_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(moeboost_tools
    test_each_tool.cpp
)

target_include_directories(moeboost_tools PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
