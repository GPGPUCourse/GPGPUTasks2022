cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(aplusb)

set(CMAKE_CXX_STANDARD 11)

link_directories(C:/Windows/system32)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} libclew libutils OpenCL)
