cmake_minimum_required(VERSION 3.10)
add_compile_options(-mavx -mavx2 -mfma -O3 -mbmi2)
project(18645-canny-edge)

find_package(OpenCV REQUIRED)

add_executable(
    baseline
    main.cpp
)

target_link_libraries(
    baseline
    ${OpenCV_LIBS}
)
