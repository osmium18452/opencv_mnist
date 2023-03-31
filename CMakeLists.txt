cmake_minimum_required(VERSION 3.0)
set (project_name opencv_mnist)
project(${project_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH $ENV{OPENCV_INCLUDE_DIR})
set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_DIR})
set(OpenCV_DIR $ENV{OPENCV_ROOT}\\build)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(${project_name} main.cpp)
target_link_libraries(${project_name} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${project_name}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:${project_name}>)
endif (MSVC)

#D:\opencv\opencv\build\x64\vc16\bin\opencv_world470.dll
