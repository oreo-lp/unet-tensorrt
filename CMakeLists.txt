cmake_minimum_required(VERSION 2.6)

project(unet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
# target_link_libraries(myplugins nvinfer cudart)

# opencv
find_package(OpenCV 3.4.14)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(unet ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/unet.cpp)
target_link_libraries(unet nvinfer)
target_link_libraries(unet cudart)
#target_link_libraries(unet myplugins)
target_link_libraries(unet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

