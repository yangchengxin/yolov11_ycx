cmake_minimum_required(VERSION 3.28)
project(yolo11_trtx_v8)

set(CMAKE_CXX_STANDARD 11)
# 设置nvcc编译cu文件时候使用utf-8编码
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

enable_language(CUDA)

# 设置cuda多个框架支持
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# OpenCV
set(OpenCV_DIR E:\\env\\opencv\\build\\x64\\vc16\\lib)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR C:\\Program\ Files\\NVIDIA\ GPU\ Computing\ Toolkit\\CUDA\\v12.5)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# TensorRT
set(TENSORRT_ROOT D:\\TensorRT-8.6.1.6)
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)
link_libraries(
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        cudart
        cublas
        nvinfer
)

include_directories(${CMAKE_SOURCE_DIR}/windows)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/plugin)
include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_definitions(-DNOMINMAX)

add_definitions(-DAPI_EXPORTS)

file(GLOB_RECURSE SRCS ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

add_executable(yolo11_det yolo11_det.cpp ${SRCS})
target_link_libraries(yolo11_det nvinfer)
target_link_libraries(yolo11_det cudart)
target_link_libraries(yolo11_det myplugins)
target_link_libraries(yolo11_det ${OpenCV_LIBS})

add_executable(yolo11_cls yolo11_cls.cpp ${SRCS})
target_link_libraries(yolo11_cls nvinfer)
target_link_libraries(yolo11_cls cudart)
target_link_libraries(yolo11_cls myplugins)
target_link_libraries(yolo11_cls ${OpenCV_LIBS})

add_executable(yolo11_seg yolo11_seg.cpp ${SRCS})
target_link_libraries(yolo11_seg nvinfer)
target_link_libraries(yolo11_seg cudart)
target_link_libraries(yolo11_seg myplugins)
target_link_libraries(yolo11_seg ${OpenCV_LIBS})

add_executable(yolo11_pose yolo11_pose.cpp ${SRCS})
target_link_libraries(yolo11_pose nvinfer)
target_link_libraries(yolo11_pose cudart)
target_link_libraries(yolo11_pose myplugins)
target_link_libraries(yolo11_pose ${OpenCV_LIBS})

add_executable(yolo11_obb yolo11_obb.cpp ${SRCS})
target_link_libraries(yolo11_obb nvinfer)
target_link_libraries(yolo11_obb cudart)
target_link_libraries(yolo11_obb myplugins)
target_link_libraries(yolo11_obb ${OpenCV_LIBS})
