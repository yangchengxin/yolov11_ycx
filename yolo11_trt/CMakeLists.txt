cmake_minimum_required(VERSION 3.12)
project(yolov11-tensorrt)

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)  # 必须放在 project() 之后
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
add_definitions(-DAPI_EXPORTS)

# Include CUDA directories
# CUDA
# include_directories(${PROJECT_SOURCE_DIR}/windows) 
# find_package(CUDA REQUIRED) 
# include_directories(${CUDA_INCLUDE_DIRS}) 
# link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
# 在 enable_language(CUDA) 前添加
#set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES "70;72;75;80;86")  # 覆盖从 Volta 到 Ampere 的主流架构
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin/nvcc.exe")
enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(inc/)

# Add source files
set(SOURCES
    main.cpp
    src/yolov11.cpp
    src/utils.cpp
    # src/dll_manager.cpp
)

# Add CUDA source files
set(CUDA_SOURCES 
    src/preprocess.cu
    src/postprocess.cu)

# Add headers
set(HEADERS
    inc/yolov11.h
    inc/macros.h
    inc/logging.h
    inc/cuda_utils.h
    inc/preprocess.h
    inc/postprocess.h
    inc/common.h
    inc/utils.h
    inc/TRTLogger.h
    # inc/dll_manager.h
)

# Find and include OpenCV
# OpenCV
find_package(OpenCV REQUIRED
            CONFIG
            PATHS D:\\opencv\\build\\x64\\vc16\\lib) # bendi
            #PATHS E:\\opencv\\build\\x64\\vc16\\lib) # fuwuqi
            # PATHS D:\\opencv453-vs19\\install\\x64\\vc16\\lib)
include_directories(${OpenCV_INCLUDE_DIRS})
if (OpenCV_FOUND) 
    include_directories(${OpenCV_INCLUDE_DIRS}) 
    link_directories(${OpenCV_LIBRARY_DIRS}) 
    add_definitions(${OpenCV_DEFINITIONS}) 
else() 
    message(FATAL_ERROR "OpenCV not found!") 
endif()

# Set TensorRT path if not set in environment variables
#set(TENSORRT_DIR "D:\\TensorRT-8.6.1.6")
#set(TENSORRT_DIR "D:\\company_Tenda\\env\\TensorRT-8.4.2.4") # bendi
set(TENSORRT_DIR "T:\\chengxin.yang\\TensorRT-8.6.1.6") # fuwuqi 

# Include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)

# Create an executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${CUDA_SOURCES})
# target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)  # 正确方式
# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)