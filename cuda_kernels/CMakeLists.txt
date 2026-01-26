# MiniDriveWorld CUDA 自定义算子

cmake_minimum_required(VERSION 3.18)
project(mini_drive_world_cuda LANGUAGES CXX CUDA)

# CUDA 配置
find_package(CUDA REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 架构（根据你的 GPU 调整）
# RTX 5070 Ti 是 Blackwell 架构，sm_100
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# 包含目录
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# 编译 CUDA 库
add_library(mini_drive_cuda SHARED
    attention.cu
    layernorm.cu
    softmax.cu
    binding.cpp
)

# 链接库
target_link_libraries(mini_drive_cuda
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${Python_LIBRARIES}
)

# 设置输出名称
set_target_properties(mini_drive_cuda PROPERTIES
    PREFIX ""
    OUTPUT_NAME "mini_drive_cuda"
)

# 安装
install(TARGETS mini_drive_cuda
    LIBRARY DESTINATION lib
)
