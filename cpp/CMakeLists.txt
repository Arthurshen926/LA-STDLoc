cmake_minimum_required(VERSION 3.16)
project(lafgs_poselib LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 REQUIRED)

set(
    POSELIB_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/PoseLib"
    CACHE PATH
    "Path to an unmodified PoseLib source checkout"
)
if(NOT EXISTS "${POSELIB_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "PoseLib source checkout not found: ${POSELIB_SOURCE_DIR}")
endif()

set(PYTHON_PACKAGE OFF CACHE BOOL "" FORCE)
set(WITH_BENCHMARK OFF CACHE BOOL "" FORCE)
add_subdirectory(
    "${POSELIB_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/PoseLib"
)
target_compile_options(PoseLib PRIVATE -Wno-error -Wno-unused-variable)
add_subdirectory(
    "${POSELIB_SOURCE_DIR}/pybind/pybind11"
    "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
)

pybind11_add_module(_lafgs_poselib MODULE lafgs_dependency_pose.cpp)
target_link_libraries(_lafgs_poselib PRIVATE PoseLib::PoseLib Eigen3::Eigen)
target_compile_options(_lafgs_poselib PRIVATE -O3 -Wall -Wextra)
set_target_properties(
    _lafgs_poselib
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../localization_training"
)
