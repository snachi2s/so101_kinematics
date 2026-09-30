cmake_minimum_required(VERSION 3.18)
project(SO101_Kinematics CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
include(FetchContent)

# Eigen: use the system copy (apt install libeigen3-dev) or download 3.4.0 (header only)
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 not found, downloading 3.4.0")
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        SOURCE_SUBDIR headers_only)
    FetchContent_MakeAvailable(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${eigen_SOURCE_DIR})
endif()

# Sophus: use a local checkout in ./Sophus if present, otherwise download the pinned version (header only)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Sophus/sophus/se3.hpp")
    set(SOPHUS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Sophus")
else()
    message(STATUS "Sophus not found in ./Sophus, downloading 1.24.6")
    FetchContent_Declare(sophus
        URL https://github.com/strasdat/Sophus/archive/d0b7315a0d90fc6143defa54596a3a95d9fa10ec.tar.gz
        URL_HASH SHA256=876e24fd402ac4116cd7863685225388b77be86ec356b777bf68defd79decc90
        SOURCE_SUBDIR headers_only)
    FetchContent_MakeAvailable(sophus)
    set(SOPHUS_INCLUDE_DIR "${sophus_SOURCE_DIR}")
endif()

# Python module, built by `pip install .` (scikit-build-core sets SKBUILD)
if(SKBUILD)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core python/bindings.cpp)
    target_include_directories(_core PRIVATE ${SOPHUS_INCLUDE_DIR} include)
    target_link_libraries(_core PRIVATE Eigen3::Eigen)
    target_compile_definitions(_core PRIVATE SOPHUS_ENABLE_ENSURE_HANDLER)
    install(TARGETS _core DESTINATION so101_kinematics)
    return()
endif()

add_executable(tests src/tests.cpp)
target_include_directories(tests PRIVATE ${SOPHUS_INCLUDE_DIR} include)
target_link_libraries(tests PRIVATE Eigen3::Eigen)
target_compile_options(tests PRIVATE -Wall -Wextra)

enable_testing()
add_test(NAME kinematics_tests COMMAND tests)
