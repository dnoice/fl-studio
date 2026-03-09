cmake_minimum_required(VERSION 3.16)
project(fl_dsp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Core DSP library
add_library(fl_dsp STATIC
    src/dsp_core.cpp
    src/filters.cpp
    src/effects.cpp
)

target_include_directories(fl_dsp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optimization flags
if(MSVC)
    target_compile_options(fl_dsp PRIVATE /O2 /fp:fast /arch:AVX2)
else()
    target_compile_options(fl_dsp PRIVATE -O3 -ffast-math -march=native)
endif()

# Python bindings (optional - requires pybind11)
# Try CMake find first, then fall back to pip-installed pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _pybind11_result
    )
    if(_pybind11_result EQUAL 0 AND EXISTS "${_pybind11_cmake_dir}")
        set(pybind11_DIR "${_pybind11_cmake_dir}")
        find_package(pybind11 QUIET)
    endif()
endif()

if(pybind11_FOUND)
    pybind11_add_module(fl_dsp_py bindings/pybind_dsp.cpp)
    target_link_libraries(fl_dsp_py PRIVATE fl_dsp)

    message(STATUS "pybind11 found - building Python bindings")
else()
    message(STATUS "pybind11 not found - skipping Python bindings")
    message(STATUS "Install with: pip install pybind11")
endif()
