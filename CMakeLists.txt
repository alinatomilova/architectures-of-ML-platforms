cmake_minimum_required(VERSION 3.16)
project(hw1_simd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(USE_NATIVE_ARCH "Use -march=native (portable if OFF)" OFF)
option(ARCH_X86_64_V3 "Use -march=x86-64-v3 if native is off (AVX2+FMA)" ON)
option(DISABLE_AUTOVEC  "Disable auto-vectorization for honest Scalar baseline" ON)
option(BUILD_BENCH      "Build benchmark" ON)

find_package(Threads REQUIRED)

# Common flags
add_compile_options(
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Debug>:-O0>
  $<$<CONFIG:Debug>:-g>
  -fwrapv
  -Wall
  -Wextra
)

# Arch
if(USE_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
    message(STATUS "SIMD arch: -march=native")
  else()
    message(WARNING "-march=native not supported, falling back to default")
  endif()
elseif(ARCH_X86_64_V3 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=x86-64-v3" HAS_V3)
  if(HAS_V3)
    add_compile_options(-march=x86-64-v3)
    message(STATUS "SIMD arch: -march=x86-64-v3 (AVX2+FMA, portable)")
  else()
    message(WARNING "-march=x86-64-v3 not supported, using default")
  endif()
endif()

# for Scalar baseline
if(DISABLE_AUTOVEC)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fno-vectorize -fno-slp-vectorize)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fno-tree-vectorize -fno-tree-slp-vectorize)
  endif()
endif()

# Header-only
add_library(hw INTERFACE)
target_include_directories(hw INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(hw INTERFACE Threads::Threads)

# Benchmark
if(BUILD_BENCH)
  add_executable(hw_bench bench/bench.cpp)
  target_link_libraries(hw_bench PRIVATE hw)
endif()

# Report
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  add_custom_target(report
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_report_tables.py
            ${CMAKE_CURRENT_SOURCE_DIR}/results.csv
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating tables from results.csv"
    USES_TERMINAL
  )
endif()