cmake_minimum_required(VERSION 3.16)
project(hw1_simd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -march=native подбирает ширину SIMD под процессор, на котором идёт сборка (AVX2 -> 32 байта, AVX-512 -> 64)
# Для переносимого x86-64 бинарника: cmake -DHW_NATIVE=OFF (тогда AVX2 + FMA)
option(HW_NATIVE "Compile with -march=native" ON)

find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)

add_library(hw INTERFACE)
target_include_directories(hw INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hw INTERFACE Threads::Threads)

# Код использует векторные расширения GCC/Clang (__attribute__((vector_size)))
# Автовекторизацию выключаем, иначе компилятор векторизует и "скалярную" версию
# -fwrapv: переполнение int32 определено (по модулю 2^32)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(HW_NOVEC -fno-vectorize -fno-slp-vectorize)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(HW_NOVEC -fno-tree-vectorize -fno-tree-slp-vectorize)
else()
  message(FATAL_ERROR "GCC or Clang is required (GCC/Clang vector extensions are used)")
endif()
target_compile_options(hw INTERFACE ${HW_NOVEC} -fwrapv)

if(HW_NATIVE)
  check_cxx_compiler_flag(-march=native HW_HAS_MARCH_NATIVE)
  if(HW_HAS_MARCH_NATIVE)
    target_compile_options(hw INTERFACE -march=native)
  else()
    check_cxx_compiler_flag(-mcpu=native HW_HAS_MCPU_NATIVE)  # Apple Silicon / ARM
    if(HW_HAS_MCPU_NATIVE)
      target_compile_options(hw INTERFACE -mcpu=native)
    endif()
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(hw INTERFACE -mavx2 -mfma)
endif()

add_executable(hw_tests tests/test_main.cpp)
target_link_libraries(hw_tests PRIVATE hw)
target_compile_options(hw_tests PRIVATE -Wall -Wextra)

add_executable(hw_bench bench/bench.cpp)
target_link_libraries(hw_bench PRIVATE hw)
target_compile_options(hw_bench PRIVATE -Wall -Wextra)

enable_testing()
add_test(NAME hw_tests COMMAND hw_tests)
