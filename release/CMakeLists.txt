cmake_minimum_required(VERSION 3.18)
project(msc_enumeration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(OpenMP REQUIRED)

function(add_msc_target target source)
  add_executable(${target} ${source} src/semantic_graph.cpp)
  target_include_directories(${target} PRIVATE src)
  target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  if(MSVC)
    target_compile_options(${target} PRIVATE /O2 /arch:AVX2)
  else()
    target_compile_options(${target} PRIVATE -O3 -mavx2)
  endif()
endfunction()

foreach(algorithm RANGE 1 4)
  add_msc_target(alg${algorithm}-raw src/alg${algorithm}-raw.cpp)
  add_msc_target(alg${algorithm}-stats src/alg${algorithm}-stats.cpp)
endforeach()

add_executable(audit-graph scripts/audit_graph.cpp)
target_include_directories(audit-graph PRIVATE src)
target_link_libraries(audit-graph PRIVATE OpenMP::OpenMP_CXX)

add_msc_target(compute-tau scripts/compute_tau.cpp)
add_msc_target(build-pairsim-graph scripts/build_pairsim_graph.cpp)