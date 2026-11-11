cmake_minimum_required(VERSION 3.14)
project(nn VERSION 2.0.0 DESCRIPTION "CPU-efficient neural network library in C++17" LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(NN_TOP_LEVEL ON)
else()
    set(NN_TOP_LEVEL OFF)
endif()

# Default top-level build = the classic single-hidden-layer program (./bin/NN),
# exactly as before. The new library is opt-in: -DNN_BUILD_LIBRARY=ON.
# When used via add_subdirectory() the library is built by default.
if(NN_TOP_LEVEL)
    set(NN_LIBRARY_DEFAULT OFF)
else()
    set(NN_LIBRARY_DEFAULT ON)
endif()
option(NN_BUILD_LIBRARY    "Build the nn library (plus tests/examples/benchmarks)" ${NN_LIBRARY_DEFAULT})
option(NN_BUILD_LEGACY     "Build the classic NN program from src/*.cpp"    ${NN_TOP_LEVEL})
option(NN_BUILD_EXAMPLES   "Build example programs"                         ${NN_TOP_LEVEL})
option(NN_BUILD_TESTS      "Build unit tests"                               ${NN_TOP_LEVEL})
option(NN_BUILD_BENCHMARKS "Build benchmarks"                               ${NN_TOP_LEVEL})
option(NN_USE_OPENMP       "Multi-threaded kernels via OpenMP"              ON)
option(NN_USE_BLAS         "Use an external CBLAS (e.g. OpenBLAS) for GEMM" OFF)
option(NN_NATIVE_ARCH      "Optimise for the build machine (-march=native)" ${NN_TOP_LEVEL})
option(NN_INSTALL          "Generate install target"                        ${NN_TOP_LEVEL})

if(NN_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------- classic program
if(NN_BUILD_LEGACY)
    add_executable(NN
        src/main.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/Neuron.cpp)
    target_include_directories(NN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(NN PROPERTIES
        CXX_STANDARD 14
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)  # run from bin/: reads ../data
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(NN PRIVATE -Wall -Wpedantic)
    endif()
endif()

if(NOT NN_BUILD_LIBRARY)
    return()
endif()

# ---------------------------------------------------------------- library

set(NN_SOURCES
    src/activation.cpp
    src/data.cpp
    src/kernels.cpp
    src/layer_factory.cpp
    src/loss.cpp
    src/matrix.cpp
    src/network.cpp
    src/optimizer.cpp
    src/spiking.cpp
    src/layers/conv.cpp
    src/layers/dense.cpp
    src/layers/dropout.cpp
    src/layers/layernorm.cpp
    src/layers/rbf.cpp
    src/layers/recurrent.cpp
)

add_library(nn ${NN_SOURCES})
add_library(nn::nn ALIAS nn)

target_compile_features(nn PUBLIC cxx_std_17)
set_target_properties(nn PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

include(GNUInstallDirs)
target_include_directories(nn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -fno-trapping-math / -fno-math-errno let GCC if-convert and vectorise the
    # branch-free math kernels; IEEE results are unchanged (no -ffast-math).
    target_compile_options(nn PRIVATE -Wall -Wextra -Wpedantic -fno-trapping-math -fno-math-errno
        $<$<CONFIG:Release,RelWithDebInfo>:-O3>)
    if(NN_NATIVE_ARCH)
        target_compile_options(nn PRIVATE -march=native)
    endif()
elseif(MSVC)
    target_compile_options(nn PRIVATE /W4 $<$<CONFIG:Release>:/O2 /fp:fast>)
    if(NN_NATIVE_ARCH)
        target_compile_options(nn PRIVATE /arch:AVX2)
    endif()
endif()

set(NN_CONFIG_DEPS "")
if(NN_USE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(nn PRIVATE OpenMP::OpenMP_CXX)
        string(APPEND NN_CONFIG_DEPS "find_dependency(OpenMP COMPONENTS CXX)\n")
    else()
        message(STATUS "nn: OpenMP not found, kernels run single-threaded")
    endif()
endif()
if(NOT OpenMP_CXX_FOUND AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # SIMD pragmas without the OpenMP runtime
    target_compile_options(nn PRIVATE -fopenmp-simd)
    target_compile_definitions(nn PRIVATE NN_HAS_OPENMP_SIMD)
endif()

if(NN_USE_BLAS)
    find_package(BLAS REQUIRED)
    target_link_libraries(nn PRIVATE BLAS::BLAS)
    target_compile_definitions(nn PRIVATE NN_USE_BLAS)
    string(APPEND NN_CONFIG_DEPS "find_dependency(BLAS)\n")
endif()

# ---------------------------------------------------------------- extras
if(NN_BUILD_EXAMPLES)
    foreach(ex seeds xor sequence mnist_like spiking)
        add_executable(example_${ex} examples/${ex}.cpp)
        target_link_libraries(example_${ex} PRIVATE nn::nn)
    endforeach()
    # copy data so the seeds example finds it next to the binary
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(NN_BUILD_TESTS)
    enable_testing()
    add_executable(nn_tests tests/test_main.cpp)
    target_link_libraries(nn_tests PRIVATE nn::nn)
    target_compile_definitions(nn_tests PRIVATE NN_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    add_test(NAME nn_tests COMMAND nn_tests)
endif()

if(NN_BUILD_BENCHMARKS)
    add_executable(nn_bench benchmarks/bench.cpp)
    target_link_libraries(nn_bench PRIVATE nn::nn)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NN_NATIVE_ARCH)
        target_compile_options(nn_bench PRIVATE -O3 -march=native)
    endif()
endif()

# ---------------------------------------------------------------- install
if(NN_INSTALL)
    include(CMakePackageConfigHelpers)
    install(TARGETS nn EXPORT nnTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/nn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT nnTargets NAMESPACE nn:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nn)
    configure_package_config_file(cmake/nnConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/nnConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nn)
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/nnConfigVersion.cmake
        COMPATIBILITY SameMajorVersion)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nnConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/nnConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nn)
endif()
