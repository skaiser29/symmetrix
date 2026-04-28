cmake_minimum_required(VERSION 3.20)
project(LibSymmetrix)

message(STATUS "Symmetrix: Beginning to process CMakeLists.txt.")

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----- dependencies -----

find_package(BLAS REQUIRED)
option(SYMMETRIX_SPHERICART_SYCL "Use SYCL sphericart for on-device SH computation." OFF)
if(SYMMETRIX_SPHERICART_SYCL)
    set(SPHERICART_ENABLE_SYCL ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(external/sphericart)
option(SYMMETRIX_KOKKOS "Enable Kokkos mode in Symmetrix." ON)
option(SYMMETRIX_BUILD_SHARED "Build symmetrix as a shared library." ${BUILD_SHARED_LIBS})
if(SYMMETRIX_KOKKOS)
    if(TARGET Kokkos::kokkos)
        message(STATUS "Symmetrix: Found existing Kokkos target.")
    else()
        add_subdirectory(external/kokkos)
    endif()
    add_subdirectory(external/kokkos-kernels)
endif()

# ----- libsymmetrix -----

set(SYMMETRIX_SOURCE_FILES
    source/cubic_spline.cpp
    source/cubic_spline_set.cpp
    source/mace.cpp
    source/multilayer_perceptron.cpp
    source/multivariate_polynomial.cpp
    source/spherical_harmonic.cpp
    source/tools.cpp
    source/zbl.cpp)

if(SYMMETRIX_KOKKOS)
    list(APPEND SYMMETRIX_SOURCE_FILES
        source/cubic_spline_kokkos.cpp
        source/cubic_spline_set_kokkos.cpp
        source/mace_kokkos.cpp
        source/multilayer_perceptron_kokkos.cpp
        source/multivariate_polynomial_kokkos.cpp
        source/radial_function_set_kokkos.cpp
        source/tools_kokkos.cpp
        source/zbl_kokkos.cpp)
endif()

if(SYMMETRIX_BUILD_SHARED)
    add_library(symmetrix SHARED ${SYMMETRIX_SOURCE_FILES})
else()
    add_library(symmetrix STATIC ${SYMMETRIX_SOURCE_FILES})
endif()

target_compile_options(symmetrix
    PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:IntelLLVM>>>:-march=native>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:IntelLLVM>>:-xHost>
            $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)

target_include_directories(symmetrix
    PUBLIC  source
    PRIVATE external/json/single_include
            external/sphericart/sphericart/include)

target_link_libraries(symmetrix
    PRIVATE ${BLAS_LIBRARIES}
            sphericart)
if(SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix
        PRIVATE Kokkos::kokkos
                Kokkos::kokkoskernels)
endif()

if(SYMMETRIX_KOKKOS AND Kokkos_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(symmetrix PRIVATE CUDA::cublas CUDA::cublasLt)
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_HAVE_CUBLAS)
endif()

if(SYMMETRIX_KOKKOS AND Kokkos_ENABLE_SYCL)
    find_package(MKL REQUIRED)
    find_library(MKL_SYCL_BLAS_LIB
        NAMES mkl_sycl_blas mkl_sycl
        HINTS ENV MKLROOT
        PATH_SUFFIXES lib lib/intel64)
    if(NOT MKL_SYCL_BLAS_LIB)
        message(FATAL_ERROR
            "Could not find mkl_sycl_blas. Ensure the oneAPI MKL module is loaded "
            "and MKLROOT is set.")
    endif()
    message(STATUS "Symmetrix: Found MKL SYCL BLAS: ${MKL_SYCL_BLAS_LIB}")
    target_link_libraries(symmetrix PRIVATE MKL::MKL ${MKL_SYCL_BLAS_LIB})
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_HAVE_ONEMKL)
endif()


# ----- options -----

option(SYMMETRIX_SPHERICART_CUDA OFF)
if (SYMMETRIX_SPHERICART_CUDA)
    message(STATUS "Symmetrix: Will use CUDA version of sphericart.")
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_CUDA)
endif()

if(SYMMETRIX_SPHERICART_SYCL)
    message(STATUS "Symmetrix: Will use SYCL version of sphericart.")
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_SYCL)
endif()
