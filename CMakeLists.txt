cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(pmpp_ex1 CUDA CXX)

if(MSVC)
	string(REGEX REPLACE "[-/]W[0-9]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
	string(APPEND CMAKE_CXX_FLAGS " /W4 /WX /wd4127 /wd4800 /D_CRT_SECURE_NO_WARNINGS /D_SCL_SECURE_NO_WARNINGS")
	foreach(TYPE EXE STATIC SHARED MODULE)
		string(APPEND CMAKE_${TYPE}_LINKER_FLAGS " /WX /IGNORE:4199")
	endforeach()
else()
	string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wpedantic -Werror")
	string(APPEND CMAKE_CXX_FLAGS " -Wno-attributes") # old GCC doesn't support [nodiscard]
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
foreach(lang CXX CUDA)
	set(CMAKE_${lang}_STANDARD_REQUIRED ON)
	set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets -Wreorder -Werror=all-warnings")
if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL 11.8.89)
	# workaround for CUDA 11.8 containing /DEFAULTLIB:LIBCMT directives in cudadevrt.lib and cudart_static.lib
	foreach(type EXE SHARED MODULE)
		string(APPEND CMAKE_${type}_LINKER_FLAGS " /NODEFAULTLIB:LIBCMT")
	endforeach()
endif()

add_executable(
	${PROJECT_NAME}
	src/main.cpp
	src/pointer.h
	src/pointer.cpp
	src/waveEvol.h
	src/waveEvol.cu
)

find_package(CUDAToolkit)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

find_package(OpenMP REQUIRED)

if(MSVC)
	# as of VC 19.34, OpenMP is not fully compatible with /permissive- or /std:c++20 (esp. #pragma omp barrier, which is used by thrust)
	target_compile_options(OpenMP::OpenMP_CXX INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/permissive>)
endif()
