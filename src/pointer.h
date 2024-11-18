#pragma once

#include <memory>
#include <cuda_runtime.h>

struct cuda_deleter
{
	void operator() (void* p) const { cudaFree(p); }
};

template<typename T>
using host_ptr = std::unique_ptr<T>;
template<typename T>
using cuda_ptr = std::unique_ptr<T, cuda_deleter>;

template<typename T>
host_ptr<T[]> make_managed_host_array(
	std::size_t elements
)
{
	return std::make_unique<T[]>(elements);
}

template<typename T>
cuda_ptr<T[]> make_managed_cuda_array(
	std::size_t elements,
	unsigned flags = cudaMemAttachGlobal,
	cudaError_t* error = nullptr
)
{
	void * pointer = nullptr;
	auto err = cudaMallocManaged(&pointer, sizeof(T) * elements, flags);
	if(error) *error = err;
	if(!pointer) throw std::bad_alloc();
	return cuda_ptr<T[]>(static_cast<T*>(pointer));
}
