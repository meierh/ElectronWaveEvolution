#include "waveEvol.h"

template<typename T>
__global__ void check_collision_kernel
(
	T* src_data,
	T src_data_len,
	T activation,
	T deactivation,
	T* collision_bits
)
{
	// TODO: Compute of collision occures and set bit in collision_bits if it happens
}

template<typename T>
__global__ void compute_offsets_kernel
(
	T* collisions,
	T src_data_len,
	T* offsets
)
{
	// TODO: Compute offsets from collision_bits
}

template<typename T>
__global__ void evolve_kernel
(
	T* src_data,
	T src_data_len,
	T activation,
	T deactivation,
	T* offsets,
	T* dst_data
)
{
	// TODO: Compute evolved data and store in dst_data according to offsets numbers
}

template<typename T>
void evolve(std::vector<T>& wave, T activation, T deactivation)
{
	cudaError_t allocError,cpyError;
	dim3 blockSz;
	dim3 gridSz;
	
	cuda_ptr<T> waveIn = make_managed_cuda_array<T>(wave.size(),cudaMemAttachGlobal,&allocError);
	cpyError = cudaMemcpy(waveIn.get(),wave.data(),wave.size()*sizeof(T),cudaMemcpyHostToDevice);
	
	cuda_ptr<T> collisions = make_managed_cuda_array<T>(/* change later */wave.size(),cudaMemAttachGlobal,&allocError);
	
	blockSz = { 32 };
	gridSz = { (wave.size()/blockSz.x)+1 };
	check_collision_kernel<<<gridSz,blockSz>>>(waveIn.get(),wave.size(),activation,deactivation,collisions.get());
	
	cuda_ptr<T> offsets = make_managed_cuda_array<T>(/* change later */wave.size(),cudaMemAttachGlobal,&allocError);
	
	blockSz = { 32 };
	gridSz = { (wave.size()/blockSz.x)+1 };
	compute_offsets_kernel<<<gridSz,blockSz>>>(collisions.get(),wave.size(),offsets.get());
	
	T outLen;
	/*
	 * compute outLen from offsets
	 */
	
	cuda_ptr<T> waveOut = make_managed_cuda_array<T>(outLen,cudaMemAttachGlobal,&allocError);
		
	blockSz = { 32 };
	gridSz = { (wave.size()/blockSz.x)+1 };
	evolve_kernel<<<gridSz,blockSz>>>(waveIn.get(),wave.size(),activation,deactivation,offsets.get(),waveOut.get());
	
	wave.resize(outLen);
	cpyError = cudaMemcpy(wave.data(),waveOut.get(),outLen*sizeof(T),cudaMemcpyDeviceToHost);
}
