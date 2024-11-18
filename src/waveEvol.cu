#include "waveEvol.h"

template<typename T>
__global__ void evolve_kernel
(
	T* src_data,
	T src_data_len,
	T activation,
	T deactivation,
	T* dst_data, 
	T* dst_data_len
)
{
}

template<typename T>
void evolve(std::vector<T>& wave, T activation, T deactivation)
{
	cudaError_t allocError,cpyError;
	
	cuda_ptr<T> waveIn = make_managed_cuda_array<T>(wave.size(),cudaMemAttachGlobal,&allocError);
	cpyError = cudaMemcpy(waveIn.get(),wave.data(),wave.size()*sizeof(T),cudaMemcpyHostToDevice);
	
	cuda_ptr<T> waveOut = make_managed_cuda_array<T>(wave.size(),cudaMemAttachGlobal,&allocError);
	
	cuda_ptr<T> waveOutLen = make_managed_cuda_array<T>((uint)1,cudaMemAttachGlobal,&allocError);
	
	dim3 blockSz = { 32 };
	dim3 gridSz = { (wave.size()/blockSz.x)+1 };
	evolve_kernel<<<gridSz,blockSz>>>(waveIn.get(),wave.size(),activation,deactivation,waveOut.get(),waveOutLen.get());
	
	T outLen;
	cpyError = cudaMemcpy(&outLen,waveOutLen.get(),sizeof(T),cudaMemcpyDeviceToHost);
	wave.resize(outLen);
	
	cpyError = cudaMemcpy(wave.data(),waveOut.get(),outLen*sizeof(T),cudaMemcpyDeviceToHost);
}
