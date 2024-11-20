#include <waveform_evolution.hpp>

#include <iostream>

__global__ void check_collision_kernel
(
	const std::uint64_t* wave_data,
	std::uint64_t wave_data_len,
	std::uint64_t activation,
	std::uint64_t deactivation,
	bool* collision,
	std::uint64_t* non_collision_offset
)
{
	// TODO: Compute of collision occures and set bit in collision_bits if it happens
	std::uint64_t wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_data_index<wave_data_len)
	{
		std::uint64_t wave = wave_data[wave_data_index];
		bool col = (bool)((wave & activation) | ((~wave) & deactivation));
		collision[wave_data_index] = col;
		non_collision_offset[wave_data_index] = col ? 0 : 1;
	}
}

template<uint num_threads>
concept ThreadsOK = num_threads%8==0 && num_threads>0 && num_threads<30000;

template<uint num_threads>
__global__ void inclusive_scan_kernel
(
	std::uint64_t* non_collision_offset,
	std::uint64_t wave_data_len,
	std::uint64_t* non_collision_offset_endBlock,
	std::uint16_t blockOffset
)
requires ThreadsOK<num_threads>
{
	std::uint64_t wave_data_index = blockDim.x*(blockIdx.x+blockOffset) + threadIdx.x;
	__shared__ std::uint64_t offsets[num_threads];
	offsets[threadIdx.x] = 0;
	if(wave_data_index<wave_data_len)
	{
		offsets[threadIdx.x] = non_collision_offset[wave_data_index];
	}
	if(threadIdx.x==0 && blockIdx.x>blockOffset)
	{
		offsets[threadIdx.x] += non_collision_offset_endBlock[blockIdx.x];
	}
	__syncthreads();

	for(uint stride = 1; stride<=blockDim.x; stride*=2)
	{
		__syncthreads();
		uint indx = (threadIdx.x+1)*2*stride-1;
		if(indx<num_threads)
			offsets[indx] += offsets[indx-stride];
	}
	for(int stride = num_threads/4; stride>0; stride/=2)
	{
		__syncthreads();
		uint indx = (threadIdx.x+1)*2*stride-1;
		if(indx+stride<num_threads)
			offsets[indx+stride] += offsets[indx];
	}
	__syncthreads();

	if(wave_data_index<wave_data_len)
	{
		non_collision_offset[wave_data_index] = offsets[threadIdx.x];
	}

	if(threadIdx.x==blockDim.x-1)
		non_collision_offset_endBlock[blockIdx.x+blockOffset] = offsets[blockDim.x-1];
}

void inclusive_scan
(
	std::uint64_t* non_collision_offset,
	std::uint64_t wave_data_len
)
{
	cudaError_t allocError;
	constexpr uint blockSize = 32;
	int gridSize = wave_data_len/blockSize;

	pmpp::cuda_ptr<std::uint64_t[]> non_collision_offset_endBlock = pmpp::make_managed_cuda_array<std::uint64_t>(gridSize,cudaMemAttachGlobal,&allocError);

	for(std::uint16_t blockOffset=0; blockOffset<gridSize; blockOffset++, gridSize--)
	{
		inclusive_scan_kernel<blockSize><<<dim3(gridSize),dim3(blockSize)>>>
		(
			non_collision_offset,
			wave_data_len,
			non_collision_offset_endBlock.get(),
			blockOffset
		);
	}
}

__global__ void evolve_kernel
(
	const std::uint64_t* wave_data,
	std::uint64_t wave_data_len,
	std::uint64_t activation,
	std::uint64_t deactivation,
	const bool* collision,
	const std::uint64_t* non_collision_offset,
 	std::uint64_t* wave_data_out
)
{
	// TODO: Compute evolved data and store in dst_data according to offsets numbers
	std::uint64_t wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_data_index < wave_data_len)
	{
		std::uint64_t wave = wave_data[wave_data_index];
		bool wave_collision = collision[wave_data_index];
		std::uint64_t wave_offset = non_collision_offset[wave_data_index];

		if(!wave_collision)
		{
			std::uint64_t new_wave = wave;
			new_wave |= activation;
			new_wave &= ~deactivation;
			wave_data_out[wave_data_len+wave_offset] = new_wave;
		}
	}
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{
	/* TODO */
	cudaError_t allocError,cpyError;
	dim3 blockSz;
	dim3 gridSz;

	/*
	 * Compute collision data
	 */
	std::size_t collision_size = device_wavefunction.size()/(sizeof(std::uint64_t));
	pmpp::cuda_ptr<bool[]> collisions;
	collisions = pmpp::make_managed_cuda_array<bool>(collision_size,cudaMemAttachGlobal,&allocError);
	pmpp::cuda_ptr<std::uint64_t[]> non_collision_offset;
	non_collision_offset = pmpp::make_managed_cuda_array<std::uint64_t>(1,cudaMemAttachGlobal,&allocError);
	constexpr uint num_threads = 32;
	gridSz = { (static_cast<uint>(device_wavefunction.size())/2)+1 };
	check_collision_kernel<<<gridSz,dim3(num_threads)>>>
	(
		device_wavefunction.data(),
		device_wavefunction.size(),
		activation,
		deactivation,
		collisions.get(),
		non_collision_offset.get()
	);
	cudaDeviceSynchronize();

	/*
	 * Compute offsets
	 */
	inclusive_scan(non_collision_offset.get(),device_wavefunction.size());
	std::uint64_t maxOffset;
	cudaMemcpy
	(
		&maxOffset,
		non_collision_offset.get()+device_wavefunction.size()-1,
		sizeof(std::uint64_t),
		cudaMemcpyDeviceToDevice
	);

	/*
	 * Compute evolution
	 */
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> waveOut;
	pmpp::cuda_ptr<std::uint64_t[]>& wave_data_out = waveOut.first;
	waveOut.second = device_wavefunction.size()+maxOffset;
	wave_data_out = pmpp::make_managed_cuda_array<std::uint64_t>
	(
		device_wavefunction.size()+maxOffset,
		cudaMemAttachGlobal,
		&allocError
	);
	blockSz = { 32 };
	gridSz = { 123 };
	evolve_kernel<<<gridSz,blockSz>>>
	(
		device_wavefunction.data(),
		device_wavefunction.size(),
		activation,
		deactivation,
		collisions.get(),
		non_collision_offset.get(),
		wave_data_out.get()
	);
	cpyError = cudaMemcpy
	(
		wave_data_out.get(),
		device_wavefunction.data(),
		device_wavefunction.size()*sizeof(std::uint64_t),
		cudaMemcpyDeviceToDevice
	);
	cudaDeviceSynchronize();
	std::cout<<"Error: "<<cudaGetErrorName(cpyError)<<"  --  "<<cudaGetErrorString(cpyError)<<std::endl;

	return waveOut;
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations
)
{
	/* TODO */
	for(std::uint64_t operatorInd=0; operatorInd<activations.size(); operatorInd++)
	{
		// cuda::std::span -> pmpp::cuda_ptr -> cuda::std::span sucks!!!
	}
	return {nullptr, 0};
}
