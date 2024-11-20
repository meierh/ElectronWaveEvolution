#include <waveform_evolution.hpp>

template<uint num_threads>
concept ThreadsOK = num_threads%8==0 && num_threads>0;

template<uint num_threads>
__global__ void check_collision_kernel
(
	std::uint64_t* wave_data,
	std::uint64_t wave_data_len,
	std::uint64_t activation,
	std::uint64_t deactivation,
	std::uint8_t* collision_bits
)
requires ThreadsOK<num_threads>
{
	// TODO: Compute of collision occures and set bit in collision_bits if it happens

	__shared__ bool collision[num_threads];
	num_threads[threadIdx.x] = false;
	std::uint64_t wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;

	std::uint64_t wave = 0;
	if(wave_data_index<wave_data_len)
	{
		wave = wave_data[wave_data_index];
	}
	collision[threadIdx.x] = (bool)((wave & activation) | ((~wave) & deactivation));
	__syncthreads();

	if(threadIdx.x%8==0 && wave_data_index<wave_data_len)
	{
		std::uint8_t col_bits = 0;
		for(uint i=; i<8; i++)
		{
			std::uint64_t blockGlobalIndex = i+threadIdx.x;
			std::uint8_t bit = collision[blockGlobalIndex] ? 1 : 0;
			col_bits |= bit << (7-blockGlobalIndex);
		}
		std::uint64_t collision_bits_index = wave_data_index/8;
		collision_bits[collision_bits_index] = col_bits;
	}
}

__global__ void compute_offsets_kernel
(
	std::uint8_t* collisions,
	std::uint64_t wave_data_len,
	std::uint64_t* offsets
)
{
	// TODO: Compute offsets from collision_bits for added waves
}

__global__ void evolve_kernel
(
	std::uint64_t* wave_data,
	std::uint64_t wave_data_len,
	std::uint64_t activation,
	std::uint64_t deactivation,
	std::uint64_t* offsets,
	std::uint64_t offsets_len,
 	std::uint64_t* wave_data_out
)
{
	// TODO: Compute evolved data and store in dst_data according to offsets numbers
	std::uint64_t globalIndex = blockDim.x*blockIdx.x + threadIdx.x;
	if(globalIndex < offsets_len)
	{
		std::uint64_t wave_data_index = offsets[globalIndex];
		std::uint64_t wave = wave_data[wave_data_index];
		std::uint64_t new_wave = wave;
		new_wave |= activation;
		new_wave &= ~deactivation;
		wave_data_out[wave_data_len+globalIndex] = new_wave;
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
	cuda_ptr<std::uint8_t> collisions;
	collisions = make_managed_cuda_array<std::uint8_t>(collision_size,cudaMemAttachGlobal,&allocError);

	blockSz = { 32 };
	gridSz = { (wave.size()/blockSz.x)+1 };
	check_collision_kernel<<<gridSz,blockSz>>><blockSz>
	(
		device_wavefunction.data(),device_wavefunction.size(),
		activation,deactivation,
		collisions.get()
	);
	cudaDeviceSynchronize();

	/*
	 * Compute offsets
	 */
	cuda_ptr<std::uint64_t> offsets;
	offsets = make_managed_cuda_array<std::uint64_t>(collision_size,cudaMemAttachGlobal,&allocError);
	blockSz = { 32 };
	gridSz = { (wave.size()/blockSz.x)+1 };
	compute_offsets_kernel<<<gridSz,blockSz>>>
	(
		collisions.get(),data(),
		device_wavefunction.size(),
		offsets.get()
	);
	cudaDeviceSynchronize();
	std::uint64_t maxOffset; // Compute max offset

	/*
	 * Compute evolution
	 */
	cuda_ptr<std::uint64_t> wave_data_out;
	wave_data_out = make_managed_cuda_array<std::uint64_t>
	(
		device_wavefunction.size()+maxOffset,
		cudaMemAttachGlobal,
		&allocError
	);
	blockSz = { 32 };
	gridSz = { (wave.size()/blockSz.x)+1 };
	evolve_kernel<<<gridSz,blockSz>>>
	(
		device_wavefunction.data(),device_wavefunction.size(),
		activation,deactivation,
		collisions.get(),collision_size,
		wave_data_out.data(),
	);
	cudaMemcpy
	(
		wave_data_out.data(),
		device_wavefunction.data(),
		device_wavefunction.size()*sizeof(std::uint64_t),
		cudaMemcpyDeviceToDevice
	);

	return {wave_data_out, device_wavefunction.size()+maxOffset};
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations
)
{
	/* TODO */
	for(label operatorInd=0; operatorInd<activations.size(); operatorInd++)
	{
		// cuda::std::span -> pmpp::cuda_ptr -> cuda::std::span sucks!!!
	}
	return {nullptr, 0};
}
