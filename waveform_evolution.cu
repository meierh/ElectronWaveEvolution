#include <waveform_evolution.hpp>

#include <iostream>
#include <chrono>

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

void collisionEvaluation
(
	cuda::std::span<std::uint64_t const> const & device_wavefunction,
	std::uint64_t activation,
	std::uint64_t deactivation,
	pmpp::cuda_ptr<bool[]>& collisions,
	pmpp::cuda_ptr<std::uint64_t[]>& non_collision_offset
)
{
	cudaError_t allocError;
	dim3 gridSz;

	std::size_t collision_size = device_wavefunction.size();
	collisions = pmpp::make_managed_cuda_array<bool>(collision_size,cudaMemAttachGlobal,&allocError);
	non_collision_offset = pmpp::make_managed_cuda_array<std::uint64_t>(collision_size,cudaMemAttachGlobal,&allocError);
	constexpr uint num_threads = 64;
	gridSz = { (static_cast<uint>(device_wavefunction.size())/num_threads)+1 };
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
}

template<uint num_threads>
concept ThreadsOK = num_threads%8==0 && num_threads>0 && num_threads<30000;

template<uint num_threads>
__global__ void inclusive_scan_kernel
(
	std::uint64_t* sequence,
	std::uint64_t len,
	std::uint64_t* sequence_endBlock
)
requires ThreadsOK<num_threads>
{
	std::uint64_t wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ std::uint64_t offsets[num_threads];
	offsets[threadIdx.x] = 0;
	if(wave_data_index < len)
	{
		offsets[threadIdx.x] = sequence[wave_data_index];
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

	if(wave_data_index < len)
		sequence[wave_data_index] = offsets[threadIdx.x];

	if(threadIdx.x==blockDim.x-1)
		sequence_endBlock[blockIdx.x] = offsets[threadIdx.x];
	if(wave_data_index == len-1)
		sequence_endBlock[blockIdx.x] = offsets[threadIdx.x];
}


template<uint num_threads>
__global__ void addition_scan_kernel
(
	std::uint64_t* sequence,
	std::uint64_t len,
	std::uint64_t* sequence_endBlock,
	std::uint16_t blockOffset
)
requires ThreadsOK<num_threads>
{
	if(blockOffset < blockIdx.x)
	{
		std::uint64_t addition = sequence_endBlock[blockIdx.x-1];
		std::uint64_t wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
		if(wave_data_index<len)
			sequence[wave_data_index] += addition;
		__syncthreads();

		if(blockIdx.x<gridDim.x-1)
		{
			if(threadIdx.x==blockDim.x-1)
				sequence_endBlock[blockIdx.x] = addition;
		}
	}
}

void inclusive_scan
(
	std::uint64_t* sequence,
	std::uint64_t len
)
{
	cudaError_t allocError;
	constexpr uint blockSize = 1024;
	std::uint64_t gridSize = (len/blockSize) + 1;

	pmpp::cuda_ptr<std::uint64_t[]> sequence_endBlock = pmpp::make_managed_cuda_array<std::uint64_t>(gridSize,cudaMemAttachGlobal,&allocError);

	inclusive_scan_kernel<blockSize><<<dim3(gridSize),dim3(blockSize)>>>
	(
		sequence,
		len,
		sequence_endBlock.get()
	);

	for(std::uint64_t blockOffset=0; blockOffset<gridSize-1; blockOffset++)
	{
		addition_scan_kernel<blockSize><<<dim3(gridSize),dim3(blockSize)>>>
		(
			sequence,
			len,
			sequence_endBlock.get(),
			blockOffset
		);
		cudaDeviceSynchronize();
	}
}

void computeOffsets
(
	const cuda::std::span<std::uint64_t const>& device_wavefunction,
	pmpp::cuda_ptr<std::uint64_t[]>& non_collision_offset,
	std::uint64_t& maxOffset
)
{
	maxOffset = std::numeric_limits<std::uint64_t>::max();
	inclusive_scan(non_collision_offset.get(),device_wavefunction.size());
	std::uint64_t* non_collision_offset_ptr = non_collision_offset.get();
	non_collision_offset_ptr += (device_wavefunction.size()-1);
	cudaMemcpy
	(
		&maxOffset,
		non_collision_offset_ptr,
		sizeof(std::uint64_t),
		cudaMemcpyDeviceToHost
	);
}

__global__ void evolve_kernel
(
	const std::uint64_t* wave_data,
	std::uint64_t wave_data_len,
	std::uint64_t activation,
	std::uint64_t deactivation,
	const bool* collision,
	const std::uint64_t* non_collision_offset,
 	std::uint64_t* wave_added
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
			wave_added[wave_offset-1] = new_wave;
		}
	}
}

void evolutionEvaluation
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	std::uint64_t activation,
	std::uint64_t deactivation,
	const pmpp::cuda_ptr<bool[]>& collisions,
	const pmpp::cuda_ptr<std::uint64_t[]>& non_collision_offset,
	std::uint64_t maxOffset,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added
)
{
	cudaError_t allocError;
	wave_added = pmpp::make_managed_cuda_array<std::uint64_t>(maxOffset,cudaMemAttachGlobal,&allocError);
	uint threadNum = 64;
	dim3 blockSz = { threadNum };
	dim3 gridSz = { (static_cast<uint>(device_wavefunction.size())/threadNum)+1 };
	evolve_kernel<<<gridSz,blockSz>>>
	(
		device_wavefunction.data(),
		device_wavefunction.size(),
		activation,
		deactivation,
		collisions.get(),
		non_collision_offset.get(),
		wave_added.get()
	);
	cudaDeviceSynchronize();
}

__global__ void duplicateDetection_kernel
(
	const std::uint64_t* wave_data,
	std::uint64_t wave_data_len,
	const std::uint64_t* wave_added,
	std::uint64_t wave_added_len,
	uint* duplicate
)
{
	std::uint64_t wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_data_index < wave_data_len)
	{
		std::uint64_t wave = wave_data[wave_data_index];
		for(std::uint64_t wave_added_index=0; wave_added_index<wave_added_len; wave_added_index++)
		{
			std::uint64_t one_wave_added = wave_added[wave_added_index];
			if(one_wave_added==wave)
			{
				uint duplicateCount = duplicate[wave_added_index];
				if(duplicateCount==0)
					atomicAdd((duplicate+wave_added_index), 1);
			}
		}
	}
}

void detectDuplicates
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	std::uint64_t maxOffset,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	pmpp::cuda_ptr<uint[]>& duplicate
)
{
	cudaError_t allocError;
	duplicate = pmpp::make_managed_cuda_array<uint>(maxOffset,cudaMemAttachGlobal,&allocError);
	std::vector<uint> zeros(maxOffset);
	std::fill(zeros.begin(),zeros.end(),0);
	cudaMemcpy(duplicate.get(),zeros.data(),maxOffset*sizeof(uint),cudaMemcpyHostToDevice);

	uint num_threads = 64;
	dim3 blockSz = { num_threads };
	dim3 gridSz = { (static_cast<uint>(device_wavefunction.size())/num_threads)+1 };
	duplicateDetection_kernel<<<gridSz,blockSz>>>
	(
		device_wavefunction.data(),
		device_wavefunction.size(),
		wave_added.get(),
		maxOffset,
		duplicate.get()
	);
	cudaDeviceSynchronize();
}

__global__ void duplicateToOffset_kernel
(
	std::uint64_t wave_added_len,
	const uint* duplicate,
	bool* isDuplicate,
	std::uint64_t* nonduplicateOffset
)
{
	std::uint64_t wave_added_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_added_index < wave_added_len)
	{
		bool isDuplicateReg = (duplicate[wave_added_index]!=0);
		isDuplicate[wave_added_index] = isDuplicateReg;
		nonduplicateOffset[wave_added_index] = isDuplicateReg ? 0 : 1;
	}
}

void duplicatesToOffset
(
	std::uint64_t maxOffset,
	pmpp::cuda_ptr<uint[]>& duplicate,
	pmpp::cuda_ptr<bool[]>& isDuplicate,
	pmpp::cuda_ptr<std::uint64_t[]>& nonduplicateOffset
)
{
	cudaError_t allocError;
	isDuplicate = pmpp::make_managed_cuda_array<bool>(maxOffset,cudaMemAttachGlobal,&allocError);
	nonduplicateOffset = pmpp::make_managed_cuda_array<std::uint64_t>(maxOffset,cudaMemAttachGlobal,&allocError);

	uint num_threads = 64;
	dim3 blockSz = { num_threads };
	dim3 gridSz = { (static_cast<uint>(maxOffset)/num_threads)+1 };
	duplicateToOffset_kernel<<<gridSz,blockSz>>>(maxOffset,duplicate.get(),isDuplicate.get(),nonduplicateOffset.get());	cudaDeviceSynchronize();
}

void duplicateFinalizeOffset
(
	pmpp::cuda_ptr<std::uint64_t[]>& nonduplicateOffset,
	std::uint64_t maxOffset,
	std::uint64_t& reducedMaxOffset
)
{
	inclusive_scan(nonduplicateOffset.get(),maxOffset);
	cudaMemcpy
	(
		&reducedMaxOffset,
		nonduplicateOffset.get()+maxOffset-1,
		sizeof(std::uint64_t),
		cudaMemcpyDeviceToHost
	);
}

__global__ void duplicateRemoval_kernel
(
	const std::uint64_t* wave_added,
	std::uint64_t wave_added_len,
	const bool* isDuplicate,
	const std::uint64_t* nonduplicateOffset,
 	std::uint64_t* reduced_wave_added
)
{
	// TODO: Compute evolved data and store in dst_data according to offsets numbers
	std::uint64_t wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_data_index < wave_added_len)
	{
		std::uint64_t wave = wave_added[wave_data_index];
		bool nonDuplicate = !isDuplicate[wave_data_index];
		std::uint64_t wave_offset = nonduplicateOffset[wave_data_index];

		if(nonDuplicate)
		{
			reduced_wave_added[wave_offset-1] = wave;
		}
	}
}

void removeDuplicates
(
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	std::uint64_t maxOffset,
	pmpp::cuda_ptr<bool[]>& isDuplicate,
	pmpp::cuda_ptr<std::uint64_t[]>& nonduplicateOffset,
	std::uint64_t reducedMaxOffset
)
{
	if(reducedMaxOffset>0)
	{
		cudaError_t allocError;
		pmpp::cuda_ptr<std::uint64_t[]> reduced_wave_added;
		reduced_wave_added = pmpp::make_managed_cuda_array<std::uint64_t>(reducedMaxOffset,cudaMemAttachGlobal,&allocError);


		uint num_threads = 64;
		dim3 blockSz = { num_threads };
		dim3 gridSz = { (static_cast<uint>(maxOffset)/num_threads)+1 };
		duplicateRemoval_kernel<<<gridSz,blockSz>>>
		(
			wave_added.get(),
			maxOffset,
			isDuplicate.get(),
			nonduplicateOffset.get(),
			reduced_wave_added.get()
		);
		cudaDeviceSynchronize();

		wave_added = std::move(reduced_wave_added);
	}
	else
		wave_added.release();
}

void treatDuplicates
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	std::uint64_t maxOffset,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	std::uint64_t& reducedMaxOffset
)
{
	using best_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	/*
	* Detect duplicates
	*/
	auto t1 = best_clock::now();
	pmpp::cuda_ptr<uint[]> duplicate;
	detectDuplicates(device_wavefunction,maxOffset,wave_added,duplicate);
	auto t2 = best_clock::now();
	std::uint64_t milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"    Detect duplicates took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	/*
	* Duplicates to offset
	*/
	t1 = best_clock::now();
	pmpp::cuda_ptr<bool[]> isDuplicate;
	pmpp::cuda_ptr<std::uint64_t[]> nonduplicateOffset;
	duplicatesToOffset(maxOffset,duplicate,isDuplicate,nonduplicateOffset);
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"    Duplicates to offset took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	/*
	* Duplicates to offset
	*/
	t1 = best_clock::now();
	duplicateFinalizeOffset(nonduplicateOffset,maxOffset,reducedMaxOffset);
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"    Finalize duplicate offset took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	/*
	* Remove duplicates
	*/
	t1 = best_clock::now();
	removeDuplicates(wave_added,maxOffset,isDuplicate,nonduplicateOffset,reducedMaxOffset);
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"    Remove duplicates took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{
	using best_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	cudaError_t allocError;
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> waveOut;

	/*
	 * Compute collision data
	 */
	auto t1 = best_clock::now();
	pmpp::cuda_ptr<bool[]> collisions;
	pmpp::cuda_ptr<std::uint64_t[]> non_collision_offset;
	collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);
	auto t2 = best_clock::now();
	std::uint64_t milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"  Compute collision took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	/*
	 * Compute offsets
	 */
	t1 = best_clock::now();
	std::uint64_t maxOffset;
	computeOffsets(device_wavefunction,non_collision_offset,maxOffset);
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"  Compute offsets took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	if(maxOffset>0)
	{
		/*
		* Compute evolution
		*/
		t1 = best_clock::now();
		pmpp::cuda_ptr<std::uint64_t[]> wave_added;
		evolutionEvaluation
		(
			device_wavefunction,
			activation,
			deactivation,
			collisions,
			non_collision_offset,
			maxOffset,
			wave_added
		);
		t2 = best_clock::now();
		milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout<<"  Compute evolution took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

		/*
		* Treat duplicates
		*/
		std::cout<<"maxOffset:"<<maxOffset<<std::endl;
		std::uint64_t reducedMaxOffset;
		treatDuplicates(device_wavefunction,maxOffset,wave_added,reducedMaxOffset);

		/*
		* Compose
		*/
		t1 = best_clock::now();
		waveOut.second = device_wavefunction.size()+reducedMaxOffset;
		waveOut.first = pmpp::make_managed_cuda_array<std::uint64_t>
		(
			device_wavefunction.size()+reducedMaxOffset,
			cudaMemAttachGlobal,
			&allocError
		);
		cudaMemcpy
		(
			waveOut.first.get(),
			device_wavefunction.data(),
			device_wavefunction.size()*sizeof(std::uint64_t),
			cudaMemcpyDeviceToDevice
		);
		if(reducedMaxOffset>0)
		{
			cudaMemcpy
			(
				waveOut.first.get()+device_wavefunction.size(),
				wave_added.get(),
				reducedMaxOffset*sizeof(std::uint64_t),
				cudaMemcpyDeviceToDevice
			);
		}
		t2 = best_clock::now();
		milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout<<"  Compose took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;
	}
	else
	{
		t1 = best_clock::now();
		waveOut.second = device_wavefunction.size();
		waveOut.first = pmpp::make_managed_cuda_array<std::uint64_t>
		(
			device_wavefunction.size(),
			cudaMemAttachGlobal,
			&allocError
		);
		cudaMemcpy
		(
			waveOut.first.get(),
			device_wavefunction.data(),
			device_wavefunction.size()*sizeof(std::uint64_t),
			cudaMemcpyDeviceToDevice
		);
		t2 = best_clock::now();
		milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout<<"  Compose took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;
	}

	return waveOut;
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations
)
{
	/* TODO */
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result;
	for(std::uint64_t operatorInd=0; operatorInd<activations.size(); operatorInd++)
	{
		result = evolve_operator(device_wavefunction,activations[operatorInd],deactivations[operatorInd]);
		device_wavefunction = cuda::std::span<std::uint64_t const>(result.first.get(),result.second);
	}
	return result;
}
