#include <waveform_evolution.hpp>

#include <iostream>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define MEASURE_TIME 0

__global__ void check_collision_kernel
(
	const std::uint64_t* wave_data,
	std::uint64_t wave_data_len,
	std::uint64_t activation,
	std::uint64_t deactivation,
	bool* collision,
	waveSizeCountType* non_collision_offset
)
{
	waveSizeCountType wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_data_index<wave_data_len)
	{
		std::uint64_t wave = wave_data[wave_data_index];
		bool col = (bool)((wave & activation) | ((~wave) & deactivation));
		collision[wave_data_index] = col;
		non_collision_offset[wave_data_index] = static_cast<waveSizeCountType>(!col);
	}
}

__host__ void collisionEvaluation
(
	cuda::std::span<std::uint64_t const> const & device_wavefunction,
	std::uint64_t activation,
	std::uint64_t deactivation,
	pmpp::cuda_ptr<bool[]>& collisions,
	pmpp::cuda_ptr<waveSizeCountType[]>& non_collision_offset
)
{
	cudaError_t allocError;
	dim3 gridSz;

	std::size_t collision_size = device_wavefunction.size();
	collisions = pmpp::make_cuda_array<bool>(collision_size,&allocError);
	non_collision_offset = pmpp::make_cuda_array<waveSizeCountType>(collision_size,&allocError);
	constexpr uint num_threads = 1024;
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

template<uint num_threads,typename seqT,typename sizeT>
__global__ void inclusive_scan_kernel
(
	seqT* sequence,
	sizeT len,
	sizeT* sequenceBlockCarry
)
requires ThreadsOK<num_threads>
{
	waveSizeCountType wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
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
		sequenceBlockCarry[blockIdx.x] = offsets[threadIdx.x];
	if(wave_data_index == len-1)
		sequenceBlockCarry[blockIdx.x] = offsets[threadIdx.x];
}

template<typename seqT,typename sizeT>
__global__ void addition_carry_kernel
(
	seqT* sequence,
	sizeT len,
	sizeT* sequenceBlockCarry
)
{
	if(blockIdx.x>0)
	{
		sizeT carry = sequenceBlockCarry[blockIdx.x-1];
		waveSizeCountType sequenceIndex = blockDim.x*blockIdx.x + threadIdx.x;
		if(sequenceIndex<len)
			sequence[sequenceIndex] += carry;
	}
}

template<typename seqT,typename sizeT>
__host__ void inclusive_scan
(
	seqT* sequence,
	sizeT len
)
{
	cudaError_t allocError;

	constexpr uint blockSize = 1024;
	std::uint64_t gridSize;
	if(len % blockSize == 0)
		gridSize = len/blockSize;
	else
		gridSize = (len/blockSize)+1;

	pmpp::cuda_ptr<sizeT[]> sequenceBlockCarry = pmpp::make_cuda_array<sizeT>(gridSize,&allocError);

	inclusive_scan_kernel<blockSize,seqT,sizeT><<<dim3(gridSize),dim3(blockSize)>>>
	(
		sequence,
		len,
		sequenceBlockCarry.get()
	);
	cudaDeviceSynchronize();

	if(gridSize>1)
	{
		inclusive_scan<seqT,sizeT>(sequenceBlockCarry.get(),gridSize);

		addition_carry_kernel<seqT,sizeT><<<dim3(gridSize),dim3(blockSize)>>>
		(
			sequence,
			len,
			sequenceBlockCarry.get()
		);
		cudaDeviceSynchronize();
	}
}

__host__ void computeOffsets
(
	const cuda::std::span<std::uint64_t const>& device_wavefunction,
	pmpp::cuda_ptr<waveSizeCountType[]>& non_collision_offset,
	waveSizeCountType& maxOffset
)
{
	inclusive_scan<waveSizeCountType,waveSizeCountType>(non_collision_offset.get(),device_wavefunction.size());
	waveSizeCountType* non_collision_offset_ptr = non_collision_offset.get();
	non_collision_offset_ptr += (device_wavefunction.size()-1);
	cudaMemcpy
	(
		&maxOffset,
		non_collision_offset_ptr,
		sizeof(waveSizeCountType),
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
	const waveSizeCountType* non_collision_offset,
 	std::uint64_t* wave_added
)
{
	waveSizeCountType wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_data_index < wave_data_len)
	{
		std::uint64_t wave = wave_data[wave_data_index];
		bool wave_collision = collision[wave_data_index];
		waveSizeCountType wave_offset = non_collision_offset[wave_data_index];

		if(!wave_collision)
		{
			std::uint64_t new_wave = wave;
			new_wave |= activation;
			new_wave &= ~deactivation;
			wave_added[wave_offset-1] = new_wave;
		}
	}
}

__host__ void evolutionEvaluation
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	std::uint64_t activation,
	std::uint64_t deactivation,
	const pmpp::cuda_ptr<bool[]>& collisions,
	const pmpp::cuda_ptr<waveSizeCountType[]>& non_collision_offset,
	waveSizeCountType maxOffset,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added
)
{
	cudaError_t allocError;
	wave_added = pmpp::make_cuda_array<std::uint64_t>(maxOffset,&allocError);
	uint threadNum = 1024;
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
	waveSizeCountType wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
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
					*(duplicate+wave_added_index) = 1;
			}
		}
	}
}

__host__ void detectDuplicates
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	waveSizeCountType wave_added_size,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	pmpp::cuda_ptr<uint[]>& duplicate
)
{
	cudaError_t allocError;
	duplicate = pmpp::make_cuda_array<uint>(wave_added_size,&allocError);
	std::vector<uint> zeros(wave_added_size);
	std::fill(zeros.begin(),zeros.end(),0);
	cudaMemcpy(duplicate.get(),zeros.data(),wave_added_size*sizeof(uint),cudaMemcpyHostToDevice);

	uint num_threads = 32;
	dim3 blockSz = { num_threads };
	dim3 gridSz = { (static_cast<uint>(device_wavefunction.size())/num_threads)+1 };
	duplicateDetection_kernel<<<gridSz,blockSz>>>
	(
		device_wavefunction.data(),
		device_wavefunction.size(),
		wave_added.get(),
		wave_added_size,
		duplicate.get()
	);
	cudaDeviceSynchronize();
}

__global__ void setAddedWaveByteTable_kernel
(
	const std::uint64_t* wave_data,
	std::uint64_t wave_data_len,
	int* byteTable
)
{
	waveSizeCountType wave_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_index < wave_data_len)
	{
		std::uint64_t one_wave = wave_data[wave_index];
		std::uint8_t byteWise_one_wave[sizeof(std::uint64_t)];
		memcpy(byteWise_one_wave,&one_wave,sizeof(std::uint64_t));
		for(std::uint16_t byteInd=0; byteInd<sizeof(std::uint64_t); byteInd++)
		{
			std::uint16_t byteTable_offset = byteInd*256;
			std::uint8_t byte_one_wave = byteWise_one_wave[byteInd];
			int* byteTable_ptr = byteTable+byteTable_offset+byte_one_wave;
			*byteTable_ptr = 1;
		}
	}
}

__global__ void duplicateDetectionOnByteTable_kernel
(
	const std::uint64_t* wave_added,
	std::uint64_t wave_added_len,
	const int* byteTable,
	uint* duplicate
)
{
	waveSizeCountType wave_added_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_added_index < wave_added_len)
	{
		std::uint64_t one_wave_added = wave_added[wave_added_index];
		bool isDuplicate = true;

		std::uint8_t byteWise_one_wave_added[sizeof(std::uint64_t)];
		memcpy(byteWise_one_wave_added,&one_wave_added,sizeof(std::uint64_t));
		for(std::uint8_t byteInd=0; byteInd<sizeof(std::uint64_t); byteInd++)
		{
			std::uint16_t byteTable_offset = byteInd*256;
			std::uint8_t byte_one_wave_added = byteWise_one_wave_added[byteInd];
			const int* byteTable_ptr = byteTable+byteTable_offset+byte_one_wave_added;
			if((*byteTable_ptr)==0)
				isDuplicate = false;
		}
		if(isDuplicate)
			duplicate[wave_added_index] = 1;
		else
			duplicate[wave_added_index] = 0;
	}
}

__host__ void detectDuplicatesWithTable
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	std::uint64_t wave_added_size,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	pmpp::cuda_ptr<uint[]>& duplicate
)
{
	cudaError_t allocError;

	std::vector<int> wave_ByteTable_cpu(256*256*(sizeof(std::uint64_t)/2));
	std::fill(wave_ByteTable_cpu.begin(),wave_ByteTable_cpu.end(),0);
	pmpp::cuda_ptr<int[]> wave_ByteTable = pmpp::make_cuda_array<int>(wave_ByteTable_cpu.size(),&allocError);
	cudaMemcpy
	(
		wave_ByteTable.get(),
		wave_ByteTable_cpu.data(),
		wave_ByteTable_cpu.size()*sizeof(int),
		cudaMemcpyHostToDevice
	);

	uint num_threads = 32;
	dim3 blockSz = { num_threads };
	dim3 gridSz = { (static_cast<uint>(device_wavefunction.size())/num_threads)+1 };
	setAddedWaveByteTable_kernel<<<gridSz,blockSz>>>
	(
		device_wavefunction.data(),
		device_wavefunction.size(),
		wave_ByteTable.get()
	);
	cudaDeviceSynchronize();

	duplicate = pmpp::make_cuda_array<uint>(wave_added_size,&allocError);
	num_threads = 32;
	blockSz = { num_threads };
	gridSz = { (static_cast<uint>(device_wavefunction.size())/num_threads)+1 };
	duplicateDetectionOnByteTable_kernel<<<gridSz,blockSz>>>
	(
		wave_added.get(),
		wave_added_size,
		wave_ByteTable.get(),
		duplicate.get()
	);
	cudaDeviceSynchronize();
}

__host__ void sort
(
	std::uint64_t* sequence,
	std::uint64_t len
)
{
	auto sequence_thrust = thrust::device_ptr<std::uint64_t>(sequence);
	thrust::sort(sequence_thrust,sequence_thrust+len);
}

__global__ void halfSequence_kernel
(
	const std::uint64_t* fullSequence,
	std::uint64_t fullSequenceLen,
	std::uint64_t* halfSequence,
	std::uint64_t halfSequenceLen
)
{
	waveSizeCountType halfSequenceIndex = blockDim.x*blockIdx.x + threadIdx.x;
	waveSizeCountType fullSequenceIndex = halfSequenceIndex*2;
	if(fullSequenceIndex < fullSequenceLen)
	{
		std::uint64_t fullSequenceItem = fullSequence[fullSequenceIndex];
		std::uint64_t* halfSequenceItem = halfSequence + halfSequenceIndex;
		*halfSequenceItem = fullSequenceItem;
	}
}

__global__ void iterateValuesPosition_kernel
(
	const std::uint64_t* values,
	std::uint64_t valuesLen,
	std::int64_t* valuesPosition,
	const std::uint64_t* sequence,
	std::uint64_t sequenceLen
)
{
	waveSizeCountType valuesIndex = blockDim.x*blockIdx.x + threadIdx.x;
	if(valuesIndex < valuesLen)
	{
		std::uint64_t value = values[valuesIndex];
		std::int64_t* priorPositionPtr = valuesPosition + valuesIndex;
		std::int64_t priorPosition = *priorPositionPtr;

		if(priorPosition == -1)
		{
			std::uint64_t divider = sequence[1];
			if(value < divider)
				*priorPositionPtr = 0;
			else
				*priorPositionPtr = 1;
		}
		else
		{
			std::uint64_t doubledPosition = priorPosition*2;
			if(doubledPosition==sequenceLen-1)
				*priorPositionPtr = doubledPosition;
			else
			{
				std::uint64_t divider = sequence[doubledPosition+1];
				if(value < divider)
					*priorPositionPtr = doubledPosition;
				else
					*priorPositionPtr = doubledPosition+1;
			}
		}
	}
}

__host__ void findNearestValuesInSortedArray
(
	const std::uint64_t* sortedSequence,
	const std::uint64_t sortedSequenceLen,
	const std::uint64_t* values,
	const std::uint64_t valuesLen,
	std::int64_t* valuesPosition
)
{
	cudaError_t allocError;
	if(sortedSequenceLen>2)
	{
		std::uint64_t halfSize = sortedSequenceLen/2;
		if(sortedSequenceLen%2!=0)
			halfSize++;

		pmpp::cuda_ptr<std::uint64_t[]> sortedSequenceHalfLen = pmpp::make_cuda_array<std::uint64_t>(halfSize,&allocError);

		uint num_threads = 1024;
		dim3 blockSz = { num_threads };
		dim3 gridSz = { (static_cast<uint>(halfSize)/num_threads)+1 };
		halfSequence_kernel<<<gridSz,blockSz>>>
		(
			sortedSequence,
			sortedSequenceLen,
			sortedSequenceHalfLen.get(),
			halfSize
		);
		cudaDeviceSynchronize();
		findNearestValuesInSortedArray(sortedSequenceHalfLen.get(),halfSize,values,valuesLen,valuesPosition);
	}

	uint num_threads = 1024;
	dim3 blockSz = { num_threads };
	dim3 gridSz = { (static_cast<uint>(valuesLen)/num_threads)+1 };
	iterateValuesPosition_kernel<<<gridSz,blockSz>>>
	(
		values,
		valuesLen,
		valuesPosition,
		sortedSequence,
		sortedSequenceLen
	);
	cudaDeviceSynchronize();
}

__global__ void duplicateDetectionWithSorting_kernel
(
	const std::uint64_t* wave_added,
	std::uint64_t wave_added_len,
	std::int64_t* wave_added_position,
	const std::uint64_t* wave_data,
	std::uint64_t wave_data_len,
	uint* duplicate
)
{
	waveSizeCountType wave_added_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_added_index < wave_added_len)
	{
		std::uint64_t one_wave_added = wave_added[wave_added_index];
		std::uint64_t one_wave_added_position = wave_added_position[wave_added_index];

		std::uint64_t position_value = wave_data[one_wave_added_position];
		if(position_value == one_wave_added)
		{
			duplicate[wave_added_index] = 1;
			return;
		}
		if(one_wave_added_position+1 < wave_data_len)
		{
			std::uint64_t next_position_value = wave_data[one_wave_added_position+1];
			if(next_position_value == one_wave_added)
			{
				duplicate[wave_added_index] = 1;
				return;
			}
		}
		duplicate[wave_added_index] = 0;
	}
}

__host__ void detectDuplicatesWithSorting
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	std::uint64_t wave_added_size,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	pmpp::cuda_ptr<uint[]>& duplicate
)
{
	#if MEASURE_TIME
	using best_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	auto t1 = best_clock::now();
	#endif //MEASURE_TIME

	cudaError_t allocError;

	pmpp::cuda_ptr<std::uint64_t[]> device_wavefunction_sorted;

	//Dirty hack
	device_wavefunction_sorted.reset(const_cast<std::uint64_t*>(device_wavefunction.data()));

	sort(device_wavefunction_sorted.get(),device_wavefunction.size());

	#if MEASURE_TIME
	auto t2 = best_clock::now();
	std::uint64_t milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"      Sort wave took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	t1 = best_clock::now();
	#endif //MEASURE_TIME

	pmpp::cuda_ptr<std::int64_t[]> wave_added_position = pmpp::make_cuda_array<std::int64_t>(wave_added_size,&allocError);
	cudaMemset(wave_added_position.get(),-1,wave_added_size*sizeof(std::int64_t));

	findNearestValuesInSortedArray
	(
		device_wavefunction_sorted.get(),
		device_wavefunction.size(),
		wave_added.get(),
		wave_added_size,
		wave_added_position.get()
	);
	cudaDeviceSynchronize();


	#if MEASURE_TIME
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"      Find approx nearest took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	t1 = best_clock::now();
	#endif //MEASURE_TIME

	duplicate = pmpp::make_cuda_array<uint>(wave_added_size,&allocError);
	uint num_threads = 1024;
	dim3 blockSz = { num_threads };
	dim3 gridSz = { (static_cast<uint>(device_wavefunction.size())/num_threads)+1 };
	duplicateDetectionWithSorting_kernel<<<gridSz,blockSz>>>
	(
		wave_added.get(),
		wave_added_size,
		wave_added_position.get(),
		device_wavefunction_sorted.get(),
		device_wavefunction.size(),
		duplicate.get()
	);
	cudaDeviceSynchronize();

	#if MEASURE_TIME
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"      Duplicate set took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;
	#endif //MEASURE_TIME

	device_wavefunction_sorted.release();
}

__global__ void duplicateToOffset_kernel
(
	std::uint64_t wave_added_len,
	const uint* duplicate,
	bool* isDuplicate,
	waveSizeCountType* nonduplicateOffset
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

__host__ void duplicatesToOffset
(
	std::uint64_t maxOffset,
	pmpp::cuda_ptr<uint[]>& duplicate,
	pmpp::cuda_ptr<bool[]>& isDuplicate,
	pmpp::cuda_ptr<waveSizeCountType[]>& nonduplicateOffset
)
{
	cudaError_t allocError;
	isDuplicate = pmpp::make_cuda_array<bool>(maxOffset,&allocError);
	nonduplicateOffset = pmpp::make_cuda_array<waveSizeCountType>(maxOffset,&allocError);

	uint num_threads = 1024;
	dim3 blockSz = { num_threads };
	dim3 gridSz = { (static_cast<uint>(maxOffset)/num_threads)+1 };
	duplicateToOffset_kernel<<<gridSz,blockSz>>>(maxOffset,duplicate.get(),isDuplicate.get(),nonduplicateOffset.get());	cudaDeviceSynchronize();
}

__host__ void duplicateFinalizeOffset
(
	pmpp::cuda_ptr<waveSizeCountType[]>& nonduplicateOffset,
	waveSizeCountType maxOffset,
	waveSizeCountType& reducedMaxOffset
)
{
	inclusive_scan<waveSizeCountType,waveSizeCountType>(nonduplicateOffset.get(),maxOffset);
	cudaMemcpy
	(
		&reducedMaxOffset,
		nonduplicateOffset.get()+maxOffset-1,
		sizeof(waveSizeCountType),
		cudaMemcpyDeviceToHost
	);
}

__global__ void duplicateRemoval_kernel
(
	const std::uint64_t* wave_added,
	std::uint64_t wave_added_len,
	const bool* isDuplicate,
	const waveSizeCountType* nonduplicateOffset,
 	std::uint64_t* reduced_wave_added
)
{
	waveSizeCountType wave_data_index = blockDim.x*blockIdx.x + threadIdx.x;
	if(wave_data_index < wave_added_len)
	{
		std::uint64_t wave = wave_added[wave_data_index];
		bool nonDuplicate = !isDuplicate[wave_data_index];
		waveSizeCountType wave_offset = nonduplicateOffset[wave_data_index];

		if(nonDuplicate)
		{
			reduced_wave_added[wave_offset-1] = wave;
		}
	}
}

__host__ void removeDuplicates
(
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	waveSizeCountType maxOffset,
	pmpp::cuda_ptr<bool[]>& isDuplicate,
	pmpp::cuda_ptr<waveSizeCountType[]>& nonduplicateOffset,
	waveSizeCountType reducedMaxOffset
)
{
	if(reducedMaxOffset>0)
	{
		cudaError_t allocError;
		pmpp::cuda_ptr<std::uint64_t[]> reduced_wave_added;
		reduced_wave_added = pmpp::make_cuda_array<std::uint64_t>(reducedMaxOffset,&allocError);


		uint num_threads = 1024;
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
		wave_added.reset(nullptr);
}

__host__ void treatDuplicates
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	waveSizeCountType maxOffset,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	waveSizeCountType& reducedMaxOffset
)
{
	#if MEASURE_TIME
	using best_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	/*
	* Detect duplicates
	*/
	auto t1 = best_clock::now();
	#endif //MEASURE_TIME

	pmpp::cuda_ptr<uint[]> duplicate;
	//detectDuplicatesWithTable(device_wavefunction,maxOffset,wave_added,duplicate);
	//detectDuplicates(device_wavefunction,maxOffset,wave_added,duplicate);
	detectDuplicatesWithSorting(device_wavefunction,maxOffset,wave_added,duplicate);

	#if MEASURE_TIME
	auto t2 = best_clock::now();
	std::uint64_t milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"    Detect duplicates took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	/*
	* Duplicates to offset
	*/
	t1 = best_clock::now();
	#endif //MEASURE_TIME

	pmpp::cuda_ptr<bool[]> isDuplicate;
	pmpp::cuda_ptr<waveSizeCountType[]> nonduplicateOffset;
	duplicatesToOffset(maxOffset,duplicate,isDuplicate,nonduplicateOffset);
	duplicate.reset(nullptr);

	#if MEASURE_TIME
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"    Duplicates to offset took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	/*
	* Duplicates to offset
	*/
	t1 = best_clock::now();
	#endif //MEASURE_TIME

	duplicateFinalizeOffset(nonduplicateOffset,maxOffset,reducedMaxOffset);

	#if MEASURE_TIME
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"    Finalize duplicate offset took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	/*
	* Remove duplicates
	*/
	t1 = best_clock::now();
	#endif //MEASURE_TIME

	removeDuplicates(wave_added,maxOffset,isDuplicate,nonduplicateOffset,reducedMaxOffset);

	#if MEASURE_TIME
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"    Remove duplicates took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;
	#endif //MEASURE_TIME
}

__host__ cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{
	#if MEASURE_TIME
	using best_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	/*
	 * Compute collision data
	 */
	auto t1 = best_clock::now();
	#endif //MEASURE_TIME

	cudaError_t allocError;
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> waveOut;

	pmpp::cuda_ptr<bool[]> collisions;
	pmpp::cuda_ptr<waveSizeCountType[]> non_collision_offset;
	collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);

	#if MEASURE_TIME
	auto t2 = best_clock::now();
	std::uint64_t milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"  Compute collision took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;

	/*
	 * Compute offsets
	 */
	t1 = best_clock::now();
	#endif //MEASURE_TIME

	waveSizeCountType maxOffset;
	computeOffsets(device_wavefunction,non_collision_offset,maxOffset);

	#if MEASURE_TIME
	t2 = best_clock::now();
	milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout<<"  Compute offsets took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;
	#endif //MEASURE_TIME

	if(maxOffset>0)
	{
		/*
		* Compute evolution
		*/
		#if MEASURE_TIME
		t1 = best_clock::now();
		#endif //MEASURE_TIME
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
		collisions.reset(nullptr);
		non_collision_offset.reset(nullptr);

		#if MEASURE_TIME
		t2 = best_clock::now();
		milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout<<"  Compute evolution took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;
		#endif //MEASURE_TIME

		/*
		* Treat duplicates
		*/
		waveSizeCountType reducedMaxOffset;
		treatDuplicates(device_wavefunction,maxOffset,wave_added,reducedMaxOffset);

		/*
		* Compose
		*/
		#if MEASURE_TIME
		t1 = best_clock::now();
		#endif //MEASURE_TIME
		waveOut.second = device_wavefunction.size()+reducedMaxOffset;
		waveOut.first = pmpp::make_cuda_array<std::uint64_t>
		(
			device_wavefunction.size()+reducedMaxOffset,

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
		#if MEASURE_TIME
		t2 = best_clock::now();
		milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout<<"  Compose took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;
		#endif //MEASURE_TIME
	}
	else
	{
		#if MEASURE_TIME
		t1 = best_clock::now();
		#endif //MEASURE_TIME
		waveOut.second = device_wavefunction.size();
		waveOut.first = pmpp::make_cuda_array<std::uint64_t>
		(
			device_wavefunction.size(),
			&allocError
		);
		cudaMemcpy
		(
			waveOut.first.get(),
			device_wavefunction.data(),
			device_wavefunction.size()*sizeof(std::uint64_t),
			cudaMemcpyDeviceToDevice
		);

		#if MEASURE_TIME
		t2 = best_clock::now();
		milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout<<"  Compose took:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl;
		#endif //MEASURE_TIME
	}

	return waveOut;
}

__host__ cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations
)
{
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result;
	for(std::uint64_t operatorInd=0; operatorInd<activations.size(); operatorInd++)
	{
		// uncomment for debugging
		// std::cout<<"operatorInd:"<<operatorInd<<" waveSize:"<<result.second<<std::endl;
		result = evolve_operator(device_wavefunction,activations[operatorInd],deactivations[operatorInd]);
		device_wavefunction = cuda::std::span<std::uint64_t const>(result.first.get(),result.second);
	}
	return result;
}
