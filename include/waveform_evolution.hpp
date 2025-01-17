#pragma once

#include <helpers.hpp>
#include <cuda/std/utility>
#include <cuda/std/span>
#include <unordered_set>

typedef std::uint32_t waveSizeCountType;

/// @brief
/// @param device_wavefunction
/// @param activation
/// @param deactivation
/// @param collisions
/// @param non_collision_offset
/// @return
void collisionEvaluation
(
	cuda::std::span<std::uint64_t const> const & device_wavefunction,
	std::uint64_t activation,
	std::uint64_t deactivation,
	pmpp::cuda_ptr<bool[]>& collisions,
	pmpp::cuda_ptr<waveSizeCountType[]>& non_collision_offset
);

/// @brief
/// @param device_wavefunction
/// @param non_collision_offset
/// @param maxOffset
/// @return
void computeOffsets
(
	const cuda::std::span<std::uint64_t const>& device_wavefunction,
	pmpp::cuda_ptr<waveSizeCountType[]>& non_collision_offset,
	waveSizeCountType& maxOffset
);

/// @brief
/// @param device_wavefunction
/// @param activation
/// @param deactivation
/// @param collisions
/// @param non_collision_offset
/// @param maxOffset
/// @param waveOut
/// @return
void evolutionEvaluation
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	std::uint64_t activation,
	std::uint64_t deactivation,
	const pmpp::cuda_ptr<bool[]>& collisions,
	const pmpp::cuda_ptr<waveSizeCountType[]>& non_collision_offset,
	waveSizeCountType maxOffset,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added
);

void detectDuplicates
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	waveSizeCountType maxOffset,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	pmpp::cuda_ptr<uint[]>& duplicate
);

void sort
(
	std::uint64_t* sequence,
	std::uint64_t len
);

void findNearestValuesInSortedArray
(
	const std::uint64_t* sortedSequence,
	const std::uint64_t sortedSequenceLen,
	const std::uint64_t* values,
	const std::uint64_t valuesLen,
	std::int64_t* valuesPosition
);

void detectDuplicatesWithSorting
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	std::uint64_t wave_added_size,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	pmpp::cuda_ptr<uint[]>& duplicate
);

void duplicatesToOffset
(
	std::uint64_t maxOffset,
	pmpp::cuda_ptr<uint[]>& duplicate,
	pmpp::cuda_ptr<bool[]>& isDuplicate,
	pmpp::cuda_ptr<waveSizeCountType[]>& nonduplicateOffset
);

void duplicateFinalizeOffset
(
	pmpp::cuda_ptr<waveSizeCountType[]>& nonduplicateOffset,
	waveSizeCountType maxOffset,
	waveSizeCountType& reducedMaxOffset
);

void removeDuplicates
(
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	waveSizeCountType maxOffset,
	pmpp::cuda_ptr<bool[]>& isDuplicate,
	pmpp::cuda_ptr<waveSizeCountType[]>& nonduplicateOffset,
	waveSizeCountType reducedMaxOffset
);

/// @brief
/// @param device_wavefunction
/// @param maxOffset
/// @param wave_added
/// @param reducedMaxOffset
/// @return
void treatDuplicates
(
	const cuda::std::span<std::uint64_t const> & device_wavefunction,
	waveSizeCountType maxOffset,
	pmpp::cuda_ptr<std::uint64_t[]>& wave_added,
	waveSizeCountType& reducedMaxOffset
);

/// @brief Evolve a wavefunction using a single operator
/// @param device_wavefunction Current wavefunction (stored in device or managed memory)
/// @param activation Activation part of the operator
/// @param deactivation Deactivation part of the operator
/// @return The resulting wavefunction (stored in device or managed memory) and the size of the resulting wavefunction
cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
);

/// @brief Evolve a wavefunction using a full ansatz
/// @param device_wavefunction Initial wavefunction (stored in device or managed memory)
/// @param activation Activation parts of the operators (stored in device or managed memory)
/// @param deactivation Deactivation parts of the operators (stored in device or managed memory; same size as activation)
/// @return The final wavefunction (stored in device or managed memory) and the size of the final wavefunction
cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations
);
