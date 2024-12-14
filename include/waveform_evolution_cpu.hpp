// /// @brief Evolve a wavefunction using a single operator
// /// @param device_wavefunction Current wavefunction (stored in device or managed memory)
// /// @param activation Activation part of the operator
// /// @param deactivation Deactivation part of the operator
// /// @return The resulting wavefunction (stored in device or managed memory) and the size of the resulting wavefunction
// cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator_cpu(
// 	cuda::std::span<std::uint64_t const> device_wavefunction,
// 	std::uint64_t activation, std::uint64_t deactivation
// );