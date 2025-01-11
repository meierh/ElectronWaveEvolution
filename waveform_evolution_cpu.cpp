#include <waveform_evolution_cpu.hpp>

#include <vector>
#include <unordered_set>

// TODO: remove printf in this file
cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator_cpu(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{
    std::vector<uint64_t> wave_data_vector_cpu(device_wavefunction.size());
    cudaMemcpy(wave_data_vector_cpu.data(), device_wavefunction.data(), sizeof(std::uint64_t) * device_wavefunction.size(), cudaMemcpyDeviceToHost);
    uint64_t size = device_wavefunction.size();
    // to save the new waves
    uint64_t wave_added_size = 0;
    // to check for dupicated values in constant time
    std::unordered_set<uint64_t> wave_data_set_cpu(wave_data_vector_cpu.begin(), wave_data_vector_cpu.end());

    // get number of new waves
    for(uint64_t wave_data_index = 0; wave_data_index<size; wave_data_index++)
    {
        std::uint64_t wave = wave_data_vector_cpu[wave_data_index];
        // check for collision
        bool col = (bool)((wave & activation) | ((~wave) & deactivation));
        if(!col)
        {
            // apply activation and deactivation operator
            std::uint64_t new_wave = wave;
			new_wave |= activation;
			new_wave &= ~deactivation;

            // check whether the value is a duplicate
            if(!(wave_data_set_cpu.contains(new_wave)))
            {
                // add the new wave if it is no dupicate 
                wave_data_vector_cpu.push_back(new_wave);
            }
        }
    }

    // set return value
    cudaError_t allocError;
    cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> waveOut;
    waveOut.second = wave_data_vector_cpu.size();
    waveOut.first = pmpp::make_managed_cuda_array<std::uint64_t>
    (
        wave_data_vector_cpu.size(),
        cudaMemAttachGlobal,
        &allocError
    );

    cudaMemcpy
    (
        waveOut.first.get(),
        wave_data_vector_cpu.data(),
        wave_data_vector_cpu.size()*sizeof(std::uint64_t),
        cudaMemcpyHostToDevice
    );
    return waveOut;
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz_cpu(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations
)
{
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result;
	for(std::uint64_t operatorInd=0; operatorInd<activations.size(); operatorInd++)
	{
		result = evolve_operator_cpu(device_wavefunction,activations[operatorInd],deactivations[operatorInd]);
		device_wavefunction = cuda::std::span<std::uint64_t const>(result.first.get(),result.second);
	}
	return result;
}
