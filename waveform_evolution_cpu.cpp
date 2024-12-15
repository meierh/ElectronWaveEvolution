#include <waveform_evolution_cpu.hpp>

// TODO: remove printf in this file
cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator_cpu(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{
    const std::uint64_t* wave_data = device_wavefunction.data();

    // to save the new waves
    // pmpp::cuda_ptr<std::uint64_t[]> wave_added = pmpp::make_managed_cuda_array<std::uint64_t>(maxOffset,cudaMemAttachGlobal,&allocError);
    uint64_t wave_added_size = 0;

    // get number of new waves
    for(uint64_t wave_data_index = 0; wave_data_index<device_wavefunction.size(); wave_data_index++)
    {
        std::uint64_t wave = wave_data[wave_data_index];
        // check for collision
        bool col = (bool)((wave & activation) | ((~wave) & deactivation));
        if(!col)
        {
            // apply activation and deactivation operator
            std::uint64_t new_wave = wave;
			new_wave |= activation;
			new_wave &= ~deactivation;

            // check whether the value is a duplicate
            bool isDuplicate = false;

            // duplicate_index works like wave_data_index
            for(uint64_t duplicate_index = 0; duplicate_index<device_wavefunction.size(); duplicate_index++)
            { 
                if(new_wave == wave_data[duplicate_index])
               {
                    isDuplicate = true;
                    break;
                }
            }
            if(isDuplicate == false)
            { 
                wave_added_size++;
            }
        }
    } 
    cudaError_t allocError;

    // to save the new waves
    pmpp::cuda_ptr<std::uint64_t[]> wave_added = pmpp::make_managed_cuda_array<std::uint64_t>(wave_added_size,cudaMemAttachGlobal,&allocError);

    // needed since dynamic sized data structures are difficult in c++
    uint64_t wave_added_index = 0;

    // almost the same loop again to save the values
    for(uint64_t wave_data_index = 0; wave_data_index<device_wavefunction.size(); wave_data_index++)
    {
        std::printf("Iteration: %d, ",wave_data_index);
        std::uint64_t wave = wave_data[wave_data_index];
        std::printf("Input: %d, ",wave);
        // check for collision
        bool col = (bool)((wave & activation) | ((~wave) & deactivation));
        std::printf("col: %d\n",col);
        if(!col)
        {
            // apply activation and deactivation operator
            std::uint64_t new_wave = wave;
			new_wave |= activation;
			new_wave &= ~deactivation;
            // check whether the value is a duplicate
            bool isDuplicate = false;

            // duplicate_index works like wave_data_index
            for(uint64_t duplicate_index = 0; duplicate_index<device_wavefunction.size(); duplicate_index++)
            { 
                if(new_wave == wave_data[duplicate_index])
               {
                    isDuplicate = true;
                    std::printf("duplicate\n");
                    break;
                }
            }
            if(isDuplicate == false)
            {
                std::printf("Added %d\n",new_wave);
                wave_added[wave_added_index] = new_wave;
                wave_added_index++;
                std::printf("wave_added: ");
                for(int i = 0; i<wave_added_size;i++)
                {
                    std::printf("%d ",wave_added[i]);
                }
                std::printf("\n");
            }
        }
    } 
    std::printf("wave_added: ");
    for(int i = 0; i<wave_added_size;i++)
    {
        std::printf("%d ",wave_added[i]);
    }
    std::printf("\n");
    // set return value
    cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> waveOut;
    waveOut.second = device_wavefunction.size()+wave_added_size;
    waveOut.first = pmpp::make_managed_cuda_array<std::uint64_t>
    (
        device_wavefunction.size()+wave_added_size,
        cudaMemAttachGlobal,
        &allocError
    );

    cudaMemcpy
    (
        waveOut.first.get(),
        device_wavefunction.data(),
        device_wavefunction.size()*sizeof(std::uint64_t),
        cudaMemcpyHostToHost
    );
    if(wave_added_size>0)
    {
        cudaMemcpy
        (
            waveOut.first.get()+device_wavefunction.size(),
            wave_added.get(),
            wave_added_size*sizeof(std::uint64_t),
            cudaMemcpyHostToHost
        );
    }
    return waveOut;
}

