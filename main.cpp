#include <waveform_evolution.hpp>
#include <waveform_evolution_cpu.hpp>

#include "test_data_loader.hpp"

#include <algorithm>
#include <bit>
#include <unordered_set>
#include <span>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <fstream>
#include <map>
#include <algorithm>
#include <chrono>

int main()
{
    // TODO: Maybe add multiple iterations and averaging of time

    #define USE_CPU 0
    // values from 0 to 3
    int8_t selected_testcase = 1;
    std::string testcase[4] =
    {
        "electrons-10_orbitals-20.bin",
        "electrons-15_orbitals-30.bin",
        "electrons-20_orbitals-40.bin",
        "electrons-25_orbitals-50.bin"
    };

    std::cout<< "Using file: " << testcase[selected_testcase] << std::endl;

    using best_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    #if USE_CPU
    std::cout<<"time_cpu_" + testcase[selected_testcase]<<std::endl;
    #else
    std::cout<<"time_gpu_" + testcase[selected_testcase]<<std::endl;
    #endif
    // total time: whole program with loading data (excluded writing to csv and if possible stdout)
    // cuda time: creation of cuda arrays, data tarnsfer and kernel
    // kernel time: kernel only
    std::cout<<"total time in ms"<<","<<"cuda time in ms"<<","<<"kernel time in ms"<<std::endl;

    auto t_start_total = best_clock::now();

    test_data_loader loader(testcase[selected_testcase].c_str());
    auto host_wavefunction = loader.first_wavefunction();
    auto host_activations = loader.activations();
    auto host_deactivations = loader.deactivations();
    std::cout <<"activation size: " << host_activations.size() << std::endl;
    std::cout <<"deactivation size: " << host_deactivations.size() << std::endl;

    using best_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

    auto t_start_cuda = best_clock::now();

    auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_wavefunction));
    auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), size(host_wavefunction));
    std::copy_n(data(host_wavefunction), size(host_wavefunction), device_wavefunction.data());

    auto device_activations_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_activations));
    auto device_activations = cuda::std::span(device_activations_ptr.get(), size(host_activations));
    std::copy_n(data(host_activations), size(host_activations), device_activations.data());

    auto device_deactivations_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_deactivations));
    auto device_deactivations = cuda::std::span(device_deactivations_ptr.get(), size(host_deactivations));
    std::copy_n(data(host_deactivations), size(host_deactivations), device_deactivations.data());

    auto t_start_kernel = best_clock::now();
    #if USE_CPU
    auto [result_wavefunction, result_size] = evolve_ansatz_cpu(device_wavefunction, device_activations, device_deactivations);
    #else
    auto [result_wavefunction, result_size] = evolve_ansatz(device_wavefunction, device_activations, device_deactivations);
    #endif
    auto t_end_kernel = best_clock::now();

    auto t_end_cuda = best_clock::now();

    std::vector<std::uint64_t> result(result_size);
    if(result_size)
        cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);
    result;

    auto t_end_total = best_clock::now();

    std::cout<<"waveSizeInput:"<<host_wavefunction.size()<<" waveSizeOutput:"<<result.size()<< std::endl;

    std::uint64_t milliseconds_elapsed_total = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_total - t_start_total).count();
    std::uint64_t milliseconds_elapsed_cuda = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_cuda - t_start_cuda).count();
    std::uint64_t milliseconds_elapsed_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_kernel - t_start_kernel).count();
    std::cout<<milliseconds_elapsed_total<<","<<milliseconds_elapsed_cuda<<","<<milliseconds_elapsed_kernel<<std::endl;
}