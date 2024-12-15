#include <waveform_evolution.hpp>
#include <waveform_evolution_cpu.hpp>

#include "test_data_loader.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <bit>
#include <unordered_set>
#include <span>
#include <vector>

std::vector<std::uint64_t> evolve_operator_host(
	std::span<std::uint64_t const> host_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{
	using std::size;
	using std::data;

	auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_wavefunction));
	auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), size(host_wavefunction));
	std::copy_n(data(host_wavefunction), size(host_wavefunction), device_wavefunction.data());

	auto [result_wavefunction, result_size] = evolve_operator(device_wavefunction, activation, deactivation);

	std::vector<std::uint64_t> result(result_size);
	if(result_size)
		cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);
	return result;
}

// evolve_operator_host() with minimal modifications
std::vector<std::uint64_t> evolve_operator_host_cpu(
	std::span<std::uint64_t const> host_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{
	using std::size;
	using std::data;

	auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_wavefunction));
	auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), size(host_wavefunction));
	std::copy_n(data(host_wavefunction), size(host_wavefunction), device_wavefunction.data());

	auto [result_wavefunction, result_size] = evolve_operator_cpu(device_wavefunction, activation, deactivation);

	std::vector<std::uint64_t> result(result_size);
	if(result_size)
		cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);
	return result;
}

std::vector<std::uint64_t> evolve_ansatz_host(
	std::span<std::uint64_t const> host_wavefunction,
	std::span<std::uint64_t const> host_activations,
	std::span<std::uint64_t const> host_deactivations
)
{
	using std::size;
	using std::data;

	auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_wavefunction));
	auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), size(host_wavefunction));
	std::copy_n(data(host_wavefunction), size(host_wavefunction), device_wavefunction.data());

	auto device_activations_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_activations));
	auto device_activations = cuda::std::span(device_activations_ptr.get(), size(host_activations));
	std::copy_n(data(host_activations), size(host_activations), device_activations.data());

	auto device_deactivations_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_deactivations));
	auto device_deactivations = cuda::std::span(device_deactivations_ptr.get(), size(host_deactivations));
	std::copy_n(data(host_deactivations), size(host_deactivations), device_deactivations.data());

	auto [result_wavefunction, result_size] = evolve_ansatz(device_wavefunction, device_activations, device_deactivations);

	std::vector<std::uint64_t> result(result_size);
	if(result_size)
		cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);
	return result;
}


#include <iostream>
TEST_CASE("check_collision_kernel test", "[self-test]")
{
	//Input
	std::vector<std::vector<std::uint64_t>> multi_host_wavefunction =
	{
		{0xC,0x5,0x6,0xA,0x3},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330,0x07,0x0F,0x87,0x8F,0x1397498F}
	};
	std::vector<std::uint64_t> multi_activation = {0x2,0x60,0x60,0x70};
	std::vector<std::uint64_t> multi_deactivation = {0x4,0x6,0x3,0x7};

	//Expected output
	std::vector<std::vector<std::uint8_t>> multi_target_collisions =
	{
		{0,0,1,1,1},
		{0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1},
		{0,1,1,1,1,1,1,1,1},
		{0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0}
	};
	std::vector<std::vector<std::uint64_t>> multi_target_non_collision_offset =
	{
		{1,1,0,0,0},
		{1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0},
		{1,0,0,0,0,0,0,0,0},
		{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1}
	};

	for(uint i=0; i<multi_host_wavefunction.size(); i++)
	{
		const std::vector<std::uint64_t>& host_wavefunction = multi_host_wavefunction[i];
		uint waveSize = host_wavefunction.size();
		std::uint64_t activation = multi_activation[i];
		std::uint64_t deactivation = multi_deactivation[i];

		auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(waveSize);
		auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), waveSize);
		std::copy_n(data(host_wavefunction), waveSize, device_wavefunction.data());

		pmpp::cuda_ptr<bool[]> collisions;
		pmpp::cuda_ptr<waveSizeCountType[]> non_collision_offset;

		collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);

		std::vector<std::uint8_t> collisions_cpu(waveSize);
		std::vector<waveSizeCountType> non_collision_offset_cpu(waveSize);
		cudaMemcpy(data(collisions_cpu),collisions.get(),waveSize*sizeof(bool),cudaMemcpyDeviceToHost);
		cudaMemcpy(data(non_collision_offset_cpu),non_collision_offset.get(),waveSize*sizeof(waveSizeCountType),cudaMemcpyDeviceToHost);

		/*
		std::cout<<"collisions_cpu ("<<waveSize<<"):";
		for(uint w=0; w<waveSize; w++)
			std::cout<<"  "<<(uint)collisions_cpu[w];
		std::cout<<std::endl;

		std::cout<<"non_collision_offset_cpu ("<<waveSize<<"):";
		for(uint w=0; w<waveSize; w++)
			std::cout<<"  "<<non_collision_offset_cpu[w];
		std::cout<<std::endl;
		*/


		REQUIRE(waveSize == multi_target_collisions[i].size());
		REQUIRE(waveSize == multi_target_non_collision_offset[i].size());
		for(uint w=0; w<waveSize; w++)
		{
			REQUIRE(static_cast<bool>(collisions_cpu[w]) == static_cast<bool>(multi_target_collisions[i][w]));
			REQUIRE(non_collision_offset_cpu[w] == multi_target_non_collision_offset[i][w]);
		}
	}
	std::cout<<"------------------------------------------------------"<<std::endl;;
}

TEST_CASE("computeOffsets test", "[self-test]")
{
	//Input
	std::vector<std::vector<std::uint64_t>> multi_host_wavefunction =
	{
		{0xC,0x5,0x6,0xA,0x3},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330,0x07,0x0F,0x87,0x8F,0x1397498F}
	};
	std::vector<std::uint64_t> multi_activation = {0x2,0x60,0x60,0x70};
	std::vector<std::uint64_t> multi_deactivation = {0x4,0x6,0x3,0x7};

	//Expected output
	std::vector<std::vector<std::uint64_t>> multi_target_non_collision_offset =
	{
		{1,2,2,2,2},
		{1,1,1,1,1,1,1,2,3,3,3,3,3,3,3,3,3,3,3,3,4,5,5,5,5,5,5},
		{1,1,1,1,1,1,1,1,1},
		{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,4,5,6}
	};
	std::vector<std::uint64_t> multi_maxOffset = {2};

	for(uint i=0; i<multi_host_wavefunction.size(); i++)
	{
		const std::vector<std::uint64_t>& host_wavefunction = multi_host_wavefunction[i];
		uint waveSize = host_wavefunction.size();
		std::uint64_t activation = multi_activation[i];
		std::uint64_t deactivation = multi_deactivation[i];

		auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(waveSize);
		auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), waveSize);
		std::copy_n(data(host_wavefunction), waveSize, device_wavefunction.data());

		pmpp::cuda_ptr<bool[]> collisions;
		pmpp::cuda_ptr<waveSizeCountType[]> non_collision_offset;

		collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);

		waveSizeCountType maxOffset;
		computeOffsets(device_wavefunction,non_collision_offset,maxOffset);

		std::vector<waveSizeCountType> non_collision_offset_cpu(waveSize);
		cudaMemcpy(data(non_collision_offset_cpu),non_collision_offset.get(),waveSize*sizeof(waveSizeCountType),cudaMemcpyDeviceToHost);

		/*
		std::cout<<"non_collision_offset_cpu:";
		for(uint w=0; w<waveSize; w++)
			std::cout<<"  "<<non_collision_offset_cpu[w];
		std::cout<<std::endl;
		std::cout<<"maxOffset:"<<maxOffset<<std::endl;
		*/

		REQUIRE(waveSize == multi_target_non_collision_offset[i].size());
		for(uint w=0; w<waveSize; w++)
		{
			REQUIRE(non_collision_offset_cpu[w] == multi_target_non_collision_offset[i][w]);
		}
	}
}

#include <numeric>
TEST_CASE("evolve_kernel test", "[self-test]")
{
	//Input
	std::vector<std::vector<std::uint64_t>> multi_host_wavefunction =
	{
		{0xC},
		{0xC,0x5,0x6,0xA,0x3},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330,0x07,0x0F,0x87,0x8F,0x1397498F}
	};
	std::vector<std::uint64_t> multi_activation = {0x2,0x2,0x60,0x60,0x70};
	std::vector<std::uint64_t> multi_deactivation = {0x4,0x4,0x6,0x3,0x7};

	//Expected output
	std::vector<std::vector<std::uint64_t>> multi_target_wave_added =
	{
		{0xA},
		{0xa,0x3},
		{0x69,0xf0,0x270,0x1e0,0x360},
		{0x6c},
		{0x78,0x70,0x78,0xf0,0xf8,0x139749f8}
	};

	for(uint i=0; i<multi_host_wavefunction.size(); i++)
	{
		// timing measurement events: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// getting and returning the data is also measured since it needed for any application to work
		cudaEventRecord(start);

		const std::vector<std::uint64_t>& host_wavefunction = multi_host_wavefunction[i];
		uint waveSize = host_wavefunction.size();
		std::uint64_t activation = multi_activation[i];
		std::uint64_t deactivation = multi_deactivation[i];

		auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(waveSize);
		auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), waveSize);
		std::copy_n(data(host_wavefunction), waveSize, device_wavefunction.data());

		pmpp::cuda_ptr<bool[]> collisions;
		pmpp::cuda_ptr<waveSizeCountType[]> non_collision_offset;

		collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);

		waveSizeCountType maxOffset;
		computeOffsets(device_wavefunction,non_collision_offset,maxOffset);

		pmpp::cuda_ptr<std::uint64_t[]> wave_added;
		evolutionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset,maxOffset,wave_added);

		std::vector<std::uint64_t> wave_added_cpu(maxOffset);
		cudaMemcpy(data(wave_added_cpu),wave_added.get(),maxOffset*sizeof(std::uint64_t),cudaMemcpyDeviceToHost);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		/*
		std::cout<<"wave_added_cpu:";
		for(uint w=0; w<wave_added_cpu.size(); w++)
			std::cout<<"  "<<std::hex<<wave_added_cpu[w];
		std::cout<<std::endl;

		std::printf("Time testcase %d: %fms\n", i, milliseconds);
		*/

		REQUIRE(wave_added_cpu.size() == multi_target_wave_added[i].size());
		for(uint w=0; w<wave_added_cpu.size(); w++)
		{
			REQUIRE(wave_added_cpu[w] == multi_target_wave_added[i][w]);
		}
	}
}

TEST_CASE("removeDuplicates_kernel test", "[self-test]")
{
	//Input
	std::vector<std::vector<std::uint64_t>> multi_host_wavefunction =
	{
		{0xC},
		{0xC,0x5,0x6,0xA,0x3},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216}
	};
	std::vector<std::uint64_t> multi_activation = {0x2,0x2,0x60,0x60};
	std::vector<std::uint64_t> multi_deactivation = {0x4,0x4,0x6,0x3};

	//Expected output
	std::vector<std::vector<std::uint64_t>> multi_target_wave_added =
	{
		{0xA},
		{},
		{0x69,0x1e0,0x360},
		{0x6c}
	};

	for(uint i=0; i<multi_host_wavefunction.size(); i++)
	{
		const std::vector<std::uint64_t>& host_wavefunction = multi_host_wavefunction[i];
		uint waveSize = host_wavefunction.size();
		std::uint64_t activation = multi_activation[i];
		std::uint64_t deactivation = multi_deactivation[i];

		auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(waveSize);
		auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), waveSize);
		std::copy_n(data(host_wavefunction), waveSize, device_wavefunction.data());

		pmpp::cuda_ptr<bool[]> collisions;
		pmpp::cuda_ptr<waveSizeCountType[]> non_collision_offset;

		collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);

		waveSizeCountType maxOffset;
		computeOffsets(device_wavefunction,non_collision_offset,maxOffset);

		pmpp::cuda_ptr<std::uint64_t[]> wave_added;
		evolutionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset,maxOffset,wave_added);

		waveSizeCountType reducedMaxOffset;
		treatDuplicates(device_wavefunction,maxOffset,wave_added,reducedMaxOffset);

		std::vector<std::uint64_t> wave_added_cpu(reducedMaxOffset);
		cudaMemcpy(data(wave_added_cpu),wave_added.get(),reducedMaxOffset*sizeof(std::uint64_t),cudaMemcpyDeviceToHost);

		/*
		std::cout<<"reducedMaxOffset:"<<std::dec<<reducedMaxOffset<<std::endl;
		std::cout<<"wave_added_cpu:";
		for(uint w=0; w<wave_added_cpu.size(); w++)
			std::cout<<"  "<<std::hex<<wave_added_cpu[w];
		std::cout<<std::endl;
		*/

		REQUIRE(wave_added_cpu.size() == multi_target_wave_added[i].size());
		for(uint w=0; w<wave_added_cpu.size(); w++)
		{
			REQUIRE(wave_added_cpu[w] == multi_target_wave_added[i][w]);
		}
	}
}

// "Test evolve operator" with minimal modifications
TEST_CASE("Test evolve operator_cpu", "[self-test]")
{
	using std::begin;
	using std::end;

	test_data_loader loader("example_evolution.bin");
	loader.for_each_step([&] (
		std::span<std::uint64_t const> wfn_in,
		std::span<std::uint64_t const> wfn_out,
		std::uint64_t activation,
		std::uint64_t deactivation
	) {
		auto wfn_out_dut = evolve_operator_host_cpu(wfn_in, activation, deactivation);
		auto wfn_out_set = std::unordered_set(begin(wfn_out), end(wfn_out));
		auto wfn_out_dut_set = std::unordered_set(begin(wfn_out_dut), end(wfn_out_dut));

		/*
		printf("activation:            %ld\n", activation);
    	printf("deactivation:          %ld\n", deactivation);

		// output code generated by ChatGPT (with own small adjustments)
		std::unordered_set<std::uint64_t> wfn_in_set(wfn_in.begin(), wfn_in.end());
		// output for wfn_in (unsorted)
		std::vector<std::uint64_t> wfn_in_vec(wfn_in.begin(), wfn_in.end());
		//std::sort(wfn_in_vec.begin(), wfn_in_vec.end());
		std::printf("wfn_in:                ");
		for (const auto& value : wfn_in_vec) {
			std::printf("%lu ", value);
		}
		std::printf("\n");

		// output for wfn_out_set (sorted)
        std::vector<std::uint64_t> wfn_out_vec(wfn_out_set.begin(), wfn_out_set.end());
        std::sort(wfn_out_vec.begin(), wfn_out_vec.end());
        std::printf("wfn_out_set(sort):     ");
        for (const auto& value : wfn_out_vec) {
            std::printf("%lu ", value);
        }
        std::printf("\n");

        // output for wfn_out_dut_set (sorted)
		std::vector<std::uint64_t> wfn_out_dut_vec(wfn_out_dut_set.begin(), wfn_out_dut_set.end());
        std::sort(wfn_out_dut_vec.begin(), wfn_out_dut_vec.end());
        std::printf("wfn_out_dut_set(sort): ");
        for (const auto& value : wfn_out_dut_vec) {
            std::printf("%lu ", value);
        }
        std::printf("\n");

		// output new value for wfn_out_set
		std::vector<std::uint64_t> out_not_in;
		for (const auto& value : wfn_out_set) {
			if (wfn_in_set.find(value) == wfn_in_set.end()) {
				out_not_in.push_back(value);
			}
		}
		std::printf("new wfn_out_set:       ");
		for (const auto& value : out_not_in) {
			std::printf("%lu ", value);
		}
		std::printf("\n");
		// output new value for wfn_out_dut_set
		std::vector<std::uint64_t> out_not_in_dut;
		for (const auto& value : wfn_out_dut_set) {
			if (wfn_in_set.find(value) == wfn_in_set.end()) {
				out_not_in_dut.push_back(value);
			}
		}
		std::printf("new wfn_out_dut_set:   ");
		for (const auto& value : out_not_in_dut) {
			std::printf("%lu ", value);
		}
		std::printf("\n");
		std::printf("\n");
		// end of generated code
		*/

		REQUIRE(wfn_out_dut.size() == wfn_out_dut_set.size());
		REQUIRE(wfn_out_set == wfn_out_dut_set);
	});
}

TEST_CASE("Self test input data", "[self-test]")
{
	test_data_loader loader("example_evolution.bin");

	auto electrons = loader.electrons();
	auto orbitals = loader.single_electron_density_count();
	auto activations = loader.activations();
	auto deactivations = loader.deactivations();

	REQUIRE(activations.size() == loader.ansatz_size());
	REQUIRE(deactivations.size() == loader.ansatz_size());

	auto orbital_mask = (orbitals < 64 ? std::uint64_t(1) << orbitals : 0) - 1;

	for(std::size_t i = 0, n = loader.ansatz_size(); i < n; ++i)
	{
		auto n_activations = std::popcount(activations[i]);
		auto n_deactivations = std::popcount(deactivations[i]);

		REQUIRE((activations[i] & deactivations[i]) == 0);

		REQUIRE((activations[i] & ~orbital_mask) == 0);
		REQUIRE((deactivations[i] & ~orbital_mask) == 0);

		REQUIRE(n_activations > 0);
		REQUIRE(n_activations <= 2);
		REQUIRE(n_activations == n_deactivations);
	}

	std::size_t step = 0;
	loader.for_each_step([&] (
		std::span<std::uint64_t const> wfn_in,
		std::span<std::uint64_t const> wfn_out,
		std::uint64_t activation,
		std::uint64_t deactivation
	) {
		using std::begin;
		using std::end;

		REQUIRE(activation == activations[step]);
		REQUIRE(deactivation == deactivations[step]);

		auto wfn_in_set = std::unordered_set(begin(wfn_in), end(wfn_in));
		auto wfn_out_set = std::unordered_set(begin(wfn_out), end(wfn_out));
		REQUIRE(wfn_in_set.size() == wfn_in.size());
		REQUIRE(wfn_out_set.size() == wfn_out.size());

		REQUIRE(wfn_in.size() <= wfn_out.size());
		for(auto v : wfn_in)
			wfn_out_set.erase(v);
		REQUIRE(wfn_out_set.size() == wfn_out.size() - wfn_in.size());

		if(step == 0)
		{
			REQUIRE(std::all_of(begin(wfn_in), end(wfn_in), [&] (std::uint64_t v) { return (v & ~orbital_mask) == 0; }));
			REQUIRE(std::all_of(begin(wfn_in), end(wfn_in), [&] (std::uint64_t v) { return std::popcount(v) == electrons; }));
		}
		REQUIRE(std::all_of(begin(wfn_out), end(wfn_out), [&] (std::uint64_t v) { return (v & ~orbital_mask) == 0; }));
		REQUIRE(std::all_of(begin(wfn_out), end(wfn_out), [&] (std::uint64_t v) { return std::popcount(v) == electrons; }));

		++step;
	});

	REQUIRE(step == loader.ansatz_size());
}

TEST_CASE("Test evolve operator", "[simple]")
{
	using std::begin;
	using std::end;

	test_data_loader loader("example_evolution.bin");
	loader.for_each_step([&] (
		std::span<std::uint64_t const> wfn_in,
		std::span<std::uint64_t const> wfn_out,
		std::uint64_t activation,
		std::uint64_t deactivation
	) {
		auto wfn_out_dut = evolve_operator_host(wfn_in, activation, deactivation);
		auto wfn_out_set = std::unordered_set(begin(wfn_out), end(wfn_out));
		auto wfn_out_dut_set = std::unordered_set(begin(wfn_out_dut), end(wfn_out_dut));

		REQUIRE(wfn_out_dut.size() == wfn_out_dut_set.size());
		REQUIRE(wfn_out_set == wfn_out_dut_set);
	});
}

TEST_CASE("Test evolve ansatz", "[simple]")
{
	using std::begin;
	using std::end;

	test_data_loader loader("example_evolution.bin");
	auto [wfn_in, wfn_out] = loader.first_and_last_wavefunction();
	auto wfn_out_dut = evolve_ansatz_host(wfn_in, loader.activations(), loader.deactivations());
	auto wfn_out_set = std::unordered_set(begin(wfn_out), end(wfn_out));
	auto wfn_out_dut_set = std::unordered_set(begin(wfn_out_dut), end(wfn_out_dut));
	REQUIRE(wfn_out_dut.size() == wfn_out_dut_set.size());
	REQUIRE(wfn_out_set == wfn_out_dut_set);
}

#include <map>
#include <fstream>
TEST_CASE("example_evolution timing", "[simple]")
{
	using std::begin;
	using std::end;

	std::map<std::uint64_t,std::vector<std::uint64_t>> inSize_time;
	std::map<std::uint64_t,std::vector<std::uint64_t>> outSize_time;
	using best_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	test_data_loader loader("example_evolution.bin");
	loader.for_each_step([&] (
		std::span<std::uint64_t const> wfn_in,
		std::span<std::uint64_t const> wfn_out,
		std::uint64_t activation,
		std::uint64_t deactivation
	) {

		auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(wfn_in));
		auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), size(wfn_in));
		std::copy_n(data(wfn_in), size(wfn_in), device_wavefunction.data());

		auto t_start = best_clock::now();
		auto [result_wavefunction, result_size] = evolve_operator(device_wavefunction, activation, deactivation);
		auto t_end = best_clock::now();
		std::uint64_t seconds_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
		std::cout<<"inSize:"<<wfn_in.size()<<" outSize:"<<wfn_out.size()<<" ="<<seconds_elapsed<<std::endl;

		std::vector<std::uint64_t> result(result_size);
		if(result_size)
			cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);

		inSize_time[wfn_in.size()].push_back(seconds_elapsed);
		outSize_time[wfn_out.size()].push_back(seconds_elapsed);
	});

	std::ofstream inSizeTimes("example_evolution_inSizeTimes");
	for(auto iterIn=inSize_time.begin(); iterIn!=inSize_time.end(); iterIn++)
	{
		std::uint64_t key = iterIn->first;
		std::vector<std::uint64_t> values = iterIn->second;
		std::uint64_t avg = std::accumulate(values.begin(),values.end(),0);
		avg /= values.size();
		inSizeTimes<<key<<" = "<<avg<<std::endl;
	}

	std::ofstream outSizeTimes("example_evolution_outSizeTimes");
	for(auto iterIn=outSize_time.begin(); iterIn!=outSize_time.end(); iterIn++)
	{
		std::uint64_t key = iterIn->first;
		std::vector<std::uint64_t> values = iterIn->second;
		std::uint64_t avg = std::accumulate(values.begin(),values.end(),0);
		avg /= values.size();
		outSizeTimes<<key<<" = "<<avg<<std::endl;
	}
}

#include <algorithm>
#include <chrono>
#include <random>
TEST_CASE("artificial data timing", "[simple]")
{
	using std::begin;
	using std::end;

	std::map<std::uint64_t,std::vector<std::uint64_t>> inSize_time;
	std::map<std::uint64_t,std::vector<std::uint64_t>> outSize_time;
	using best_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	std::uint64_t initalWaveSize = 1;
	std::uint64_t endWaveSize = 2e9;
	std::uint64_t perSizeIteration = 5;

	std::array<std::uint64_t,63> bitNumbers;
	for(uint i=0; i<63; i++)
		bitNumbers[i] = i;

	std::ofstream inSizeTimes("artificialData_inSizeTimes");

	for(std::uint64_t waveSize = initalWaveSize; waveSize<=endWaveSize; waveSize*=10)
	{
		std::random_device rnd_device;
		std::mt19937 engine {rnd_device()};
		std::uniform_int_distribution<std::uint64_t> dist {0, std::numeric_limits<std::uint64_t>::max()};
		auto gen = [&](){return dist(engine);};
		std::vector<std::uint64_t> wfn(waveSize);
		std::generate(wfn.begin(),wfn.end(),gen);
		std::span<std::uint64_t> wfn_gen(wfn.begin(),waveSize);

		std::vector<std::uint64_t> seconds;
		for(uint iteration=0; iteration<perSizeIteration; iteration++)
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::shuffle(bitNumbers.begin(),bitNumbers.end(),std::default_random_engine(seed));
			std::uint64_t activation = 0;
			std::uint64_t one = 1;
			activation = activation | one<<bitNumbers[0];
			activation = activation | one<<bitNumbers[1];
			std::uint64_t deactivation = 0;
			deactivation = deactivation | one<<bitNumbers[2];
			deactivation = deactivation | one<<bitNumbers[3];

			std::cout<<"waveSize:"<<std::dec<<waveSize<<" wfn_gen:"<<std::dec<<wfn_gen.size()<<" activation:"<<std::hex<<activation<<" deactivation:"<<std::hex<<deactivation<<std::endl;

			auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(wfn_gen));
			auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), size(wfn_gen));
			std::copy_n(data(wfn_gen), size(wfn_gen), device_wavefunction.data());

			auto t_start = best_clock::now();
			std::uint64_t result_size;
			auto result = evolve_operator(device_wavefunction, activation, deactivation);
			result_size = result.second;
			auto t_end = best_clock::now();

			std::uint64_t milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
			std::cout<<waveSize<<" -> "<<result_size<<"  time:"<<std::dec<<milliseconds_elapsed<<" milliseconds"<<std::endl<<std::endl;
			seconds.push_back(milliseconds_elapsed);
		}
		std::uint64_t avg = std::accumulate(seconds.begin(),seconds.end(),0);
		avg /= seconds.size();

		std::cout<<"-----------------------avg:"<<std::dec<<avg<<" milliseconds"<<std::endl<<std::endl;
		inSizeTimes<<waveSize<<" = "<<avg<<std::endl;
	}
}
