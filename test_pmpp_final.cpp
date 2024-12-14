#include <waveform_evolution.hpp>

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
		pmpp::cuda_ptr<std::uint64_t[]> non_collision_offset;

		collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);

		std::vector<std::uint8_t> collisions_cpu(waveSize);
		std::vector<std::uint64_t> non_collision_offset_cpu(waveSize);
		cudaMemcpy(data(collisions_cpu),collisions.get(),waveSize*sizeof(bool),cudaMemcpyDeviceToHost);
		cudaMemcpy(data(non_collision_offset_cpu),non_collision_offset.get(),waveSize*sizeof(std::uint64_t),cudaMemcpyDeviceToHost);

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
		pmpp::cuda_ptr<std::uint64_t[]> non_collision_offset;

		collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);

		std::uint64_t maxOffset;
		computeOffsets(device_wavefunction,non_collision_offset,maxOffset);

		std::vector<std::uint64_t> non_collision_offset_cpu(waveSize);
		cudaMemcpy(data(non_collision_offset_cpu),non_collision_offset.get(),waveSize*sizeof(std::uint64_t),cudaMemcpyDeviceToHost);

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
		{0xC,0x5,0x6,0xA,0x3},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330,0x07,0x0F,0x87,0x8F,0x1397498F}
	};
	std::vector<std::uint64_t> multi_activation = {0x2,0x60,0x60,0x70};
	std::vector<std::uint64_t> multi_deactivation = {0x4,0x6,0x3,0x7};

	//Expected output
	std::vector<std::vector<std::uint64_t>> multi_target_wave_added =
	{
		{0xa,0x3},
		{0x69,0xf0,0x270,0x1e0,0x360},
		{0x6c},
		{0x78,0x70,0x78,0xf0,0xf8,0x139749f8}
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
		pmpp::cuda_ptr<std::uint64_t[]> non_collision_offset;

		collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);

		std::uint64_t maxOffset;
		computeOffsets(device_wavefunction,non_collision_offset,maxOffset);

		pmpp::cuda_ptr<std::uint64_t[]> wave_added;
		evolutionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset,maxOffset,wave_added);

		std::vector<std::uint64_t> wave_added_cpu(maxOffset);
		cudaMemcpy(data(wave_added_cpu),wave_added.get(),maxOffset*sizeof(std::uint64_t),cudaMemcpyDeviceToHost);

		/*
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


TEST_CASE("removeDuplicates_kernel test", "[self-test]")
{
	//Input
	std::vector<std::vector<std::uint64_t>> multi_host_wavefunction =
	{
		{0xC,0x5,0x6,0xA,0x3},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216,0x6c,0xcc,0x24c,0x14a,0x66,0xc6,0x246,0x12c,0x18c,0x30c,0x126,0x186,0x306,0x39,0xf0,0x270,0x1b0,0x330},
		{0xf,0x3c,0x9c,0x21c,0x5a,0x11a,0x36,0x96,0x216}
	};
	std::vector<std::uint64_t> multi_activation = {0x2,0x60,0x60};
	std::vector<std::uint64_t> multi_deactivation = {0x4,0x6,0x3};

	//Expected output
	std::vector<std::vector<std::uint64_t>> multi_target_wave_added =
	{
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
		pmpp::cuda_ptr<std::uint64_t[]> non_collision_offset;

		collisionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset);

		std::uint64_t maxOffset;
		computeOffsets(device_wavefunction,non_collision_offset,maxOffset);

		pmpp::cuda_ptr<std::uint64_t[]> wave_added;
		evolutionEvaluation(device_wavefunction,activation,deactivation,collisions,non_collision_offset,maxOffset,wave_added);

		std::uint64_t reducedMaxOffset;
		removeDuplicates(device_wavefunction,maxOffset,wave_added,reducedMaxOffset);

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
