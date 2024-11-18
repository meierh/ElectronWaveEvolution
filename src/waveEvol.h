#pragma once
#include "pointer.h"

#include <cuda_runtime.h>
#include <vector>

template<typename T>
void evolve(std::vector<T>& wave, T activation, T deactivation) requires std::unsigned_integral<T>;
