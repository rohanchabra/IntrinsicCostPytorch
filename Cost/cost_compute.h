#include <torch/extension.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>


void Intrinsic_Cost(at::Tensor leftFeat, at::Tensor rightFeat, at::Tensor costFeat, int windowSize);
