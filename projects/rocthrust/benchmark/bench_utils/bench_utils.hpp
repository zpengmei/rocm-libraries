// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef ROCTHRUST_BENCHMARKS_BENCH_UTILS_BENCH_UTILS_HPP_
#define ROCTHRUST_BENCHMARKS_BENCH_UTILS_BENCH_UTILS_HPP_

// Utils
#include <thrust/execution_policy.h>

#include "common/types.hpp" // IWYU pragma: export
#include "custom_reporter.hpp" // IWYU pragma: export
#include "generation_utils.hpp" // IWYU pragma: export
#include "thrust_compat.hpp" // IWYU pragma: export

// HIP/CUDA
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#  include <hip/hip_runtime.h>
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <cuda_runtime.h>
#endif

// STL
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>

namespace bench_utils
{

#define HIP_CHECK(condition)                                                       \
    {                                                                              \
      hipError_t error = condition;                                                \
      if (error != hipSuccess)                                                     \
      {                                                                            \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error);                                                               \
      }                                                                            \
    }

// Binary operators
struct less_t
{
  template <typename T>
  __host__ __device__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }
};

struct max_t
{
  template <typename T>
  __host__ __device__ T operator()(const T& lhs, const T& rhs)
  {
    less_t less{};
    return less(lhs, rhs) ? rhs : lhs;
  }
};

auto StatisticsSum = [](const std::vector<double>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
};

double StatisticsMean(const std::vector<double>& v)
{
  if (v.empty())
  {
    return 0.0;
  }
  return StatisticsSum(v) * (1.0 / static_cast<double>(v.size()));
}

double StatisticsMedian(const std::vector<double>& v)
{
  if (v.size() < 3)
  {
    return StatisticsMean(v);
  }
  std::vector<double> copy(v);

  auto center = copy.begin() + v.size() / 2;
  std::nth_element(copy.begin(), center, copy.end());

  // Did we have an odd number of samples?  If yes, then center is the median.
  // If not, then we are looking for the average between center and the value
  // before.  Instead of resorting, we just look for the max value before it,
  // which is not necessarily the element immediately preceding `center` Since
  // `copy` is only partially sorted by `nth_element`.
  if (v.size() % 2 == 1)
  {
    return *center;
  }
  auto center2 = std::max_element(copy.begin(), center);
  return (*center + *center2) / 2.0;
}

// Return the sum of the squares of this sample set
auto SumSquares = [](const std::vector<double>& v) {
  return std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
};

auto Sqr = [](const double dat) {
  return dat * dat;
};
auto Sqrt = [](const double dat) {
  // Avoid NaN due to imprecision in the calculations
  if (dat < 0.0)
  {
    return 0.0;
  }
  return std::sqrt(dat);
};

double StatisticsStdDev(const std::vector<double>& v)
{
  const auto mean = StatisticsMean(v);
  if (v.empty())
  {
    return mean;
  }

  // Sample standard deviation is undefined for n = 1
  if (v.size() == 1)
  {
    return 0.0;
  }

  const double avg_squares = SumSquares(v) * (1.0 / static_cast<double>(v.size()));
  return Sqrt(static_cast<double>(v.size()) / (static_cast<double>(v.size()) - 1.0) * (avg_squares - Sqr(mean)));
}

double StatisticsCV(const std::vector<double>& v)
{
  if (v.size() < 2)
  {
    return 0.0;
  }

  const auto stddev = StatisticsStdDev(v);
  const auto mean   = StatisticsMean(v);

  if (std::fpclassify(mean) == FP_ZERO)
  {
    return 0.0;
  }

  return stddev / mean;
}

struct caching_allocator_t
{
  using value_type = char;

  caching_allocator_t() = default;
  ~caching_allocator_t()
  {
    free_all();
  }

  char* allocate(std::ptrdiff_t num_bytes)
  {
    value_type* result{};
    auto free_block = free_blocks.find(num_bytes);
    if (free_block != free_blocks.end())
    {
      result = free_block->second;
      free_blocks.erase(free_block);
    }
    else
    {
      HIP_CHECK(hipMalloc(&result, num_bytes));
    }

    allocated_blocks.emplace(result, num_bytes);
    return result;
  }

  void deallocate(value_type* ptr, size_t)
  {
    auto iter = allocated_blocks.find(ptr);
    if (iter == allocated_blocks.end())
    {
      throw std::runtime_error("Memory was not allocated by this allocator");
    }

    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);
    free_blocks.emplace(num_bytes, ptr);
  }

private:
  using FreeBlocksType      = std::multimap<std::ptrdiff_t, value_type*>;
  using AllocatedBlocksType = std::map<value_type*, std::ptrdiff_t>;

  FreeBlocksType free_blocks;
  AllocatedBlocksType allocated_blocks;

  void free_all()
  {
    for (auto free_block : free_blocks)
    {
      HIP_CHECK(hipFree(free_block.second));
    }

    for (auto allocated_block : allocated_blocks)
    {
      HIP_CHECK(hipFree(allocated_block.first));
    }
  }
};

inline std::string format_pow2(size_t n)
{
  unsigned int k = 0;
  while (!(n & 1))
  {
    k++;
    n >>= 1;
  }
  return "1 << " + std::to_string(k);
}

struct sys_info
{
  hipDeviceProp_t devProp;
  sys_info()
  {
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
  }
};

inline sys_info system;
inline constexpr size_t sizes[] = {1u << 16, 1u << 20, 1u << 24, 1u << 28};

} // namespace bench_utils

#endif // ROCTHRUST_BENCHMARKS_BENCH_UTILS_BENCH_UTILS_HPP_
