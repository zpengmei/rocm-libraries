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

//// \brief Gets the peak global memory bus bandwidth in bytes/sec.
std::size_t get_global_memory_bus_bandwidth(int device_id)
{
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));

  // Get the peak clock rate of the global memory bus in Hz.
  const std::size_t global_memory_bus_peak_clock_rate = static_cast<std::size_t>(props.memoryClockRate) * 1000; /*kHz ->
                                                                                                                   Hz*/
  // Get width of the global memory bus in bits.
  const int get_global_memory_bus_width = props.memoryBusWidth;

  // Multiply by 2 because of DDR,
  // CHAR_BIT to convert bus_width to bytes.
  return 2 * global_memory_bus_peak_clock_rate * static_cast<std::size_t>(get_global_memory_bus_width / CHAR_BIT);
}

/// \brief Adds device info and properties to the Google benchmark info
inline void add_common_benchmark_info()
{
  hipDeviceProp_t devProp;
  int device_id = 0;
  HIP_CHECK(hipGetDevice(&device_id));
  HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

  auto str = [](const std::string& name, const std::string& val) {
    benchmark::AddCustomContext(name, val);
  };

  auto num = [](const std::string& name, const auto& value) {
    benchmark::AddCustomContext(name, std::to_string(value));
  };

  auto dim2 = [num](const std::string& name, const auto* values) {
    num(name + "_x", values[0]);
    num(name + "_y", values[1]);
  };

  auto dim3 = [num, dim2](const std::string& name, const auto* values) {
    dim2(name, values);
    num(name + "_z", values[2]);
  };

  str("hdp_name", devProp.name);
  num("hdp_total_global_mem", devProp.totalGlobalMem);
  num("hdp_shared_mem_per_block", devProp.sharedMemPerBlock);
  num("hdp_regs_per_block", devProp.regsPerBlock);
  num("hdp_warp_size", devProp.warpSize);
  num("hdp_max_threads_per_block", devProp.maxThreadsPerBlock);
  dim3("hdp_max_threads_dim", devProp.maxThreadsDim);
  dim3("hdp_max_grid_size", devProp.maxGridSize);
  num("hdp_clock_rate", devProp.clockRate);
  num("hdp_memory_clock_rate", devProp.memoryClockRate);
  num("hdp_memory_bus_width", devProp.memoryBusWidth);
  num("hdp_peak_global_mem_bus_bandwidth", get_global_memory_bus_bandwidth(device_id));
  num("hdp_total_const_mem", devProp.totalConstMem);
  num("hdp_major", devProp.major);
  num("hdp_minor", devProp.minor);
  num("hdp_multi_processor_count", devProp.multiProcessorCount);
  num("hdp_l2_cache_size", devProp.l2CacheSize);
  num("hdp_max_threads_per_multiprocessor", devProp.maxThreadsPerMultiProcessor);
  num("hdp_compute_mode", devProp.computeMode);
  num("hdp_clock_instruction_rate", devProp.clockInstructionRate);
  num("hdp_concurrent_kernels", devProp.concurrentKernels);
  num("hdp_pci_domain_id", devProp.pciDomainID);
  num("hdp_pci_bus_id", devProp.pciBusID);
  num("hdp_pci_device_id", devProp.pciDeviceID);
  num("hdp_max_shared_memory_per_multi_processor", devProp.maxSharedMemoryPerMultiProcessor);
  num("hdp_is_multi_gpu_board", devProp.isMultiGpuBoard);
  num("hdp_can_map_host_memory", devProp.canMapHostMemory);

#ifdef __NVCC__
  // NVIDIA GPUs do not have a GCN Architecture Name; devProp.gcnArchName is uninitialized
  str("hdp_gcn_arch_name", "");
#else
  str("hdp_gcn_arch_name", devProp.gcnArchName);
#endif

  num("hdp_integrated", devProp.integrated);
  num("hdp_cooperative_launch", devProp.cooperativeLaunch);
  num("hdp_cooperative_multi_device_launch", devProp.cooperativeMultiDeviceLaunch);
  num("hdp_max_texture_1d_linear", devProp.maxTexture1DLinear);
  num("hdp_max_texture_1d", devProp.maxTexture1D);
  dim2("hdp_max_texture_2d", devProp.maxTexture2D);
  dim3("hdp_max_texture_3d", devProp.maxTexture3D);
  num("hdp_mem_pitch", devProp.memPitch);
  num("hdp_texture_alignment", devProp.textureAlignment);
  num("hdp_texture_pitch_alignment", devProp.texturePitchAlignment);
  num("hdp_kernel_exec_timeout_enabled", devProp.kernelExecTimeoutEnabled);
  num("hdp_ecc_enabled", devProp.ECCEnabled);
  num("hdp_tcc_driver", devProp.tccDriver);
  num("hdp_cooperative_multi_device_unmatched_func", devProp.cooperativeMultiDeviceUnmatchedFunc);
  num("hdp_cooperative_multi_device_unmatched_grid_dim", devProp.cooperativeMultiDeviceUnmatchedGridDim);
  num("hdp_cooperative_multi_device_unmatched_block_dim", devProp.cooperativeMultiDeviceUnmatchedBlockDim);
  num("hdp_cooperative_multi_device_unmatched_shared_mem", devProp.cooperativeMultiDeviceUnmatchedSharedMem);
  num("hdp_is_large_bar", devProp.isLargeBar);
  num("hdp_asic_revision", devProp.asicRevision);
  num("hdp_managed_memory", devProp.managedMemory);
  num("hdp_direct_managed_mem_access_from_host", devProp.directManagedMemAccessFromHost);
  num("hdp_concurrent_managed_access", devProp.concurrentManagedAccess);
  num("hdp_pageable_memory_access", devProp.pageableMemoryAccess);
  num("hdp_pageable_memory_access_uses_host_page_tables", devProp.pageableMemoryAccessUsesHostPageTables);

  const auto arch = devProp.arch;
  num("hdp_arch_has_global_int32_atomics", arch.hasGlobalInt32Atomics);
  num("hdp_arch_has_global_float_atomic_exch", arch.hasGlobalFloatAtomicExch);
  num("hdp_arch_has_shared_int32_atomics", arch.hasSharedInt32Atomics);
  num("hdp_arch_has_shared_float_atomic_exch", arch.hasSharedFloatAtomicExch);
  num("hdp_arch_has_float_atomic_add", arch.hasFloatAtomicAdd);
  num("hdp_arch_has_global_int64_atomics", arch.hasGlobalInt64Atomics);
  num("hdp_arch_has_shared_int64_atomics", arch.hasSharedInt64Atomics);
  num("hdp_arch_has_doubles", arch.hasDoubles);
  num("hdp_arch_has_warp_vote", arch.hasWarpVote);
  num("hdp_arch_has_warp_ballot", arch.hasWarpBallot);
  num("hdp_arch_has_warp_shuffle", arch.hasWarpShuffle);
  num("hdp_arch_has_funnel_shift", arch.hasFunnelShift);
  num("hdp_arch_has_thread_fence_system", arch.hasThreadFenceSystem);
  num("hdp_arch_has_sync_threads_ext", arch.hasSyncThreadsExt);
  num("hdp_arch_has_surface_funcs", arch.hasSurfaceFuncs);
  num("hdp_arch_has_3d_grid", arch.has3dGrid);
  num("hdp_arch_has_dynamic_parallelism", arch.hasDynamicParallelism);
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
