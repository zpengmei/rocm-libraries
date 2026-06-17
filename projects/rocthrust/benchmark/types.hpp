// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <cstdint>

// Types used in the benchmarks
#if defined(_MSC_VER)
#  define THRUST_BENCHMARKS_HAVE_INT128_SUPPORT 0
#else
#  define THRUST_BENCHMARKS_HAVE_INT128_SUPPORT 1
#endif

namespace bench_utils
{
class large_data
{
public:
  __host__ __device__ large_data()
  {
    data[0] = 0;
  }
  __host__ __device__ large_data(large_data const& val)
  {
    data[0] = val.data[0];
  }
  __host__ __device__ large_data(int n)
  {
    data[0] = static_cast<int8_t>(n);
  }
  large_data& __host__ __device__ operator=(large_data const& val)
  {
    data[0] = val.data[0];
    return *this;
  }
  bool __host__ __device__ operator==(large_data const& val) const
  {
    return data[0] == val.data[0];
  }
  large_data& __host__ __device__ operator++()
  {
    ++data[0];
    return *this;
  }
  __host__ __device__ operator int() const
  {
    return static_cast<int>(data[0]);
  }

  int8_t data[512];
};

template <class T>
bool __host__ __device__ operator==(T const& lhs, large_data const& rhs)
{
  return static_cast<large_data>(lhs).data[0] == rhs.data[0];
}

}; // namespace bench_utils
