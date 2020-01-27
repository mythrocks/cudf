/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/aggregation.hpp>
#include <rolling/rolling_detail.hpp>
#include <cudf/rolling.hpp>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/rolling/rolling.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/copying.hpp>

#include <rmm/device_scalar.hpp>

#include <thrust/binary_search.h>
#include <thrust/functional.h>

#include <memory>
#include <algorithm>

namespace cudf {
namespace experimental {

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS((preceding_window >= 0) && (following_window >= 0) && (min_periods >= 0),
               "Window sizes and min periods must be non-negative");

  auto preceding_window_begin = thrust::make_constant_iterator(preceding_window);
  auto following_window_begin = thrust::make_constant_iterator(following_window);

  return cudf::experimental::detail::rolling_window(input, preceding_window_begin,
                                                    following_window_begin, min_periods, aggr, mr, 0);
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr)
{
  if (preceding_window.size() == 0 || following_window.size() == 0) return empty_like(input);

  CUDF_EXPECTS(preceding_window.type().id() == INT32 && following_window.type().id() == INT32,
               "preceding_window/following_window must have INT32 type");

  CUDF_EXPECTS(preceding_window.size() == input.size() && following_window.size() == input.size(),
               "preceding_window/following_window size must match input size");

  return cudf::experimental::detail::rolling_window(input, preceding_window.begin<size_type>(),
                                                    following_window.begin<size_type>(),
                                                    min_periods, aggr, mr, 0);
}

std::unique_ptr<column> rolling_window(column_view const& input,
                                       rmm::device_vector<cudf::size_type> const& group_offsets,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr)
{
  if (group_offsets.empty()) {
    // Empty group_offsets list. Treat `input` as a single group. i.e. Ignore grouping.
    return rolling_window(input, preceding_window, following_window, min_periods, aggr, mr);
  }

  // `group_offsets` are interpreted in adjacent pairs, each pair representing the offsets
  // of the first, and one past the last elements in a group.
  //
  // If `group_offsets` is not empty, it must contain at least two offsets:
  //   a. 0, indicating the first element in `input`
  //   b. input.size(), indicating one past the last element in `input`.
  //
  // Thus, for an input of 1000 rows,
  //   0. [] indicates a single group, spanning the entire column.
  //   1  [10] is invalid.
  //   2. [0, 1000] indicates a single group, spanning the entire column (thus, equivalent to no groups.)
  //   3. [0, 500, 1000] indicates two equal-sized groups: [0,500), and [500,1000).

  CUDF_EXPECTS(group_offsets.size() >= 2 && group_offsets[0] == 0 
               && group_offsets[group_offsets.size()-1] == input.size(),
               "Must have at least one group.");

  auto offsets_begin = group_offsets.begin(); // Required, since __device__ lambdas cannot capture by ref,
  auto offsets_end   = group_offsets.end();   //   or capture local variables without listing them.

  auto preceding_calculator = [offsets_begin, offsets_end, preceding_window] __device__ (size_type idx) {
    // `upper_bound()` cannot return `offsets_end`, since it is capped with `input.size()`.
    auto group_end = thrust::upper_bound(thrust::device, offsets_begin, offsets_end, idx);
    auto group_start = group_end - 1; // The previous offset identifies the start of the group.
    return thrust::minimum<size_type>{}(preceding_window, idx - (*group_start));
  };
 
  auto following_calculator = [offsets_begin, offsets_end, following_window] __device__ (size_type idx) {
    // `upper_bound()` cannot return `offsets_end`, since it is capped with `input.size()`.
    auto group_end = thrust::upper_bound(thrust::device, offsets_begin, offsets_end, idx);
    return thrust::minimum<size_type>{}(following_window, (*group_end - 1) - idx);
  };
  
  return cudf::experimental::detail::rolling_window(
    input,
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), following_calculator),
    min_periods, aggr, mr
  );
}

} // namespace experimental 
} // namespace cudf