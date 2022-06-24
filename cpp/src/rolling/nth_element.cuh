/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>

#include <limits>
#include <rmm/exec_policy.hpp>

namespace cudf::detail::rolling {

auto constexpr NULL_INDEX = std::numeric_limits<size_type>::max();  // For nullifying with gather.

template <typename PrecedingIter, typename FollowingIter>
std::unique_ptr<column> nth_element_always_include(
  size_type n,
  null_policy null_handling,  // TODO: Respond to value. Currently, always include nulls.
  column_view const& input,
  PrecedingIter preceding,
  FollowingIter following,
  size_type min_periods,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto gather_iter = cudf::detail::make_counting_transform_iterator(
    0, [preceding, following, min_periods, raw_n = n] __device__(size_type i) {
      auto const window_size =
        preceding[i] + following[i];  // Note: Preceding includes current row.

      if (min_periods > window_size) { return NULL_INDEX; }

      auto const window_start_idx = i - preceding[i] + 1;

      // Normalize n for the window: Wrap around negative index.
      auto const n = raw_n < 0 ? (window_size + raw_n) : raw_n;

      return (n < 0 || n > (window_size - 1)) ? NULL_INDEX : window_start_idx + n;
    });

  auto gathered = cudf::detail::gather(table_view{{input}},
                                       gather_iter,
                                       gather_iter + input.size(),
                                       cudf::out_of_bounds_policy::NULLIFY,
                                       stream,
                                       mr)
                    ->release();
  return std::move(gathered[0]);
}

template <null_policy null_handling, typename PrecedingIter, typename FollowingIter>
std::unique_ptr<column> nth_element(size_type n,
                                    column_view const& input,
                                    PrecedingIter preceding,
                                    FollowingIter following,
                                    size_type min_periods,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto const exclude_nulls = null_handling == null_policy::EXCLUDE and input.nullable();
  auto d_input_ptr         = exclude_nulls ? column_device_view::create(input, stream)
                                           : std::unique_ptr<column_device_view>{nullptr};

  auto gather_iter = cudf::detail::make_counting_transform_iterator(
    0,
    [exclude_nulls, preceding, following, min_periods, n, input_ptr = d_input_ptr.get()] __device__(
      size_type i) {
      // preceding[i] includes the current row.
      auto const window_size = preceding[i] + following[i];
      if (min_periods > window_size) { return NULL_INDEX; }

      auto const wrapped_n = n >= 0 ? n : (window_size + n);
      if (wrapped_n < 0 || wrapped_n > (window_size - 1)) {
        return NULL_INDEX;  // n lies outside the window.
      }

      auto const window_start = i - preceding[i] + 1;

      if (not exclude_nulls) { return window_start + wrapped_n; }

      auto const window_end = window_start + window_size;
      auto const is_valid   = cudf::detail::make_validity_iterator(*input_ptr);
      // Must exclude nulls, and n is in range [-window_size, window_size-1].
      // Depending on n<0, count forwards from window_start, or backwards from window_end.
      if (n >= 0) {
        auto count_down_valids = n;
        // Count forwards from window_start.
        for (auto j = window_start; j < window_end; ++j) {
          if (is_valid[j]) {
            if (count_down_valids == 0) {
              return j;
            } else {
              --count_down_valids;
            }
          }
        }
        return NULL_INDEX;  // n is too large.
      } else {
        auto count_down_valids =
          -n - 1;  // E.g. If n == -3, it is actually at index 2 from window_end.
        // Count backwards from window_end.
        for (auto j = window_end - 1; j >= window_start; --j) {
          if (is_valid[j]) {
            if (count_down_valids == 0) {
              return j;
            } else {
              --count_down_valids;
            }
          }
        }
        return NULL_INDEX;  // n is too large.
      }
    });

  auto gathered = cudf::detail::gather(table_view{{input}},
                                       gather_iter,
                                       gather_iter + input.size(),
                                       cudf::out_of_bounds_policy::NULLIFY,
                                       stream,
                                       mr)
                    ->release();
  return std::move(gathered[0]);
}

}  // namespace cudf::detail::rolling