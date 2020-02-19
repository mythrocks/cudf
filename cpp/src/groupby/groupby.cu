/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/rolling/rolling.cuh>
#include <cudf/groupby.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <thrust/functional.h>

#include <memory>
#include <utility>

namespace cudf {
namespace experimental {
namespace groupby {

// Constructor
groupby::groupby(table_view const& keys, bool ignore_null_keys,
                 bool keys_are_sorted, std::vector<order> const& column_order,
                 std::vector<null_order> const& null_precedence)
    : _keys{keys},
      _ignore_null_keys{ignore_null_keys},
      _keys_are_sorted{keys_are_sorted},
      _column_order{column_order},
      _null_precedence{null_precedence} {}

// Select hash vs. sort groupby implementation
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
groupby::dispatch_aggregation(std::vector<aggregation_request> const& requests,
                              cudaStream_t stream,
                              rmm::mr::device_memory_resource* mr) {
  // If sort groupby has been called once on this groupby object, then
  // always use sort groupby from now on. Because once keys are sorted, 
  // all the aggs that can be done by hash groupby are efficiently done by
  // sort groupby as well.
  // Only use hash groupby if the keys aren't sorted and all requests can be
  // satisfied with a hash implementation
  if (not _keys_are_sorted and
      not _helper and
      detail::hash::can_use_hash_groupby(_keys, requests)) {
    return detail::hash::groupby(_keys, requests, _ignore_null_keys, stream,
                                 mr);
  } else {
    return sort_aggregate(requests, stream, mr);
  }
}

// Destructor
// Needs to be in source file because sort_groupby_helper was forward declared
groupby::~groupby() = default;

namespace {
/// Make an empty table with appropriate types for requested aggs
template <typename aggregation_request_t, typename F>
auto templated_empty_results(std::vector<aggregation_request_t> const& requests, F get_kind) {
  
  std::vector<aggregation_result> empty_results;

  std::transform(
      requests.begin(), requests.end(), std::back_inserter(empty_results),
      [&get_kind](auto const& request) {
        std::vector<std::unique_ptr<column>> results;

        std::transform(
            request.aggregations.begin(), request.aggregations.end(),
            std::back_inserter(results), [&request, get_kind](auto const& agg) {
              return make_empty_column(experimental::detail::target_type(
                  request.values.type(), get_kind(agg)));
            });

        return aggregation_result{std::move(results)};
      });

  return empty_results;
}

/// Verifies the agg requested on the request's values is valid
void verify_valid_requests(std::vector<aggregation_request> const& requests) {
  CUDF_EXPECTS(
      std::all_of(requests.begin(), requests.end(),
                  [](auto const& request) {
                    return std::all_of(
                        request.aggregations.begin(),
                        request.aggregations.end(),
                        [&request](auto const& agg) {
                          return experimental::detail::is_valid_aggregation(
                              request.values.type(), agg->kind);
                        });
                  }),
      "Invalid type/aggregation combination.");
}

/**
 * @brief  Applies a fixed-size rolling window function to the values in a column.
 *
 * This function aggregates values in a window around each element i of the input column, and
 * invalidates the bit mask for element i if there are not enough observations. The window size is
 * static (the same for each element). This matches Pandas' API for DataFrame.rolling with a few
 * notable differences:
 * - instead of the center flag it uses a two-part window to allow for more flexible windows.
 *   The total window size = `preceding_window + following_window + 1`. Element `i` uses elements
 *   `[i-preceding_window, i+following_window]` to do the window computation, provided that they
 *   fall within the confines of their corresponding groups, as indicated by `group_offsets`.
 * - instead of storing NA/NaN for output rows that do not meet the minimum number of observations
 *   this function updates the valid bitmask of the column to indicate which elements are valid.
 * 
 * The returned column for `op == COUNT` always has `INT32` type. All other operators return a 
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * @param[in] input_col The input column
 * @param[in] group_offsets A column of indexes, indicating partition/grouping boundaries.
 * @param[in] preceding_window The static rolling window size in the backward direction.
 * @param[in] following_window The static rolling window size in the forward direction.
 * @param[in] min_periods Minimum number of observations in window required to have a value,
 *                        otherwise element `i` is null.
 * @param[in] op The rolling window aggregation type (SUM, MAX, MIN, etc.)
 *
 * @returns   A nullable output column containing the rolling window results
 **/

std::unique_ptr<column> rolling_window(column_view const& input,
                                       rmm::device_vector<cudf::size_type> const& group_offsets,
                                       rmm::device_vector<cudf::size_type> const& group_labels,
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

#ifndef NDEBUG
  CUDF_EXPECTS(group_offsets.size() >= 2 && group_offsets[0] == 0 
               && group_offsets[group_offsets.size()-1] == input.size(),
               "Must have at least one group.");
#endif // NDEBUG

  auto preceding_calculator = 
    [
      d_group_offsets = group_offsets.data().get(),
      d_group_labels  = group_labels.data().get(),
      preceding_window
    ] __device__ (size_type idx) {
      auto group_label = d_group_labels[idx];
      auto group_start = d_group_offsets[group_label];
      return thrust::minimum<size_type>{}(preceding_window, idx - group_start);
    };
 
  auto following_calculator = 
    [
      d_group_offsets = group_offsets.data().get(),
      d_group_labels  = group_labels.data().get(),
      following_window
    ] __device__ (size_type idx) {
      auto group_label = d_group_labels[idx];
      auto group_end = d_group_offsets[group_label+1]; // Cannot fall off the end, since offsets is capped with `input.size()`.
      return thrust::minimum<size_type>{}(following_window, (group_end - 1) - idx);
    };

  return cudf::experimental::detail::rolling_window(
    input,
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), following_calculator),
    min_periods, aggr, mr
  );
}

bool is_supported_range_frame_unit(cudf::data_type const& data_type) {
  auto id = data_type.id();
  return id == cudf::TIMESTAMP_DAYS
      || id == cudf::TIMESTAMP_SECONDS
      || id == cudf::TIMESTAMP_MILLISECONDS
      || id == cudf::TIMESTAMP_MICROSECONDS
      || id == cudf::TIMESTAMP_NANOSECONDS;
}

size_t multiplication_factor(cudf::data_type const& data_type) {
  // Assume timestamps.
  switch(data_type.id()) {
    case cudf::TIMESTAMP_DAYS         : return 1L;
    case cudf::TIMESTAMP_SECONDS      : return 24L*60*60;
    case cudf::TIMESTAMP_MILLISECONDS : return 24L*60*60*1000;
    case cudf::TIMESTAMP_MICROSECONDS : return 24L*60*60*1000*1000;
    default  : 
      CUDF_EXPECTS(data_type.id() == cudf::TIMESTAMP_NANOSECONDS, 
                   "Unexpected data-type for timestamp-based rolling window operation!");
      return 24L*60*60*1000*1000*1000;
  }
}

template <typename TimestampImpl_t>
std::unique_ptr<column> range_frame_rolling_window( column_view const& input,
                                                    column_view const& timestamp_column,
                                                    rmm::device_vector<cudf::size_type> const& group_offsets,
                                                    rmm::device_vector<cudf::size_type> const& group_labels,
                                                    size_type preceding_window_in_days, // TODO: Consider taking offset-type as type_id. Assumes days for now.
                                                    size_type following_window_in_days,
                                                    size_type min_periods,
                                                    std::unique_ptr<aggregation> const& aggr,
                                                    rmm::mr::device_memory_resource* mr) {

  TimestampImpl_t mult_factor {static_cast<TimestampImpl_t>(multiplication_factor(timestamp_column.type()))};
 
  auto preceding_calculator = 
    [
      d_group_offsets = group_offsets.data().get(),
      d_group_labels  = group_labels.data().get(),
      d_timestamps    = timestamp_column.data<TimestampImpl_t>(),
      preceding_window_in_days,
      mult_factor
    ] __device__ (size_type idx) {
      auto group_label = d_group_labels[idx];
      auto group_start = d_group_offsets[group_label];
      auto lower_bound = d_timestamps[idx] - preceding_window_in_days*mult_factor;

      auto preceding_i{idx};
      while (preceding_i >= group_start && d_timestamps[preceding_i] >= lower_bound) {
        --preceding_i;
      }

      return idx - preceding_i - 1;
    };
 
  auto following_calculator = 
    [
      d_group_offsets = group_offsets.data().get(),
      d_group_labels  = group_labels.data().get(),
      d_timestamps    = timestamp_column.data<TimestampImpl_t>(),
      following_window_in_days,
      mult_factor
    ] __device__ (size_type idx) {
      auto group_label = d_group_labels[idx];
      auto group_end = d_group_offsets[group_label+1]; // Cannot fall off the end, since offsets is capped with `input.size()`.
      auto upper_bound = d_timestamps[idx] + following_window_in_days*mult_factor;

      auto following_i{idx};
      while (following_i < group_end && d_timestamps[following_i] <= upper_bound) {
        ++following_i;
      }

      return following_i - idx - 1;
    };

  return cudf::experimental::detail::rolling_window(
    input,
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), following_calculator),
    min_periods, aggr, mr
  );
  
}

std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& timestamp_column,
                                       rmm::device_vector<cudf::size_type> const& group_offsets,
                                       rmm::device_vector<cudf::size_type> const& group_labels,
                                       size_type preceding_window_in_days, // TODO: Consider taking offset-type as type_id. Assumes days for now.
                                       size_type following_window_in_days,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr)
{
  // Assumes that `group_offsets` starts with `0`, ends with `input.size`
#ifndef NDEBUG
  CUDF_EXPECTS(group_offsets.size() >= 2 && group_offsets[0] == 0 
               && group_offsets[group_offsets.size()-1] == input.size(),
               "Must have at least one group.");
#endif // NDEBUG

  // Assumes that `timestamp_column` is sorted in ascending, per group.
  CUDF_EXPECTS(is_supported_range_frame_unit(timestamp_column.type()),
               "Unsupported data-type for `timestamp`-based rolling window operation!");

  return timestamp_column.type().id() == cudf::TIMESTAMP_DAYS?
          range_frame_rolling_window<int32_t>(input, timestamp_column, group_offsets, 
            group_labels, preceding_window_in_days, following_window_in_days, min_periods, aggr, mr)
        : 
          range_frame_rolling_window<int64_t>(input, timestamp_column, group_offsets, 
            group_labels, preceding_window_in_days, following_window_in_days, min_periods, aggr, mr);
    
}

}  // namespace

// Compute aggregation requests
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
groupby::aggregate(std::vector<aggregation_request> const& requests,
                   rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(std::all_of(requests.begin(), requests.end(),
                           [this](auto const& request) {
                             return request.values.size() == _keys.num_rows();
                           }),
               "Size mismatch between request values and groupby keys.");

  verify_valid_requests(requests);

  if (_keys.num_rows() == 0) {
    std::make_pair(empty_like(_keys), 
                   templated_empty_results(requests, 
                                           [](std::unique_ptr<aggregation> const& agg)
                                           {return agg->kind;}));
  }

  return dispatch_aggregation(requests, 0, mr);
}

// Get the sort helper object
detail::sort::sort_groupby_helper& groupby::helper() {
  if (_helper)
    return *_helper;
  _helper = std::make_unique<detail::sort::sort_groupby_helper>(
    _keys, _ignore_null_keys, _keys_are_sorted);
  return *_helper;
};

std::vector<aggregation_result> groupby::windowed_aggregate(
    std::vector<window_aggregation_request> const& requests,
    rmm::mr::device_memory_resource* mr) {

  CUDF_EXPECTS(std::all_of(requests.begin(), requests.end(),
                           [this](auto const& request) {
                             return request.values.size() == _keys.num_rows();
                           }),
               "Size mismatch between request values and groupby keys.");

  CUDF_EXPECTS(this->_keys_are_sorted, 
               "Window-aggregation is currently supported only on pre-sorted key columns.");

  if (_keys.num_rows() == 0) {
    std::make_pair(empty_like(_keys), 
                  templated_empty_results(requests, 
                                          [](std::pair<window_bounds, std::unique_ptr<aggregation>> const& agg) 
                                          {return agg.second->kind;}));
  }

  auto group_offsets = helper().group_offsets();
  auto group_labels  = helper().group_labels();
  group_offsets.push_back(_keys.num_rows()); // Cap the end.

  std::vector<aggregation_result> results;
  std::transform(
    requests.begin(), requests.end(), std::back_inserter(results),
    [&](auto const& window_request) {
      std::vector<std::unique_ptr<column>> per_request_results;
      auto const& values = window_request.values;
      std::transform(
        window_request.aggregations.begin(), window_request.aggregations.end(), 
        std::back_inserter(per_request_results),
        [&](std::pair<window_bounds, std::unique_ptr<aggregation>> const& agg) {
          return rolling_window(
            values,
            group_offsets,
            group_labels,
            agg.first.preceding,
            agg.first.following,
            agg.first.min_periods,
            agg.second,
            mr
          );
        }
      );
      return aggregation_result{std::move(per_request_results)};
    }
  );

  return std::move(results);
}

std::vector<aggregation_result> groupby::time_range_windowed_aggregate(
    std::vector<time_range_window_aggregation_request> const& requests,
    rmm::mr::device_memory_resource* mr) {

  CUDF_EXPECTS(std::all_of(requests.begin(), requests.end(),
                           [this](auto const& request) {
                             return request.values.size() == _keys.num_rows();
                           }),
               "Size mismatch between request values and groupby keys.");

  CUDF_EXPECTS(this->_keys_are_sorted, 
               "Window-aggregation is currently supported only on pre-sorted key columns.");

  if (_keys.num_rows() == 0) {
    std::make_pair(empty_like(_keys), 
                  templated_empty_results(requests, 
                                          [](std::pair<window_bounds, std::unique_ptr<aggregation>> const& agg) 
                                          {return agg.second->kind;}));
  }

  auto group_offsets = helper().group_offsets();
  auto group_labels  = helper().group_labels();
  group_offsets.push_back(_keys.num_rows()); // Cap the end.

  std::vector<aggregation_result> results;
  std::transform(
    requests.begin(), requests.end(), std::back_inserter(results),
    [&](auto const& window_request) {
      std::vector<std::unique_ptr<column>> per_request_results;
      auto const& values = window_request.values;
      auto const& timestamps = window_request.timestamps;
      std::transform(
        window_request.aggregations.begin(), window_request.aggregations.end(), 
        std::back_inserter(per_request_results),
        [&](std::pair<window_bounds, std::unique_ptr<aggregation>> const& agg) {
          return rolling_window(
            values,
            timestamps,
            group_offsets,
            group_labels,
            agg.first.preceding, // TODO: Currently assumes DAYS.
            agg.first.following, // TODO: Currently assumes DAYS.
            agg.first.min_periods,
            agg.second,
            mr
          );
        }
      );
      return aggregation_result{std::move(per_request_results)};
    }
  );

  return std::move(results);
}

}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
