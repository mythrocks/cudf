/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <set>

namespace cudf::detail {

bool can_compute_via_aggregation(bool unbounded_preceding,
                                 bool unbounded_following,
                                 size_type min_periods,
                                 rolling_aggregation const& agg)
{
  auto is_supported = [](auto const& agg) {
    switch (agg.kind) {
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID:
      case cudf::aggregation::Kind::SUM:
      case cudf::aggregation::Kind::MIN:
      case cudf::aggregation::Kind::MAX:
        // TODO (future): COLLECT_LIST and COLLECT_SET can be added at a later date.
        return true;
      default:
        // Other aggregations do not fit into the [UNBOUNDED, UNBOUNDED]
        // category. For instance:
        // 1. Ranking functions (ROW_NUMBER, RANK, DENSE_RANK, PERCENT_RANK)
        //    use [UNBOUNDED PRECEDING, CURRENT ROW].
        // 2. LEAD/LAG are defined on finite row boundaries.
        return false;
    }
  };

  return unbounded_preceding && unbounded_following && (min_periods == 1) && is_supported(agg);
}

std::unique_ptr<cudf::groupby_aggregation> to_groupby_agg(cudf::rolling_aggregation const& aggr)
{
  switch (aggr.kind) {
    case cudf::aggregation::Kind::COUNT_ALL:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(null_policy::INCLUDE);
    case cudf::aggregation::Kind::COUNT_VALID:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(null_policy::EXCLUDE);
    case cudf::aggregation::Kind::SUM:
      return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::Kind::MIN:
      return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::Kind::MAX:
      return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::Kind::COLLECT_LIST:  // TODO (future).
    case cudf::aggregation::Kind::COLLECT_SET:   // TODO (future).

    default:
      CUDF_FAIL("Unsupported aggregation kind: " + std::to_string(static_cast<int>(aggr.kind)));
  }
}

std::unique_ptr<column> aggregation_based_rolling_window(table_view const& group_keys,
                                                         column_view const& input,
                                                         rolling_aggregation const& aggr,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource* mr)
{
  // TODO: Handle case where there are no group_keys.
  if (group_keys.num_columns() == 0) {
    CUDF_FAIL("Ungrouped rolling window not implemented via aggregations yet. ");
  }

  auto agg_requests = std::vector<cudf::groupby::aggregation_request>{};
  agg_requests.push_back(cudf::groupby::aggregation_request());
  agg_requests.front().values = input;
  agg_requests.front().aggregations.push_back(to_groupby_agg(aggr));

  auto group_by = cudf::groupby::groupby{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES};
  // TODO: Create detail API for groupby.aggregate() to take stream. But use default mr, for temp.
  auto aggregation_results           = group_by.aggregate(agg_requests);
  auto const& aggregation_result_col = aggregation_results.second.front().results.front();

  using cudf::groupby::detail::sort::sort_groupby_helper;
  auto helper = sort_groupby_helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES, {}};
  auto const& group_labels = helper.group_labels(stream);

  auto result_columns = cudf::detail::gather(cudf::table_view{{*aggregation_result_col}},
                                             group_labels,
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             cudf::detail::negative_index_policy::NOT_ALLOWED,
                                             stream,
                                             mr)
                          ->release();
  return std::move(result_columns.front());
}

}  // namespace cudf::detail
