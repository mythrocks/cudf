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

#pragma once

#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/groupby.hpp>

namespace cudf {
namespace test {

inline void test_single_agg(column_view const& keys,
                            column_view const& values,
                            column_view const& expect_keys,
                            column_view const& expect_vals,
                            std::unique_ptr<experimental::aggregation>&& agg,
                            bool ignore_null_keys = true,
                            bool keys_are_sorted = false,
                            std::vector<order> const& column_order = {},
                            std::vector<null_order> const& null_precedence = {})
{
    std::vector<experimental::groupby::aggregation_request> requests;
    requests.emplace_back(
        experimental::groupby::aggregation_request());
    requests[0].values = values;
    
    requests[0].aggregations.push_back(std::move(agg));

    experimental::groupby::groupby gb_obj(table_view({keys}),
        ignore_null_keys, keys_are_sorted, column_order, null_precedence);

    auto result = gb_obj.aggregate(requests);
    expect_tables_equal(table_view({expect_keys}), result.first->view());
    expect_columns_equal(expect_vals, result.second[0].results[0]->view(), true);
}

inline void test_single_rolling_window_agg(
                            column_view const& keys,
                            column_view const& values,
                            column_view const& expect_vals,
                            std::unique_ptr<experimental::aggregation>&& agg,
                            experimental::groupby::window_bounds const& window,
                            bool ignore_null_keys = true,
                            bool keys_are_sorted = true,
                            std::vector<order> const& column_order = {},
                            std::vector<null_order> const& null_precedence = {})
{
    using experimental::aggregation;
    using experimental::groupby::window_bounds;
    using experimental::groupby::window_aggregation_request;

    std::vector<experimental::groupby::window_aggregation_request> requests;
    requests.emplace_back(
        experimental::groupby::window_aggregation_request());
    requests[0].values = values;
    requests[0].aggregations.emplace_back(std::make_pair(window, std::move(agg)));

    experimental::groupby::groupby gb_obj(table_view({keys}),
        ignore_null_keys, keys_are_sorted, column_order, null_precedence);

    auto result = gb_obj.windowed_aggregate(requests);

#if 0
    std::cout << "Results: \n";
    cudf::test::print(*results[0].results[0], std::cout, "\t");
    std::cout << "\n";
    std::cout << "Checking the values: \n";
#endif
    expect_columns_equal(expect_vals, result[0].results[0]->view(), true);
}

inline void test_single_rolling_window_range_frame_agg(
                            column_view const& keys,
                            column_view const& timestamps,
                            column_view const& values,
                            column_view const& expect_vals,
                            std::unique_ptr<experimental::aggregation>&& agg,
                            experimental::groupby::window_bounds const& window,
                            bool ignore_null_keys = true,
                            bool keys_are_sorted = true,
                            std::vector<order> const& column_order = {},
                            std::vector<null_order> const& null_precedence = {})
{
    using experimental::aggregation;
    using experimental::groupby::window_bounds;
    using experimental::groupby::window_aggregation_request;

    std::vector<experimental::groupby::time_range_window_aggregation_request> requests;
    requests.emplace_back(
        experimental::groupby::time_range_window_aggregation_request());
    requests[0].values = values;
    requests[0].timestamps = timestamps;
    requests[0].aggregations.emplace_back(std::make_pair(window, std::move(agg)));

    experimental::groupby::groupby gb_obj(table_view({keys}),
        ignore_null_keys, keys_are_sorted, column_order, null_precedence);

    auto result = gb_obj.time_range_windowed_aggregate(requests);

#if 0
    std::cout << "Results: \n";
    cudf::test::print(*results[0].results[0], std::cout, "\t");
    std::cout << "\n";
    std::cout << "Checking the values: \n";
#endif
    expect_columns_equal(expect_vals, result[0].results[0]->view(), true);
}
inline auto all_valid() {
    auto all_valid = make_counting_transform_iterator(
        0, [](auto i) { return true; });
    return all_valid;
}

inline auto all_null() {
    auto all_null = make_counting_transform_iterator(
        0, [](auto i) { return false; });
    return all_null;
}

} // namespace test
} // namespace cudf
