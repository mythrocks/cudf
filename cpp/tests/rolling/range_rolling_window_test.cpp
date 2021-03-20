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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <src/rolling/rolling_detail.hpp>
#include <src/rolling/range_window_bounds_detail.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <vector>

using cudf::detail::range_bounds;

namespace cudf {
namespace test {

struct RangeRollingTest : public BaseFixture {};

template <typename T>
struct TypedRangeRollingTest : public RangeRollingTest {};

using TypesUnderTest = IntegralTypesNotBool;

TYPED_TEST_CASE(TypedRangeRollingTest, TypesUnderTest);

template <typename T>
auto do_count_over_window(cudf::column_view grouping_col,
                          cudf::column_view order_by,
                          cudf::order order,
                          cudf::column_view aggregation_col,
                          range_window_bounds&& preceding = range_bounds(numeric_scalar<T>{T{1}, true}),
                          range_window_bounds&& following = range_bounds(numeric_scalar<T>{T{1}, true}))
{
  auto const min_periods = size_type{1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_col}};

  return cudf::grouped_range_rolling_window(grouping_keys,
                                            order_by,
                                            order,
                                            aggregation_col,
                                            std::move(preceding),
                                            std::move(following),
                                            min_periods,
                                            cudf::make_count_aggregation());
}

TYPED_TEST(TypedRangeRollingTest, CountSingleGroupOrderByASCNullsFirst)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const output        = do_count_over_window<T>(grp_col,
                                                     oby_col,
                                                     cudf::order::ASCENDING,
                                                     agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {4, 4, 4, 4, 1, 2, 2, 3, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountSingleGroupOrderByASCNullsLast)
{
  using namespace cudf::test;
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 1, 0, 0, 0, 0}};

  auto const output        = do_count_over_window<T>(grp_col,
                                                     oby_col,
                                                     cudf::order::ASCENDING,
                                                     agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {2, 3, 3, 3, 2, 1, 4, 4, 4, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountMultiGroupOrderByASCNullsFirst)
{
  using namespace cudf::test;
  using T = TypeParam;

  // Groupby column.
  auto const grp_col  = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col  = fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {1, 2, 2, 1, 2, 1, 2, 3, 4, 5}, {0, 0, 0, 1, 1, 0, 0, 1, 1, 1}};

  auto const output        = do_count_over_window<T>(grp_col,
                                                     oby_col,
                                                     cudf::order::ASCENDING,
                                                     agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {3, 3, 3, 2, 2, 2, 2, 2, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountMultiGroupOrderByASCNullsLast)
{
  using namespace cudf::test;
  using T = int32_t;

  // Groupby column.
  auto const grp_col  = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col  = fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {1, 2, 2, 1, 3, 1, 2, 3, 4, 5}, {1, 1, 1, 0, 0, 1, 1, 1, 0, 0}};

  auto const output        = do_count_over_window<T>(grp_col,
                                                     oby_col,
                                                     cudf::order::ASCENDING,
                                                     agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {3, 3, 3, 2, 2, 2, 3, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountSingleGroupOrderByDESCNullsFirst)
{
  using namespace cudf::test;
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const output        = do_count_over_window<T>(grp_col,
                                                     oby_col,
                                                     cudf::order::DESCENDING,
                                                     agg_col);;


  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {4, 4, 4, 4, 1, 2, 2, 3, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountSingleGroupOrderByDESCNullsLast)
{
  using namespace cudf::test;
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, {1, 1, 1, 1, 1, 1, 0, 0, 0, 0}};

  auto const output        = do_count_over_window<T>(grp_col,
                                                     oby_col,
                                                     cudf::order::DESCENDING,
                                                     agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {2, 3, 3, 3, 2, 1, 4, 4, 4, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountMultiGroupOrderByDESCNullsFirst)
{
  using namespace cudf::test;
  using T = TypeParam;

  // Groupby column.
  auto const grp_col  = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col  = fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {4, 3, 2, 1, 0, 9, 8, 7, 6, 5}, {0, 0, 0, 1, 1, 0, 0, 1, 1, 1}};

  auto const output        = do_count_over_window<T>(grp_col,
                                                     oby_col,
                                                     cudf::order::DESCENDING,
                                                     agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {3, 3, 3, 2, 2, 2, 2, 2, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountMultiGroupOrderByDESCNullsLast)
{
  using namespace cudf::test;
  using T = TypeParam;

  // Groupby column.
  auto const grp_col  = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col  = fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {4, 3, 2, 1, 0, 9, 8, 7, 6, 5}, {1, 1, 1, 0, 0, 1, 1, 1, 0, 0}};

  auto const output  = do_count_over_window<T>(grp_col,
                                               oby_col,
                                               cudf::order::DESCENDING,
                                               agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {2, 3, 2, 2, 2, 2, 3, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountSingleGroupAllNullOrderBys)
{
  using namespace cudf::test;
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

  auto const output  = do_count_over_window<T>(grp_col,
                                               oby_col,
                                               cudf::order::ASCENDING,
                                               agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {9, 9, 9, 9, 9, 9, 9, 9, 9, 9}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountMultiGroupAllNullOrderBys)
{
  using namespace cudf::test;
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  // OrderBy column.
  auto const oby_col = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};

  auto const output  = do_count_over_window<T>(grp_col,
                                               oby_col,
                                               cudf::order::ASCENDING,
                                               agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {2, 3, 3, 3, 2, 4, 4, 4, 4, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, UnboundedPrecedingWindowSingleGroupOrderByASCNullsFirst)
{
  using namespace cudf::test;
  using T = TypeParam;

  auto const grp_col = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto const agg_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  auto const oby_col = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const output  = do_count_over_window<T>(grp_col,
                                               oby_col,
                                               cudf::order::ASCENDING,
                                               agg_col,
                                               range_window_bounds::unbounded(data_type{type_to_id<T>()}),
                                               range_bounds(numeric_scalar<T>{1, true}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {4, 4, 4, 4, 5, 6, 7, 8, 9, 9}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, UnboundedFollowingWindowSingleGroupOrderByASCNullsFirst)
{
  using namespace cudf::test;
  using T = TypeParam;

  auto const grp_col = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto const agg_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  auto const oby_col = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const output  = do_count_over_window<T>(grp_col,
                                               oby_col,
                                               cudf::order::ASCENDING,
                                               agg_col,
                                               range_bounds(numeric_scalar<T>{1, true}),
                                               range_window_bounds::unbounded(data_type{type_to_id<T>()}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {9, 9, 9, 9, 5, 5, 4, 4, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

} // namespace test;
} // namespace cudf;
