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

  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grp_col}};
  auto const preceding     = T{1};
  auto const following     = T{1};
  auto const min_periods   = 1L;
  auto const output        = cudf::grouped_range_rolling_window(grouping_keys,
                                                              oby_col,
                                                              cudf::order::ASCENDING,
                                                              agg_col,
                                                              range_bounds(numeric_scalar<T>{preceding, true}),
                                                              range_bounds(numeric_scalar<T>{following, true}),
                                                              min_periods,
                                                              cudf::make_count_aggregation());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {4, 4, 4, 4, 1, 2, 2, 3, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountSingleGroupTimestampASCNullsLast)
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

  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grp_col}};
  auto const preceding     = T{1L};
  auto const following     = T{1L};
  auto const min_periods   = 1L;
  auto const output        = cudf::grouped_range_rolling_window(grouping_keys,
                                                              oby_col,
                                                              cudf::order::ASCENDING,
                                                              agg_col,
                                                              range_bounds(numeric_scalar<T>{preceding, true}),
                                                              range_bounds(numeric_scalar<T>{following, true}),
                                                              min_periods,
                                                              cudf::make_count_aggregation());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {2, 3, 3, 3, 2, 1, 4, 4, 4, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountMultiGroupTimestampASCNullsFirst)
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

  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grp_col}};
  auto const preceding     = T{1};
  auto const following     = T{1};
  auto const min_periods   = 1L;
  auto const output        = cudf::grouped_range_rolling_window(grouping_keys,
                                                              oby_col,
                                                              cudf::order::ASCENDING,
                                                              agg_col,
                                                              range_bounds(numeric_scalar<T>{preceding, true}),
                                                              range_bounds(numeric_scalar<T>{following, true}),
                                                              min_periods,
                                                              cudf::make_count_aggregation());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {3, 3, 3, 2, 2, 2, 2, 2, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountMultiGroupTimestampASCNullsLast)
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

  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grp_col}};
  auto const preceding     = T{1};
  auto const following     = T{1};
  auto const min_periods   = 1L;
  auto const output        = cudf::grouped_range_rolling_window(grouping_keys,
                                                              oby_col,
                                                              cudf::order::ASCENDING,
                                                              agg_col,
                                                              range_bounds(numeric_scalar<T>{preceding, true}),
                                                              range_bounds(numeric_scalar<T>{following, true}),
                                                              min_periods,
                                                              cudf::make_count_aggregation());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {3, 3, 3, 2, 2, 2, 3, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, CountSingleGroupTimestampDESCNullsFirst)
{
  using namespace cudf::test;
  using T = TypeParam;

  auto const grp_col = fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto const agg_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  auto const time_col = fixed_width_column_wrapper<T>{
    {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grp_col}};
  auto const preceding     = T{1};
  auto const following     = T{1};
  auto const min_periods   = 1L;
  auto const output        = cudf::grouped_range_rolling_window(grouping_keys,
                                                              time_col,
                                                              cudf::order::DESCENDING,
                                                              agg_col,
                                                              range_bounds(numeric_scalar<T>{preceding, true}),
                                                              range_bounds(numeric_scalar<T>{following, true}),
                                                              min_periods,
                                                              cudf::make_count_aggregation());


  /*
  std::cout << "Test subtraction: T{0} - T{1} == " << (T{0} - T{1}) << std::endl;
  std::cout << "Numeric limits for unsigned: min == " << std::numeric_limits<T>::min() << " max == " << std::numeric_limits<T>::max() << std::endl;
  std::cout << "Prev: " << std::endl;
  std::cout << "1,2,3,4,1,2,2,2,2,2" << std::endl;
  std::cout << "Foll: " << std::endl;
  print(output->view());
  */

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 fixed_width_column_wrapper<cudf::size_type>{
                                   {4, 4, 4, 4, 1, 2, 2, 3, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingTest, Modulo)
{
  std::cout << "is_modulo(" << typeid(TypeParam).name() << ") == " << std::boolalpha << std::numeric_limits<TypeParam>::is_modulo << std::endl;
  std::cout << "is_signed(" << typeid(TypeParam).name() << ") == " << std::boolalpha << std::numeric_limits<TypeParam>::is_signed << std::endl;
}

} // namespace test;
} // namespace cudf;
