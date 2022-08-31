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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <src/rolling/detail/rolling.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <vector>

namespace cudf::test::rolling {

template <typename T>
using fwcw = cudf::test::fixed_width_column_wrapper<T>;
template <typename T>
using decimals = cudf::test::fixed_point_column_wrapper<T>;
using ints = fwcw<int32_t>;
using bigints = fwcw<int64_t>;
using namespace numeric;
using namespace cudf::test::iterators;

using column_ptr = std::unique_ptr<cudf::column>;

struct GroupedRollingRangeTest : public BaseFixture 
{
  column_ptr const grouping_keys = ints{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}.release();
  column_ptr const agg_values    = ints{1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}.release();
  cudf::size_type const num_rows = grouping_keys->size();
};

template <typename DecimalT>
struct GroupedRollingRangeOrderByDecimalTest : GroupedRollingRangeTest 
{
  template <typename DecimalT2 = DecimalT>
  auto make_fixed_point_range_bounds(typename DecimalT2::rep value, scale_type scale)
  {
    return cudf::range_window_bounds::get(*cudf::make_fixed_point_scalar<DecimalT2>(value, scale));
  }

  void run_test_preceding_2_following_1(column_view const& order_by, 
                                        range_window_bounds preceding, 
                                        range_window_bounds following)
  {
    auto const results = cudf::grouped_range_rolling_window(cudf::table_view{{grouping_keys->view()}},
                                                            order_by,
                                                            cudf::order::ASCENDING,
                                                            agg_values->view(),
                                                            preceding,
                                                            following, 
                                                            1, // min_periods
                                                            *cudf::make_sum_aggregation<rolling_aggregation>());
    auto const expected_results = bigints{{2, 3, 4, 4, 4, 3, 4, 6, 8, 6, 6, 9, 12, 9}, no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

};

using RepresentationTypes = ::testing::Types<numeric::decimal32>;

TYPED_TEST_SUITE(GroupedRollingRangeOrderByDecimalTest, RepresentationTypes);

TYPED_TEST(GroupedRollingRangeOrderByDecimalTest, BasicGrouping)
{
  using DecimalT = TypeParam; // Decimal type for order_by column.
  using Rep = typename DecimalT::rep; // Representation type for order_by column.

  auto const order_by      = [num_rows = this->num_rows] {
    auto const begin = thrust::make_counting_iterator<Rep>(0);
    return decimals<Rep>{begin, begin + num_rows, scale_type{-2}}.release();
  }();

  auto const preceding = this->template make_fixed_point_range_bounds<DecimalT>(Rep{2}, scale_type{-2});
  auto const following = this->template make_fixed_point_range_bounds<DecimalT>(Rep{1}, scale_type{-2});
  this->run_test_preceding_2_following_1(order_by->view(), preceding, following);
}

} // namespace cudf::test::rolling
