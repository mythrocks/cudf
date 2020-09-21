/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/maps/map_lookup.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <rmm/device_buffer.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include "cudf/scalar/scalar.hpp"

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;
using cudf::size_type;

struct MapLookupTest : public cudf::test::BaseFixture {
};

TEST_F(MapLookupTest, Basics)
{
  using namespace cudf::test;

  // auto keys = strings_column_wrapper{"0","1", "1","2", "2","3"};
  auto keys = strings_column_wrapper{"0","1", "1","2", "2","3"};
  auto values = strings_column_wrapper{"00","11", "11","22", "22","33"};
  auto pairs = structs_column_wrapper{
    {keys, values}
  }.release();

  auto maps = cudf::make_lists_column(
    3,
    fixed_width_column_wrapper<size_type>{{0,2,4,6}}.release(),
    std::move(pairs),
    cudf::UNKNOWN_NULL_COUNT,
    {}
  );

  std::cout << "Input map: ";
  print(maps->view());
  std::cout << std::endl;

  auto lookup_key = cudf::make_string_scalar("1");
  auto string_lookup_key = *static_cast<cudf::string_scalar*>(lookup_key.get());
  auto lookup = cudf::two_pass_map_lookup(maps->view(), string_lookup_key);

  std::cout << "Output vector: ";
  print(lookup->view());
  std::cout << std::endl;

  /*
  std::cout << "Testing gather. --------------" << std::endl;
  auto strings = strings_column_wrapper{"A", "B", "C", "D", "E"};
  auto table_for_gather = cudf::table_view{std::vector<cudf::column_view>{strings}};
  auto gathered = cudf::detail::gather(
    table_for_gather,
    fixed_width_column_wrapper<size_type>{4,3,2,1,-1},
    cudf::detail::out_of_bounds_policy::NULLIFY,
    cudf::detail::negative_index_policy::NOT_ALLOWED
  );
  std::cout << "With NULLIFY,NOT_ALLOWED, got: "; print(gathered->get_column(0)); std::cout << std::endl;
  std::cout << "HasNulls: " << gathered->get_column(0).view().has_nulls() << std::endl;

  gathered = cudf::detail::gather(
    table_for_gather,
    fixed_width_column_wrapper<size_type>{4,3,2,1,-1},
    cudf::detail::out_of_bounds_policy::IGNORE,
    cudf::detail::negative_index_policy::NOT_ALLOWED
  );
  std::cout << "With IGNORE,NOT_ALLOWED, got: "; print(gathered->get_column(0)); std::cout << std::endl;
  */
}

CUDF_TEST_PROGRAM_MAIN()
