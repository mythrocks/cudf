/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <structs/utilities.hpp>

namespace cudf::test {

struct UtilitiesTest : BaseFixture {};

TEST_F(UtilitiesTest, flatten_lists)
{
  // TODO: What's expected to happen here?
  // Expected: Busted.
}

TEST_F(UtilitiesTest, flatten_structs)
{
  using namespace cudf;
  using namespace cudf::groupby;
  using iterators::null_at;
  using ints = fixed_width_column_wrapper<int32_t>;

  auto child_0       = ints{{0, 1, 2, 3, 4, 5}, null_at(0)};
  auto child_1       = ints{0, 1, 2, 3, 4, 5};
  auto structs_col = structs_column_wrapper{{child_0, child_1}, null_at(2)};
  auto struct_o_structs_col = structs_column_wrapper{{structs_col}};

  auto ints_col    = ints{{0, 2, 4, 6, 8, 10}, null_at(2)};

  auto input_table = table_view{{ints_col, struct_o_structs_col}};
  auto agg_values  = ints{1,1,1,1,1,1}.release();
  std::cout << "Input table: " << std::endl;
  for (int i{0}; i<input_table.num_columns(); ++i)
  {
    print(input_table.column(i));
  }
  std::cout << std::endl;

  auto flattened = structs::detail::flatten_nested_columns(input_table,
                                                           {},
                                                           {},
                                                           structs::detail::column_nullability::FORCE);
  auto& output_table = std::get<0>(flattened);
  auto& nullability_vectors = std::get<3>(flattened);

  std::cout << "Output table: " << std::endl;
  for (auto col : output_table)
  {
      print(col);
  }

  std::cout << "\nNullability vectors: " << std::endl;
  for (auto& x : nullability_vectors)
  {
      print(x->view());
  }

  std::cout << "\nAttempting reconstruction." << std::endl;

  std::unique_ptr<cudf::table> flattened_table = std::make_unique<cudf::table>(output_table);
  
  auto unflattened = structs::detail::unflatten_nested_columns(std::move(flattened_table), 
                                                               input_table, 
                                                               std::move(nullability_vectors));

  std::cout << "Unflattened column: " << std::endl;
  for (auto col : unflattened->view()) {
    print(col);
  }

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, unflattened->view());
}

}  // namespace cudf::test
