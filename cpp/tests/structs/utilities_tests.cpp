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
#include <structs/utilities.hpp>

namespace cudf::test {

/**
 * @brief Round-trip input table through flatten/unflatten,
 *        verify that the table remains equivalent.
 */
void flatten_unflatten_compare(table_view const& input_table)
{
  using namespace cudf::structs::detail;

  auto [flattened, _, __, ___] = flatten_nested_columns(input_table,
                                                        {}, {},
                                                        column_nullability::FORCE);
  auto unflattened = unflatten_nested_columns(std::make_unique<cudf::table>(flattened), input_table);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, unflattened->view());
}

using namespace cudf;
using iterators::null_at;
using strings = strings_column_wrapper;
using structs = structs_column_wrapper;

struct StructUtilitiesTest : BaseFixture {};

template <typename T>
struct TypedStructUtilitiesTest : StructUtilitiesTest {};

TYPED_TEST_CASE(TypedStructUtilitiesTest, FixedWidthTypes);

TYPED_TEST(TypedStructUtilitiesTest, NoStructs)
{
  using T = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto col_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(0)};
  auto col_1 = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)}; 
  auto col_2 = nums{{0,1,2,3,4,5,6}, null_at(6)};

  flatten_unflatten_compare(cudf::table_view{{col_0, col_1, col_2}});
}

TYPED_TEST(TypedStructUtilitiesTest, SingleLevelStruct)
{
  using T = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto member_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(0)};
  auto member_1 = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)}; 
  auto structs_col = structs{{member_0, member_1}};
  auto nums_col    = nums{{0,1,2,3,4,5,6}, null_at(6)};

  flatten_unflatten_compare(cudf::table_view{{nums_col, structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, SingleLevelStructWithNulls)
{
  using T = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto member_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(0)};
  auto member_1 = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)}; 
  auto structs_col = structs{{member_0, member_1}, null_at(2)};
  auto nums_col    = nums{{0,1,2,3,4,5,6}, null_at(6)};

  flatten_unflatten_compare(cudf::table_view{{nums_col, structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, StructOfStruct)
{
  using T = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_col    = nums{{0,1,2,3,4,5,6}, null_at(6)};

  auto member_0_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(0)};
  auto member_0_1 = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)}; 
  auto structs_0_col = structs{{member_0_0, member_0_1}};

  auto member_1_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(3)};
  auto struct_of_structs_col = structs{{member_1_0, structs_0_col}};

  flatten_unflatten_compare(cudf::table_view{{nums_col, struct_of_structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, StructOfStructWithNullsAtLeafLevel)
{
  using T = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_col    = nums{{0,1,2,3,4,5,6}, null_at(6)};

  auto member_0_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(0)};
  auto member_0_1 = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)}; 
  auto structs_0_col = structs{{member_0_0, member_0_1}, null_at(2)};

  auto member_1_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(3)};
  auto struct_of_structs_col = structs{{member_1_0, structs_0_col}};

  flatten_unflatten_compare(cudf::table_view{{nums_col, struct_of_structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, StructOfStructWithNullsAtTopLevel)
{
  using T = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_col    = nums{{0,1,2,3,4,5,6}, null_at(6)};

  auto member_0_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(0)};
  auto member_0_1 = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)}; 
  auto structs_0_col = structs{{member_0_0, member_0_1}};

  auto member_1_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(3)};
  auto struct_of_structs_col = structs{{member_1_0, structs_0_col}, null_at(4)};

  flatten_unflatten_compare(cudf::table_view{{nums_col, struct_of_structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, StructOfStructWithNullsAtAllLevels)
{
  using T = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_col    = nums{{0,1,2,3,4,5,6}, null_at(6)};

  auto member_0_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(0)};
  auto member_0_1 = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)}; 
  auto structs_0_col = structs{{member_0_0, member_0_1}, null_at(2)};

  auto member_1_0 = nums{{0,1,22,333,4444,55555,666666}, null_at(3)};
  auto struct_of_structs_col = structs{{member_1_0, structs_0_col}, null_at(4)};

  flatten_unflatten_compare(cudf::table_view{{nums_col, struct_of_structs_col}});
}

}  // namespace cudf::test
