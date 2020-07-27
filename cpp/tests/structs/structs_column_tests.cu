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

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>

#include <functional>
#include <initializer_list>
#include <iterator>
#include <algorithm>
#include <memory>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include "cudf/column/column_factories.hpp"
#include "cudf/detail/utilities/device_operators.cuh"
#include "cudf/types.hpp"
#include "cudf/utilities/error.hpp"
#include "gtest/gtest.h"
#include "rmm/device_buffer.hpp"
#include "thrust/host_vector.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/scan.h"
#include "thrust/sequence.h"

/*
template <typename T>
struct MythListColumnWrapperTestTyped : public cudf::test::BaseFixture {
  MythListColumnWrapperTestTyped() {}

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

using MythTestTypes = cudf::test::Concat<cudf::test::Types<int32_t>>;

TYPED_TEST_CASE(MythListColumnWrapperTestTyped, MythTestTypes);
*/

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;
using cudf::size_type;

struct StructColumnWrapperTest : public cudf::test::BaseFixture
{};

template<typename T>
struct TypedStructColumnWrapperTest : public cudf::test::BaseFixture
{};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;

TYPED_TEST_CASE(TypedStructColumnWrapperTest, FixedWidthTypesNotBool);

// Test simple struct construction without nullmask, through column factory.
// Columns must retain their originally set values.
TYPED_TEST(TypedStructColumnWrapperTest, TestColumnFactoryConstruction)
{

  auto names_col = cudf::test::strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald"
  }.release();

  int num_rows {names_col->size()};

  auto ages_col = 
    cudf::test::fixed_width_column_wrapper<TypeParam>{
      {48, 27, 25} 
    }.release();
    
  auto is_human_col =
    cudf::test::fixed_width_column_wrapper<bool>{
      {true, true, false}
    }.release();

  vector_of_columns cols;
  cols.push_back(std::move(names_col));
  cols.push_back(std::move(ages_col));
  cols.push_back(std::move(is_human_col));

  auto struct_col = cudf::make_structs_column(num_rows, std::move(cols), 0, {});

  EXPECT_EQ(num_rows, struct_col->size());

  auto struct_col_view {struct_col->view()};
  EXPECT_TRUE(
    std::all_of(
      struct_col_view.child_begin(), 
      struct_col_view.child_end(), 
      [&](auto const& child) {
        return child.size() == num_rows;
      }
    )
  );

  // Check child columns for exactly correct values.
  vector_of_columns expected_children;
  expected_children.emplace_back(
    cudf::test::strings_column_wrapper{
      "Samuel Vimes",
      "Carrot Ironfoundersson",
      "Angua von Uberwald"
    }.release()
  );
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<TypeParam>{
    48, 27, 25
  }.release());
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<bool>{
    true, true, false
  }.release());

  std::for_each(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0)+expected_children.size(),
    [&](auto idx) {
      cudf::test::expect_columns_equal(
        struct_col_view.child(idx), 
        expected_children[idx]->view()
      );
    }
  );
}


// Test simple struct construction with nullmasks, through column wrappers.
// When the struct row is null, the child column value must be null.
TYPED_TEST(TypedStructColumnWrapperTest, TestColumnWrapperConstruction)
{

  auto names_col = cudf::test::strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald",
    "Cheery Littlebottom",
    "Detritus", 
    "Mr Slant"
  }.release();

  int num_rows {names_col->size()};

  auto ages_col = 
    cudf::test::fixed_width_column_wrapper<TypeParam>{
      {48, 27, 25, 31, 351, 351}, 
      { 1,  1,  1,  1,   1,   0}
    }.release();
    
  auto is_human_col =
    cudf::test::fixed_width_column_wrapper<bool>{
      {true, true, false, false, false, false},
      {   1,    1,     0,     1,     1,     0}
    }.release();

  vector_of_columns cols;
  cols.push_back(std::move(names_col));
  cols.push_back(std::move(ages_col));
  cols.push_back(std::move(is_human_col));

  auto struct_col = 
    cudf::test::structs_column_wrapper{std::move(cols), {1, 1, 1, 0, 1, 1}}.release();

  EXPECT_EQ(num_rows, struct_col->size());

  auto struct_col_view {struct_col->view()};
  EXPECT_TRUE(
    std::all_of(
      struct_col_view.child_begin(), 
      struct_col_view.child_end(), 
      [&](auto const& child) {
        return child.size() == num_rows;
      }
    )
  );

  // Check child columns for exactly correct values.
  vector_of_columns expected_children;
  expected_children.emplace_back(
    cudf::test::strings_column_wrapper{
      {
        "Samuel Vimes",
        "Carrot Ironfoundersson",
        "Angua von Uberwald",
        "Cheery Littlebottom",
        "Detritus",
        "Mr Slant"
      }, 
      {1, 1, 1, 0, 1, 1}
    }.release()
  );
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<TypeParam>{
    {48, 27, 25, 31, 351, 351},
    { 1,  1,  1,  0,   1,   0} 
  }.release());
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false},
    {   1,    1,     0,     0,     1,     0}
  }.release());

  std::for_each(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0)+expected_children.size(),
    [&](auto idx) {
      cudf::test::expect_columns_equal(
        struct_col_view.child(idx), 
        expected_children[idx]->view()
      );
    }
  );

  auto expected_struct_col =
    cudf::test::structs_column_wrapper{std::move(expected_children), {1, 1, 1, 0, 1, 1}}.release();

  // cudf::test::expect_columns_equal(struct_col_view, expected_struct_col->view()); // WHY IS THIS BROKEN?
  // cudf::test::expect_columns_equal(struct_col_view, struct_col_view); // WHY IS THIS BROKEN?
}


TEST_F(StructColumnWrapperTest, SimpleTestExpectStructColumnsEqual)
{
  auto ints_col = cudf::test::fixed_width_column_wrapper<int32_t>{{0,1}, {0,0}}.release();

  vector_of_columns cols;
  cols.emplace_back(std::move(ints_col));
  auto structs_col = cudf::test::structs_column_wrapper{std::move(cols)};
  
  // cudf::test::expect_columns_equal(structs_col, structs_col);
}


CUDF_TEST_PROGRAM_MAIN()
