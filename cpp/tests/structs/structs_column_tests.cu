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
#include "rmm/device_buffer.hpp"
#include "thrust/host_vector.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/scan.h"
#include "thrust/sequence.h"

template <typename T>
struct MythListColumnWrapperTestTyped : public cudf::test::BaseFixture {
  MythListColumnWrapperTestTyped() {}

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

using MythTestTypes = cudf::test::Concat<cudf::test::Types<int32_t>>;

TYPED_TEST_CASE(MythListColumnWrapperTestTyped, MythTestTypes);

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;
using Table = cudf::table;
using size_type = cudf::size_type;

struct StructColumnWrapperTest : public cudf::test::BaseFixture
{};

TEST_F(StructColumnWrapperTest, SimpleStructTest)
{
  std::cout << "CALEB: Testing Struct Column!\n";

  auto names_col = cudf::test::strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald"
  }.release();

  int num_rows {names_col->size()};

  auto ages_col = 
    cudf::test::fixed_width_column_wrapper<int8_t>{
      {48, 23, 103}, 
      {1, 1, 0}
    }.release();

  vector_of_columns cols;
  cols.push_back(std::move(names_col));
  cols.push_back(std::move(ages_col));

  std::cout << "Num Rows: " << cols[0]->size() << std::endl;

  auto struct_col = cudf::make_structs_column(num_rows, std::move(cols), 0, {});
  std::cout << "Printing struct column: \n";
  cudf::test::print(struct_col->view());
  
}


TEST_F(StructColumnWrapperTest, SimpleStructColumnWrapperTest)
{
  int num_rows {3};

  auto names_col = cudf::test::strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald"
  };

  auto ages_col = 
    cudf::test::fixed_width_column_wrapper<int8_t>{
      {48, 23, 103}, 
      {1, 1, 0}
    };

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.emplace_back(names_col.release());
  cols.emplace_back(ages_col.release());

  cudf::test::structs_column_wrapper struct_col {
    std::move(cols)
  };

  cudf::test::print(struct_col.operator cudf::column_view());
}


TEST_F(StructColumnWrapperTest, SimpleStructColumnWrapperTest2)
{
  using namespace cudf::test;
  auto ref = structs_column_wrapper::ref;

  auto names_col = strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald"
  };

  auto ages_col = fixed_width_column_wrapper<int8_t>{
    {48, 23, 103}, 
    {1, 1, 0}
  };

  auto struct_col = structs_column_wrapper {
    {ref(names_col), ref(ages_col)}, {}
  }.release();

  auto struct_view {struct_col->view()};
  print(struct_view);

  expect_columns_equal(struct_view.child(0), strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald"
  });
  
  expect_columns_equal(struct_view.child(1), fixed_width_column_wrapper<int8_t>{
    {48, 23, 104}, {1, 1, 0}
  });

}

TEST_F(StructColumnWrapperTest, SimpleStructColumnWrapperTestWithValidity)
{
  using namespace cudf::test;
  auto ref = structs_column_wrapper::ref;

  auto names_col = strings_column_wrapper{
    {
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald"
    },
    {0, 1, 1}
  };

  auto ages_col = fixed_width_column_wrapper<int8_t>{
    {48, 23, 103}, {1, 1, 0}
  };

  auto struct_col = structs_column_wrapper {
    {ref(names_col), ref(ages_col)}, {1,0,1}
  }.release();

  auto struct_view {struct_col->view()};
  print(struct_view);

  /*
  expect_columns_equal(struct_view.child(0), strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald"
  });
  
  expect_columns_equal(struct_view.child(1), fixed_width_column_wrapper<int8_t>{
    {48, 23, 104}, {1, 1, 0}
  });
  */

}

CUDF_TEST_PROGRAM_MAIN()
