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

#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

struct ApplyBooleanMask : public cudf::test::BaseFixture {
};

TEST_F(ApplyBooleanMask, NonNullBooleanMask)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{
    {true, false, true, false, true, false}};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, true, true}, {1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 70, 2}, {1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{10, 70, 2}, {1, 0, 1}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::apply_boolean_mask(input, boolean_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, NullBooleanMask)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{{true, false, true, false, true, false},
                                                            {0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, true}, {0, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{70, 2}, {0, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{70, 2}, {0, 1}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::apply_boolean_mask(input, boolean_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, EmptyMask)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::apply_boolean_mask(input, boolean_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, WrongMaskType)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<int16_t> boolean_mask{
    {true, false, true, false, true, false}};

  EXPECT_THROW(cudf::apply_boolean_mask(input, boolean_mask), cudf::logic_error);
}

TEST_F(ApplyBooleanMask, MaskAndInputSizeMismatch)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{{true, false, true, false, true}};

  EXPECT_THROW(cudf::apply_boolean_mask(input, boolean_mask), cudf::logic_error);
}

TEST_F(ApplyBooleanMask, StringColumnTest)
{
  cudf::test::strings_column_wrapper col1{
    {"This", "is", "the", "a", "k12", "string", "table", "column"}, {1, 1, 1, 1, 1, 0, 1, 1}};
  cudf::table_view input{{col1}};
  cudf::test::fixed_width_column_wrapper<bool> boolean_mask{
    {true, true, true, true, false, true, false, true}, {1, 1, 0, 1, 1, 1, 1, 1}};
  cudf::test::strings_column_wrapper col1_expected{{"This", "is", "a", "string", "column"},
                                                   {1, 1, 1, 0, 1}};
  cudf::table_view expected{{col1_expected}};

  auto got = cudf::apply_boolean_mask(input, boolean_mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, withoutNullString)
{
  cudf::test::strings_column_wrapper col1({"d", "e", "a", "d", "k", "d", "l"});
  cudf::table_view cudf_table_in_view{{col1}};

  cudf::test::fixed_width_column_wrapper<bool> bool_filter{{1, 1, 0, 0, 1, 0, 0}};
  cudf::column_view bool_filter_col(bool_filter);

  std::unique_ptr<cudf::table> filteredTable =
    cudf::apply_boolean_mask(cudf_table_in_view, bool_filter_col);
  cudf::table_view tableView = filteredTable->view();

  cudf::test::strings_column_wrapper expect_col1({"d", "e", "k"});
  cudf::table_view expect_cudf_table_view{{expect_col1}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect_cudf_table_view, tableView);
}

TEST_F(ApplyBooleanMask, NoNullInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col(
    {9668, 9590, 9526, 9205, 9434, 9347, 9160, 9569, 9143, 9807, 9606, 9446, 9279, 9822, 9691});
  cudf::test::fixed_width_column_wrapper<bool> mask({false,
                                                     false,
                                                     true,
                                                     false,
                                                     false,
                                                     true,
                                                     false,
                                                     true,
                                                     false,
                                                     true,
                                                     false,
                                                     false,
                                                     true,
                                                     false,
                                                     true});
  cudf::table_view input({col});
  cudf::test::fixed_width_column_wrapper<int32_t> col_expected(
    {9526, 9347, 9569, 9807, 9279, 9691});
  cudf::table_view expected({col_expected});
  auto got = cudf::apply_boolean_mask(input, mask);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(ApplyBooleanMask, StructFiltering)
{
  using namespace cudf::test;

  auto int_member = fixed_width_column_wrapper<int32_t>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                        {1, 1, 1, 1, 0, 1, 1, 1, 1, 0}};

  auto struct_column = structs_column_wrapper{{int_member}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  auto filter_mask = fixed_width_column_wrapper<bool>{{1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};

  auto filtered_table = cudf::apply_boolean_mask(cudf::table_view({struct_column}), filter_mask);
  auto filtered_struct_column = filtered_table->get_column(0);

  // Compare against expected results.
  auto expected_int_member =
    fixed_width_column_wrapper<int32_t>{{-1, 1, 2, 3, -1}, {0, 1, 1, 1, 0}};

  auto expected_struct_column = structs_column_wrapper{{expected_int_member}, {1, 1, 1, 1, 0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(filtered_struct_column, expected_struct_column);
}

TEST_F(ApplyBooleanMask, ListOfStructsFiltering)
{
  using namespace cudf::test;

  auto key_member = fixed_width_column_wrapper<int32_t>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                        {1, 1, 1, 1, 0, 1, 1, 1, 1, 0}};

  auto value_member = fixed_width_column_wrapper<int32_t>{{0, 10, 20, 30, 40, 50, 60, 70, 80, 90},
                                                          {1, 1, 1, 1, 0, 1, 1, 1, 1, 0}};

  auto struct_column =
    structs_column_wrapper{{key_member, value_member}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  auto list_of_structs_column =
    cudf::make_lists_column(5,
                            fixed_width_column_wrapper<int32_t>{0, 2, 4, 6, 8, 10}.release(),
                            struct_column.release(),
                            cudf::UNKNOWN_NULL_COUNT,
                            {});

  auto filter_mask = fixed_width_column_wrapper<bool>{{1, 0, 1, 0, 1}};

  auto filtered_table =
    cudf::apply_boolean_mask(cudf::table_view({list_of_structs_column->view()}), filter_mask);
  auto filtered_list_column = filtered_table->get_column(0);

  cudf::test::print(filtered_list_column.view());

  // Compare against expected values.
  auto expected_key_column =
    fixed_width_column_wrapper<int32_t>{{0, 1, 4, 5, 8, 9}, {0, 1, 0, 0, 1, 0}};
  auto expected_value_column =
    fixed_width_column_wrapper<int32_t>{{0, 10, 40, 50, 80, 90}, {0, 1, 0, 0, 1, 0}};

  auto expected_struct_column =
    structs_column_wrapper{{expected_key_column, expected_value_column}, {0, 1, 1, 0, 1, 1}};

  auto expected_list_of_structs_column =
    cudf::make_lists_column(3,
                            fixed_width_column_wrapper<int32_t>{0, 2, 4, 6}.release(),
                            expected_struct_column.release(),
                            cudf::UNKNOWN_NULL_COUNT,
                            {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(filtered_list_column,
                                      expected_list_of_structs_column->view());
}

CUDF_TEST_PROGRAM_MAIN()
