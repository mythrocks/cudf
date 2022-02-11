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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/row_conversion.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <gtest/gtest.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <iterator>
#include <limits>

struct ColumnToRowTests : public cudf::test::BaseFixture {
};
struct RowToColumnTests : public cudf::test::BaseFixture {
};

TEST_F(ColumnToRowTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, SimpleString)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1, 0, -1});
  cudf::test::strings_column_wrapper b(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), 5);
  cudf::test::print(*new_rows[0]);
}

TEST_F(ColumnToRowTests, DoubleString)
{
  cudf::test::strings_column_wrapper a(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::test::fixed_width_column_wrapper<int32_t> b({0, 1, 2, 3, 4});
  cudf::test::strings_column_wrapper c({"world",
                                        "hello",
                                        "this string isn't as long",
                                        "this one isn't so short though when you think about it",
                                        "dlrow"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});

  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), 5);
  cudf::test::print(*new_rows[0]);
}

void print_rows(cudf::lists_column_view const& lcv, std::vector<cudf::data_type> const& schema)
{
  for (auto s : schema) {
    printf("%c", s.id() == cudf::type_id::STRING ? 'S' : 'I');
  }
  printf("\n");

  auto offsets = lcv.offsets();
  auto data    = lcv.child();
  //  auto zero_iter = thrust::make_counting_iterator(0);
  auto const num_rows = lcv.size();

  auto h_offsets = cudf::test::to_host<cudf::size_type>(offsets);
  auto h_data    = cudf::test::to_host<int8_t>(data);

  auto print_row = [&](int r) {
    printf("row %d(%d-%d): ", r, h_offsets.first[r], h_offsets.first[r + 1]);
    auto this_schema      = schema.begin();
    int next_grouping     = this_schema->id() == cudf::type_id::INT32 ? 4 : 8;
    int idx               = h_offsets.first[r];
    bool handled_validity = false;
    while (idx < h_offsets.first[r + 1]) {
      if (this_schema == schema.end() && !handled_validity) {
        int validity_bytes =
          cudf::util::div_rounding_up_unsafe(std::distance(schema.begin(), schema.end()), 8);
        printf("V:");
        for (int i = 0; i < validity_bytes; ++i) {
          printf("%02x", h_data.first[idx++]);
        }
        printf(" ");
        handled_validity = true;
      } else if (this_schema == schema.end() && handled_validity) {
        // just print the data
        int8_t byte = h_data.first[idx];
        idx++;
        if (byte >= 'a' && byte <= 'z') {
          printf("%c", byte);
        } else {
          printf("%02x", byte);
        }
      } else if (this_schema->id() == cudf::type_id::INT32) {
        int32_t fb = *reinterpret_cast<int32_t*>(&h_data.first[idx]);
        printf("%08x ", fb);
        idx += 4;
        this_schema++;
      } else if (this_schema->id() == cudf::type_id::STRING) {
        int32_t of = *reinterpret_cast<int32_t*>(&h_data.first[idx]);
        idx += 4;
        printf("%08x-", of);
        int32_t ln = *reinterpret_cast<int32_t*>(&h_data.first[idx]);
        idx += 4;
        printf("%08x ", ln);
        this_schema++;
      }
    }
    printf("\n");
  };

  if (num_rows > 20) {
    for (auto r = 0; r < 10; ++r) {
      print_row(r);
    }
    for (auto r = num_rows - 10; r < num_rows; ++r) {
      print_row(r);
    }
  } else {
    for (auto r = 0; r < num_rows; ++r) {
      print_row(r);
    }
  }
}

TEST_F(ColumnToRowTests, BigStrings)
{
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working"};
  auto num_generator =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 50;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), num_rows);

  // pick it apart
  print_rows(new_rows[0]->view(), schema);
  //  cudf::test::print(*new_rows[0]);
}

TEST_F(ColumnToRowTests, ManyStrings)
{
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working",
    "some strings",
    "are split into multiple strings",
    "some strings have all their data",
    "lots of choices of strings and sizes is sure to test the offset calculation code to ensure "
    "that even a really long string ends up in the correct spot for the final destination allowing "
    "for even crazy run-on sentences to be inserted into the data"};
  auto num_generator =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 1000000;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), num_rows);

  // pick it apart
  print_rows(new_rows[0]->view(), schema);
  //  cudf::test::print(*new_rows[0]);
}

TEST_F(ColumnToRowTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, SingleByteCol)
{
  cudf::test::fixed_width_column_wrapper<int8_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT8}};

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Tall)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int8_t { return rand(); });
  cudf::test::fixed_width_column_wrapper<int8_t> a(r, r + (size_t)4000000);
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT8}};

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Wide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({rand()}));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, SingleByteWide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int8_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int8_t>({rand()}));
    views.push_back(cols.back());

    schema.push_back(cudf::data_type{cudf::type_id::INT8});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Non2Power)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  constexpr auto num_rows = 6 * 1024 + 557;
  for (int i = 0; i < 131; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Big)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 28; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Bigger)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Biggest)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 2 million rows
  constexpr auto num_rows = 2 * 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = cudf::convert_to_rows(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Seeded)
{
  cudf::test::fixed_width_column_wrapper<int8_t> a(
    {0x00, 0x28, 0xfc, 0x34, 0xce, 0x84, 0x9a, 0x30, 0x16, 0x4e, 0xd7, 0xc8},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT8}};

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }

  cudf::test::fixed_width_column_wrapper<int8_t> b(
    {0xc9, 0x9a, 0xee, 0x71, 0x85, 0xa2, 0xa9, 0x4a, 0x95, 0x81, 0x00, 0x15});

  cudf::table_view inb(std::vector<cudf::column_view>{b});
  std::vector<cudf::data_type> schemab = {cudf::data_type{cudf::type_id::INT8}};

  auto old_rowsb = cudf::convert_to_rows_fixed_width_optimized(inb);
  auto new_rowsb = cudf::convert_to_rows(inb);
  for (uint i = 0; i < old_rowsb.size(); ++i) {
    auto old_tblb =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rowsb[i]), schema);
    auto new_tblb = cudf::convert_from_rows(cudf::lists_column_view(*new_rowsb[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tblb, *new_tblb);
  }
}

TEST_F(RowToColumnTests, SingleByteCol)
{
  cudf::test::fixed_width_column_wrapper<int8_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT8}};

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Tall)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int8_t { return rand(); });
  cudf::test::fixed_width_column_wrapper<int8_t> a(r, r + (size_t)4000000);
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Wide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({i}));  // rand()}));
    views.push_back(cols.back());
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, SingleByteWide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int8_t>> cols;
  std::vector<cudf::column_view> views;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int8_t>({rand()}));
    views.push_back(cols.back());
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, AllTypes)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT64},
                                      cudf::data_type{cudf::type_id::FLOAT64},
                                      cudf::data_type{cudf::type_id::INT8},
                                      cudf::data_type{cudf::type_id::BOOL8},
                                      cudf::data_type{cudf::type_id::FLOAT32},
                                      cudf::data_type{cudf::type_id::INT8},
                                      cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::INT64}};

  cudf::test::fixed_width_column_wrapper<int64_t> c0({3, 9, 4, 2, 20, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<double> c1({5.0, 9.5, 0.9, 7.23, 2.8, 0.0},
                                                    {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int8_t> c2({5, 1, 0, 2, 7, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> c3({true, false, false, true, false, false},
                                                  {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<float> c4({1.0f, 3.5f, 5.9f, 7.1f, 9.8f, 0.0f},
                                                   {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int8_t> c5({2, 3, 4, 5, 9, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_point_column_wrapper<int32_t> c6(
    {-300, 500, 950, 90, 723, 0}, {1, 1, 1, 1, 1, 1, 1, 0}, numeric::scale_type{-2});
  cudf::test::fixed_point_column_wrapper<int64_t> c7(
    {-80, 30, 90, 20, 200, 0}, {1, 1, 1, 1, 1, 1, 0}, numeric::scale_type{-1});

  cudf::table_view in({c0, c1, c2, c3, c4, c5, c6, c7});

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, AllTypesLarge)
{
  std::vector<cudf::column> cols;
  std::vector<cudf::data_type> schema{};

  // 10 columns of each type with 1024 entries
  constexpr int num_rows{1024};

  std::default_random_engine re;
  std::uniform_real_distribution<double> rand_double(std::numeric_limits<double>::min(),
                                                     std::numeric_limits<double>::max());
  std::uniform_int_distribution<int64_t> rand_int64(std::numeric_limits<int64_t>::min(),
                                                    std::numeric_limits<int64_t>::max());
  auto r = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) -> int64_t { return rand_int64(re); });
  auto d = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) -> double { return rand_double(re); });

  auto all_valid  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 1; });
  auto none_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 0; });
  auto most_valid = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return rand() % 2 == 0 ? 0 : 1; });
  auto few_valid = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return rand() % 13 == 0 ? 1 : 0; });

  for (int i = 0; i < 10; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<int8_t>(r, r + num_rows, all_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::INT8});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<int16_t>(r, r + num_rows, few_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::INT16});
  }

  for (int i = 0; i < 10; ++i) {
    if (i < 5) {
      cols.push_back(*cudf::test::fixed_width_column_wrapper<int32_t>(r, r + num_rows, few_valid)
                        .release()
                        .release());
    } else {
      cols.push_back(*cudf::test::fixed_width_column_wrapper<int32_t>(r, r + num_rows, none_valid)
                        .release()
                        .release());
    }
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<float>(d, d + num_rows, most_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::FLOAT32});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<double>(d, d + num_rows, most_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::FLOAT64});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<bool>(r, r + num_rows, few_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::BOOL8});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
         r, r + num_rows, all_valid)
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>(
         r, r + num_rows, most_valid)
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(*cudf::test::fixed_point_column_wrapper<int32_t>(
                      r, r + num_rows, all_valid, numeric::scale_type{-2})
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::DECIMAL32});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(*cudf::test::fixed_point_column_wrapper<int64_t>(
                      r, r + num_rows, most_valid, numeric::scale_type{-1})
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::DECIMAL64});
  }

  std::vector<cudf::column_view> views(cols.begin(), cols.end());
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Non2Power)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  constexpr auto num_rows = 6 * 1024 + 557;
  for (int i = 0; i < 131; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Big)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 28; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Bigger)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Biggest)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 5 * 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

#define XSTR(x) STR(x)
#define STR(x)  #x

#ifdef __CUDA_ARCH__
#pragma message "__CUDA_ARCH__ defined as: " XSTR(__CUDA_ARCH__)
#else
#pragma message "__CUDA_ARCH__ undefined!"
#endif

#if __CUDA_ARCH__ >= 800
#define ASYNC_MEMCPY_SUPPORTED
#endif

void func_with_device_specific_lambda()
{
  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(1),
                   [] __device__(auto i) {
#ifdef ASYNC_MEMCPY_SUPPORTED
                     printf("Func:\t Ampere!\n");
#else
      printf("Func:\t Non-Ampere! (Turing!)\n");
#endif
                   });
}

void func_with_outer_cuda_arch_check()
{
#ifdef ASYNC_MEMCPY_SUPPORTED
  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(1),
                   [] __device__(auto i) { printf("OUTER:\t Ampere!\n"); });
#else
  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(1),
                   [] __device__(auto i) { printf("OUTER:\t Non-Ampere! (Turing!)\n"); });
#endif
}

__device__ void device_print_arch()
{
#ifdef ASYNC_MEMCPY_SUPPORTED
  printf("Device:\t Ampere!\n");
#else
  printf("Device:\t Non-Ampere! (Turing!)\n");
#endif
}

void hop_to_device()
{
  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(1),
                   [=] __device__(auto i) { device_print_arch(); });
}

__global__ void kernel()
{
#ifdef ASYNC_MEMCPY_SUPPORTED
  printf("Kernel:\t Ampere!\n");
#else
  printf("Kernel:\t Non-Ampere! (Turing!)\n");
#endif
}

void host()
{
#ifdef ASYNC_MEMCPY_SUPPORTED
  printf("HOST:\t Ampere!\n");
#else
  printf("HOST:\t Non-Ampere! (Turing!)\n");
#endif
}

TEST_F(RowToColumnTests, MythConditional)
{
  func_with_device_specific_lambda();
  kernel<<<1, 1>>>();
  hop_to_device();
  func_with_outer_cuda_arch_check();
  host();
  cudaDeviceSynchronize();
}
