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
 *
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/contains.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

namespace cudf {
namespace test {

struct IndexOfTest : public BaseFixture {
};

using IndexOfTestTypes = Concat<IntegralTypesNotBool, FloatingPointTypes, ChronoTypes>;

template <typename T>
struct IndexOfTypedTest : public IndexOfTest {
};

TYPED_TEST_SUITE(IndexOfTypedTest, IndexOfTestTypes);

namespace {

auto constexpr absent     = size_type{-1};
auto constexpr FIND_FIRST = lists::duplicate_find_option::FIND_FIRST;
auto constexpr FIND_LAST  = lists::duplicate_find_option::FIND_LAST;

template <typename T, std::enable_if_t<cudf::is_numeric<T>(), void>* = nullptr>
auto create_scalar_search_key(T const& value)
{
  auto search_key = make_numeric_scalar(data_type{type_to_id<T>()});
  search_key->set_valid_async(true);
  static_cast<scalar_type_t<T>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T, std::enable_if_t<std::is_same_v<T, std::string>, void>* = nullptr>
auto create_scalar_search_key(std::string const& value)
{
  return make_string_scalar(value);
}

template <typename T, std::enable_if_t<cudf::is_timestamp<T>(), void>* = nullptr>
auto create_scalar_search_key(typename T::rep const& value)
{
  auto search_key = make_timestamp_scalar(data_type{type_to_id<T>()});
  search_key->set_valid_async(true);
  static_cast<scalar_type_t<typename T::rep>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_duration<T>(), void>* = nullptr>
auto create_scalar_search_key(typename T::rep const& value)
{
  auto search_key = make_duration_scalar(data_type{type_to_id<T>()});
  search_key->set_valid_async(true);
  static_cast<scalar_type_t<typename T::rep>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_numeric<T>(), void>* = nullptr>
auto create_null_search_key()
{
  auto search_key = make_numeric_scalar(data_type{type_to_id<T>()});
  search_key->set_valid_async(false);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_timestamp<T>(), void>* = nullptr>
auto create_null_search_key()
{
  auto search_key = make_timestamp_scalar(data_type{type_to_id<T>()});
  search_key->set_valid_async(false);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_duration<T>(), void>* = nullptr>
auto create_null_search_key()
{
  auto search_key = make_duration_scalar(data_type{type_to_id<T>()});
  search_key->set_valid_async(false);
  return search_key;
}

}  // namespace

using iterators::null_at;
using iterators::nulls_at;
using indices = fixed_width_column_wrapper<size_type>;

TYPED_TEST(IndexOfTypedTest, ScalarKeyWithNoNulls)
{
  using T = TypeParam;

  auto search_space = lists_column_view{lists_column_wrapper<T, int32_t>{
    {0, 1, 2, 1},
    {3, 4, 5},
    {6, 7, 8},
    {9, 0, 1, 3, 1},
    {2, 3, 4},
    {5, 6, 7},
    {8, 9, 0},
    {},
    {1, 2, 1, 3},
    {}}};
  auto search_key_one = create_scalar_search_key<T>(1);

  {
    // FIND_FIRST
    auto actual_result   = lists::index_of(search_space, *search_key_one, FIND_FIRST);
    auto expected_result = indices{1, absent, absent, 2, absent, absent, absent, absent, 0, absent};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
  }
  {
    // FIND_LAST
    auto actual_result   = lists::index_of(search_space, *search_key_one, FIND_LAST);
    auto expected_result = indices{3, absent, absent, 4, absent, absent, absent, absent, 2, absent};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
  }
}

TYPED_TEST(IndexOfTypedTest, ScalarKeyWithNullLists)
{
  // Test List columns that have NULL list rows.
  using T = TypeParam;

  auto search_space = lists_column_view{lists_column_wrapper<T, int32_t>{
    {{0, 1, 2, 1},
     {3, 4, 5},
     {6, 7, 8},
     {},
     {9, 0, 1, 3, 1},
     {2, 3, 4},
     {5, 6, 7},
     {8, 9, 0},
     {},
     {1, 2, 2, 3},
     {}}, 
     nulls_at({3, 10})
    }};

  {
    // FIND_FIRST
    auto search_key_one  = create_scalar_search_key<T>(1);
    auto actual_result   = lists::index_of(search_space, *search_key_one, FIND_FIRST);
    auto expected_result = indices{{1, absent, absent, 0, 2, absent, absent, absent, absent, 0, 0}, nulls_at({3,10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
  }
  {
    // FIND_LAST
    auto search_key_one  = create_scalar_search_key<T>(1);
    auto actual_result   = lists::index_of(search_space, *search_key_one, FIND_LAST);
    auto expected_result = indices{{3, absent, absent, 0, 4, absent, absent, absent, absent, 0, 0}, nulls_at({3,10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
  }
}

TYPED_TEST(IndexOfTypedTest, SlicedLists)
{
  // Test sliced List columns.
  using namespace cudf;
  using T     = TypeParam;

  auto search_space = lists_column_wrapper<T, int32_t>{
    {{0, 1, 2, 1},
     {3, 4, 5},
     {6, 7, 8},
     {},
     {9, 0, 1, 3, 1},
     {2, 3, 4},
     {5, 6, 7},
     {8, 9, 0},
     {},
     {1, 2, 1, 3},
     {}}, nulls_at({3, 10})
    };

  {
    // First Slice.
    auto sliced_column_1 = cudf::detail::slice(search_space, {1, 8}).front();
    auto search_key_one  = create_scalar_search_key<T>(1);
    {
      // FIND_FIRST
      auto result          = lists::index_of(sliced_column_1, *search_key_one, FIND_FIRST);
      auto expected_result = indices{{absent, absent, 0, 2, absent, absent, absent}, null_at(2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
    {
      // FIND_LAST
      auto result          = lists::index_of(sliced_column_1, *search_key_one, FIND_LAST);
      auto expected_result = indices{{absent, absent, 0, 4, absent, absent, absent}, null_at(2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
  }

  {
    // Second Slice.
    auto sliced_column_2 = cudf::detail::slice(search_space, {3, 10}).front();
    auto search_key_one  = create_scalar_search_key<T>(1);
    {
      // FIND_FIRST
      auto result = lists::index_of(sliced_column_2, *search_key_one, FIND_FIRST);
      auto expected_result = indices{{0, 2, absent, absent, absent, absent, 0}, null_at(0)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
    {
      // FIND_LAST
      auto result = lists::index_of(sliced_column_2, *search_key_one, FIND_LAST);
      auto expected_result = indices{{0, 4, absent, absent, absent, absent, 2}, null_at(0)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
  }
}

TYPED_TEST(IndexOfTypedTest, ScalarKeyNonNullListsWithNullValues)
{
  // Test List columns that have no NULL list rows, but NULL elements in some list rows.
  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto search_space =
    make_lists_column(8,
                      fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
                      numerals.release(),
                      0,
                      {});

  auto search_key_one = create_scalar_search_key<T>(1);
  auto actual_result  = lists::contains(search_space->view(), *search_key_one);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 1, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(IndexOfTypedTest, ListContainsScalarWithNullsInLists)
{
  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_key_one = create_scalar_search_key<T>(1);
  auto actual_result  = lists::contains(search_space->view(), *search_key_one);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(IndexOfTest, BoolListContainsScalarWithNullsInLists)
{
  using T = bool;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_key_one = create_scalar_search_key<T>(1);
  auto actual_result  = lists::contains(search_space->view(), *search_key_one);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 1, 0, 0, 1, 0, 1}, {0, 1, 1, 1, 0, 1, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(IndexOfTest, StringListContainsScalarWithNullsInLists)
{
  using T = std::string;

  auto strings = strings_column_wrapper{
    {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "1", "2", "3", "4"},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
    strings.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_key_one = create_scalar_search_key<T>("1");
  auto actual_result  = lists::contains(search_space->view(), *search_key_one);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(IndexOfTypedTest, ContainsScalarNullSearchKey)
{
  using T = TypeParam;

  auto search_space = lists_column_wrapper<T, int32_t>{
    {{0, 1, 2},
     {3, 4, 5},
     {6, 7, 8},
     {},
     {9, 0, 1},
     {2, 3, 4},
     {5, 6, 7},
     {8, 9, 0},
     {},
     {1, 2, 3},
     {}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return (i != 3) && (i != 10);
    })}.release();

  auto search_key_null = create_null_search_key<T>();
  auto actual_result   = lists::contains(search_space->view(), *search_key_null);
  auto expected_result = fixed_width_column_wrapper<bool>{
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return false; })};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(IndexOfTest, ScalarTypeRelatedExceptions)
{
  {
    // Nested types unsupported.
    auto list_of_lists = lists_column_wrapper<int32_t>{
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3},
       {4, 5, 6}}}.release();
    auto skey = create_scalar_search_key<int32_t>(10);
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_lists->view(), *skey),
                              "Nested types not supported in lists::contains()");
  }

  {
    // Search key must match list elements in type.
    auto list_of_ints =
      lists_column_wrapper<int32_t>{
        {0, 1, 2},
        {3, 4, 5},
      }
        .release();
    auto skey = create_scalar_search_key<std::string>("Hello, World!");
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_ints->view(), *skey),
                              "Type/Scale of search key does not match list column element type.");
  }
}

template <typename T>
struct VectorIndexOfTypedTest : public IndexOfTest {
};

using VectorIndexOfTestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_CASE(VectorIndexOfTypedTest, VectorIndexOfTestTypes);

TYPED_TEST(VectorIndexOfTypedTest, ListContainsVectorWithNoNulls)
{
  using T = TypeParam;

  auto search_space = lists_column_wrapper<T, int32_t>{
    {0, 1, 2},
    {3, 4, 5},
    {6, 7, 8},
    {9, 0, 1},
    {2, 3, 4},
    {5, 6, 7},
    {8, 9, 0},
    {},
    {1, 2, 3},
    {}}.release();

  auto search_key      = fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
  auto actual_result   = lists::contains(search_space->view(), search_key);
  auto expected_result = fixed_width_column_wrapper<bool>{1, 0, 0, 1, 1, 0, 0, 0, 1, 0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(VectorIndexOfTypedTest, ListContainsVectorWithNullLists)
{
  // Test List columns that have NULL list rows.

  using T = TypeParam;

  auto search_space = lists_column_wrapper<T, int32_t>{
    {{0, 1, 2},
     {3, 4, 5},
     {6, 7, 8},
     {},
     {9, 0, 1},
     {2, 3, 4},
     {5, 6, 7},
     {8, 9, 0},
     {},
     {1, 2, 3},
     {}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return (i != 3) && (i != 10);
    })}.release();

  auto search_keys   = fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2};
  auto actual_result = lists::contains(search_space->view(), search_keys);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0},
                                     cudf::detail::make_counting_transform_iterator(
                                       0, [](auto i) { return (i != 3) && (i != 10); })};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(VectorIndexOfTypedTest, ListContainsVectorNonNullListsWithNullValues)
{
  // Test List columns that have no NULL list rows, but NULL elements in some list rows.
  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto search_space =
    make_lists_column(8,
                      fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
                      numerals.release(),
                      0,
                      {});

  auto search_keys   = fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 3};
  auto actual_result = lists::contains(search_space->view(), search_keys);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 1, 1}, {0, 1, 0, 1, 1, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(VectorIndexOfTypedTest, ListContainsVectorWithNullsInLists)
{
  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_keys   = fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 3};
  auto actual_result = lists::contains(search_space->view(), search_keys);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 1, 1}, {0, 1, 0, 1, 0, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(VectorIndexOfTypedTest, ListContainsVectorWithNullsInListsAndInSearchKeys)
{
  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_keys = fixed_width_column_wrapper<T, int32_t>{
    {1, 2, 3, 1, 2, 3, 1, 3},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 6; })};

  auto actual_result = lists::contains(search_space->view(), search_keys);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 0, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(IndexOfTest, BoolListContainsVectorWithNullsInListsAndInSearchKeys)
{
  using T = bool;

  auto numerals = fixed_width_column_wrapper<T, int32_t>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_keys = fixed_width_column_wrapper<T, int32_t>{
    {0, 1, 0, 1, 0, 0, 1, 1},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 6; })};

  auto actual_result = lists::contains(search_space->view(), search_keys);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 0, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(IndexOfTest, StringListContainsVectorWithNullsInListsAndInSearchKeys)
{
  auto numerals = strings_column_wrapper{
    {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "1", "2", "3", "4"},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_keys = strings_column_wrapper{
    {"1", "2", "3", "1", "2", "3", "1", "3"},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 6; })};

  auto actual_result = lists::contains(search_space->view(), search_keys);
  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 0, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(IndexOfTest, VectorTypeRelatedExceptions)
{
  {
    // Nested types unsupported.
    auto list_of_lists = lists_column_wrapper<int32_t>{
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3},
       {4, 5, 6}}}.release();
    auto skey = fixed_width_column_wrapper<int32_t>{0, 1, 2};
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_lists->view(), skey),
                              "Nested types not supported in lists::contains()");
  }

  {
    // Search key must match list elements in type.
    auto list_of_ints =
      lists_column_wrapper<int32_t>{
        {0, 1, 2},
        {3, 4, 5},
      }
        .release();
    auto skey = strings_column_wrapper{"Hello", "World"};
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_ints->view(), skey),
                              "Type/Scale of search key does not match list column element type.");
  }

  {
    // Search key column size must match lists column size.
    auto list_of_ints = lists_column_wrapper<int32_t>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}.release();

    auto skey = fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_ints->view(), skey),
                              "Number of search keys must match list column size.");
  }
}

template <typename T>
struct IndexOfNaNsTypedTest : public IndexOfTest {
};

TYPED_TEST_CASE(IndexOfNaNsTypedTest, FloatingPointTypes);

namespace {
template <typename T>
T get_nan(const char* nan_contents)
{
  return std::nan(nan_contents);
}

template <>
float get_nan<float>(const char* nan_contents)
{
  return std::nanf(nan_contents);
}
}  // namespace

TYPED_TEST(IndexOfNaNsTypedTest, ListWithNaNsContainsScalar)
{
  using T = TypeParam;

  auto nan_1 = get_nan<T>("1");
  auto nan_2 = get_nan<T>("2");
  auto nan_3 = get_nan<T>("3");

  auto search_space = lists_column_wrapper<T>{
    {0.0, 1.0, 2.0},
    {3, 4, 5},
    {6, 7, 8},
    {9, 0, 1},
    {nan_1, 3.0, 4.0},
    {5, 6, 7},
    {8, nan_2, 0},
    {},
    {1, 2, 3},
    {}}.release();

  auto search_key_nan  = create_scalar_search_key<T>(nan_3);
  auto actual_result   = lists::contains(search_space->view(), *search_key_nan);
  auto expected_result = fixed_width_column_wrapper<bool>{0, 0, 0, 0, 1, 0, 1, 0, 0, 0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(IndexOfNaNsTypedTest, ListWithNaNsContainsVector)
{
  // Test that different bit representations of NaN values
  // are recognized as NaN.
  // Also checks that a null handling is not broken by the
  // presence of NaN values:
  //   1. If the search key is null, null is still returned.
  //   2. If the list contains a null, and the non-null search
  //      key is not found, null is returned.
  using T = TypeParam;

  auto nan_1 = get_nan<T>("1");
  auto nan_2 = get_nan<T>("2");
  auto nan_3 = get_nan<T>("3");

  auto null_at_index_2 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 2; });

  auto search_space = lists_column_wrapper<T>{
    {0.0, 1.0, 2.0},
    {{3, 4, 5}, null_at_index_2},  // i.e. {3, 4, ∅}.
    {6, 7, 8},
    {9, 0, 1},
    {nan_1, 3.0, 4.0},
    {5, 6, 7},
    {8, nan_2, 0},
    {},
    {1, 2, 3},
    {}}.release();

  auto search_key_values = std::vector<T>{1.0, 2.0, 3.0, nan_3, nan_3, nan_3, 0.0, nan_3, 2.0, 0.0};

  {
    // With nulls in the search key rows. (At index 2.)
    auto search_keys =
      fixed_width_column_wrapper<T>{
        search_key_values.begin(), search_key_values.end(), null_at_index_2}
        .release();

    auto actual_result = lists::contains(search_space->view(), search_keys->view());
    auto null_at_index_1_and_2 =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1 && i != 2; });

    auto expected_result =
      fixed_width_column_wrapper<bool>{{1, 0, 0, 0, 1, 0, 1, 0, 1, 0}, null_at_index_1_and_2};

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
  }

  {
    // No nulls in the search key rows.
    auto search_keys =
      fixed_width_column_wrapper<T>(search_key_values.begin(), search_key_values.end()).release();

    auto actual_result = lists::contains(search_space->view(), search_keys->view());
    auto null_at_index_1 =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; });

    auto expected_result =
      fixed_width_column_wrapper<bool>{{1, 0, 0, 0, 1, 0, 1, 0, 1, 0}, null_at_index_1};

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
  }
}

template <typename T>
struct IndexOfDecimalsTypedTest : public IndexOfTest {
};

TYPED_TEST_CASE(IndexOfDecimalsTypedTest, FixedPointTypes);

TYPED_TEST(IndexOfDecimalsTypedTest, ListContainsScalar)
{
  using T = TypeParam;

  auto const values = std::vector<typename T::rep>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
                                                   2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3};
  auto decimals     = fixed_point_column_wrapper<typename T::rep>{
    values.begin(), values.end(), numeric::scale_type{0}};

  auto list_offsets = fixed_width_column_wrapper<size_type>{0, 3, 6, 9, 12, 15, 18, 21, 21, 24, 24};

  auto const search_space =
    make_lists_column(10, list_offsets.release(), decimals.release(), 0, {});

  auto search_key_one  = make_fixed_point_scalar<T>(typename T::rep{1}, numeric::scale_type{0});
  auto actual_result   = lists::contains(search_space->view(), *search_key_one);
  auto expected_result = fixed_width_column_wrapper<bool>{1, 0, 0, 1, 0, 0, 0, 0, 1, 0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(IndexOfDecimalsTypedTest, ListContainsVector)
{
  using T = TypeParam;

  auto const values = std::vector<typename T::rep>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
                                                   2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3};
  auto decimals     = fixed_point_column_wrapper<typename T::rep>{
    values.begin(), values.end(), numeric::scale_type{0}};

  auto list_offsets = fixed_width_column_wrapper<size_type>{0, 3, 6, 9, 12, 15, 18, 21, 21, 24, 24};

  auto const search_space =
    make_lists_column(10, list_offsets.release(), decimals.release(), 0, {});

  auto search_key = fixed_point_column_wrapper<typename T::rep>{
    {1, 2, 3, 1, 2, 3, 1, 2, 3, 1},
    numeric::scale_type{
      0}}.release();

  auto actual_result   = lists::contains(search_space->view(), search_key->view());
  auto expected_result = fixed_width_column_wrapper<bool>{1, 0, 0, 1, 1, 0, 0, 0, 1, 0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

}  // namespace test

}  // namespace cudf
