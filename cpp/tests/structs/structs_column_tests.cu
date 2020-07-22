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

void myth() 
{
  auto prev_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>{ 1, 2, 2, 2, 2, 1, 2, 2, 2 }.release();
  auto foll_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>{ 1, 1, 1, 1, 0, 1, 1, 1, 0 }.release();

  auto prev = prev_col->view();
  auto foll = foll_col->view();

  EXPECT_EQ(prev.size(), foll.size());

  const int INPUT_SIZE {prev.size()};

  auto gather_map_size {
    thrust::reduce(thrust::device, prev.begin<size_type>(), prev.end<size_type>(), 0, thrust::plus<size_type>{})
    +
    thrust::reduce(thrust::device, foll.begin<size_type>(), foll.end<size_type>(), 0, thrust::plus<size_type>{})
  };

  std::cout << "gather_map_size == " << gather_map_size << std::endl;

  auto offsets_col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32}, INPUT_SIZE+1);
  auto offsets{offsets_col->mutable_view()};

  // Add prev and foll.
  thrust::transform(
    thrust::device,
    prev.begin<size_type>(), prev.end<size_type>(),
    foll.begin<size_type>(),
    offsets.begin<size_type>(),
    thrust::plus<size_type>()
  );

  std::cout << "Sums: " << std::endl; 
  cudf::test::print(offsets);

  // Cumulative sums, for start offsets, via exclusive_scan.
  thrust::exclusive_scan(thrust::device, offsets.begin<size_type>(), offsets.end<size_type>()+1, offsets.begin<size_type>());

  auto p_gather_map = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, gather_map_size);
  auto gather_map = p_gather_map->mutable_view();


  std::cout << "Cumulative sums: " << std::endl;
  cudf::test::print(offsets);

  auto write_gather_map_entries = [
      gather_map = gather_map.data<size_type>(),
      offsets = offsets.data<size_type>(),
      prev = prev.data<size_type>(),
      foll = foll.data<size_type>()
    ] __device__ (auto i)
  {
    auto out_start = gather_map + offsets[i];
    auto num_entries = prev[i] + foll[i];
    thrust::sequence(thrust::seq, out_start, out_start+num_entries, i-prev[i]+1);
  };

  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator(0), 
    thrust::make_counting_iterator(INPUT_SIZE), 
    write_gather_map_entries
  );

  std::cout << "Gather Map: " << std::endl;
  cudf::test::print(gather_map);

  // Gather map is now in place.

  // auto input_col = cudf::test::fixed_width_column_wrapper<int8_t> {
    // 'A', 'B', 'C', 'D', 'E', 'W', 'X', 'Y', 'Z'
  // };

  auto input_col = cudf::test::lists_column_wrapper<int8_t> {
    {11},
    {22, 22},
    {33},
    {44, 44},
    {55},
    {66, 66}, 
    {77},
    {88, 88},
    {99}
  };

  vector_of_columns cols;
  cols.push_back(input_col.release());

  Table table{std::move(cols)};

  std::cout << "Printing input table: " << std::endl;
  cudf::test::print(table.view().column(0));

  auto gathered = cudf::gather(table.view(), gather_map)->release();
  std::cout << "Results from cudf::gather() (size == " << gathered[0]->size() << ")\n";
  cudf::test::print(*gathered[0]);
  std::cout << "Using offsets: (Size == " << offsets_col->size() << ") \n";
  cudf::test::print(*offsets_col);

  auto lists_result = cudf::make_lists_column(INPUT_SIZE, std::move(offsets_col), std::move(gathered[0]), 0, rmm::device_buffer{0});
  std::cout << "Lists result: \n";
  cudf::test::print(*lists_result);
}

TYPED_TEST(MythListColumnWrapperTestTyped, MythExperimentGatherMap)
{
  myth();
}

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

namespace cudf
{
  namespace test 
  {
    class structs_column_wrapper : public detail::column_wrapper
    {
      public:

        structs_column_wrapper(std::vector<std::unique_ptr<cudf::column>>&& child_columns, std::vector<bool> const& validity = {})
        {
          init(std::move(child_columns), validity);
          /*
          size_type num_rows = child_columns.empty()? 0 : child_columns[0]->size();

          CUDF_EXPECTS(
            std::all_of(child_columns.begin(), child_columns.end(), [&](auto const& p_column) {return p_column->size() == num_rows;}), 
            "All struct member columns must have the same row count."
          );

          CUDF_EXPECTS(
            validity.size() <= 0 || static_cast<size_type>(validity.size()) == num_rows,
            "Validity buffer must have as many elements as rows in the struct column."
          );

          wrapped = cudf::make_structs_column(
            num_rows, 
            std::move(child_columns), 
            validity.size() <= 0? 0 : cudf::UNKNOWN_NULL_COUNT,
            validity.size() <= 0? rmm::device_buffer{0} : detail::make_null_mask(validity.begin(), validity.end()));
            */
        }

        structs_column_wrapper(std::initializer_list<std::reference_wrapper<detail::column_wrapper>> child_columns, std::vector<bool> const& validity = {})
        {
          std::vector<std::unique_ptr<cudf::column>> released;
          released.reserve(child_columns.size());
          std::transform(
            child_columns.begin(), 
            child_columns.end(), 
            std::back_inserter(released), 
            [&](auto column_wrapper){return column_wrapper.get().release();}
          );
          init(std::move(released), validity);
        }

      private:

        void init(std::vector<std::unique_ptr<cudf::column>>&& child_columns, std::vector<bool> const& validity)
        {
          size_type num_rows = child_columns.empty()? 0 : child_columns[0]->size();

          CUDF_EXPECTS(
            std::all_of(child_columns.begin(), child_columns.end(), [&](auto const& p_column) {return p_column->size() == num_rows;}), 
            "All struct member columns must have the same row count."
          );

          CUDF_EXPECTS(
            validity.size() <= 0 || static_cast<size_type>(validity.size()) == num_rows,
            "Validity buffer must have as many elements as rows in the struct column."
          );

          wrapped = cudf::make_structs_column(
            num_rows, 
            std::move(child_columns), 
            validity.size() <= 0? 0 : cudf::UNKNOWN_NULL_COUNT,
            validity.size() <= 0? rmm::device_buffer{0} : detail::make_null_mask(validity.begin(), validity.end()));
        }
    };
  }
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

  // auto struct_col = cudf::make_structs_column(2, std::move(cols), 0, {});
  cudf::test::structs_column_wrapper struct_col {
    std::move(cols)
  };

  std::cout << "Printing struct column: \n";
  cudf::test::print(struct_col.operator cudf::column_view());
  
}


TEST_F(StructColumnWrapperTest, SimpleStructColumnWrapperTest2)
{
  std::cout << "CALEB: Testing Struct Column!\n";

  auto names_col = cudf::test::strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald"
  };

  auto ages_col = cudf::test::fixed_width_column_wrapper<int8_t>{
    {48, 23, 103}, 
    {1, 1, 0}
  };

  cudf::test::structs_column_wrapper struct_col {
    {std::ref(static_cast<cudf::test::detail::column_wrapper&>(names_col)), std::ref(static_cast<cudf::test::detail::column_wrapper&>(ages_col))}, {}
  };

  std::cout << "Printing struct column: \n";
  cudf::test::print(struct_col.operator cudf::column_view());
  
}

CUDF_TEST_PROGRAM_MAIN()
