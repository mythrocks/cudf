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

#include <iterator>
#include <memory>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include "cudf/column/column_factories.hpp"
#include "cudf/detail/utilities/device_operators.cuh"
#include "cudf/types.hpp"
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

TEST_F(StructColumnWrapperTest, StructTest)
{
  std::cout << "CALEB: Testing Struct Column!\n";

  auto names_col = cudf::test::strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson "
  }.release();

  auto ages_col = cudf::test::fixed_width_column_wrapper<int8_t>{
    48, 
    23
  }.release();

  vector_of_columns cols;
  cols.push_back(std::move(names_col));
  cols.push_back(std::move(ages_col));

  // Table table{std::move(cols)};
  // std::cout << "Table: " << std::endl;
  // cudf::test::print(table.view().column(0));
  // cudf::test::print(table.view().column(1));

  auto struct_col = cudf::make_structs_column(2, std::move(cols), 0, {});
  std::cout << "Printing struct column: \n";
  cudf::test::print(struct_col->view());
  
}

CUDF_TEST_PROGRAM_MAIN()