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
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <structs/utilities.hpp>

namespace cudf::test {

struct UtilitiesTest : BaseFixture {};

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;
using column_index_t = typename vector_of_columns::size_type;

std::unique_ptr<cudf::column> unflatten_struct(vector_of_columns& flattened,
                                               column_index_t& current_index,
                                               cudf::column_view blueprint,
                                               vector_of_columns& struct_null_vectors,
                                               column_index_t& current_null_vector_index)
{
  // "Consume" columns from `flattened`, starting at `current_index`,
  // based on the provided `blueprint` struct col. Recurse for struct children.
  CUDF_EXPECTS(blueprint.type().id() == type_id::STRUCT, 
               "Expected blueprint column to be a STRUCT column.");

  CUDF_EXPECTS(not flattened.empty(), "STRUCT column can't have 0 children.");

  auto num_rows = flattened.front()->size();

  // Extract null-vector *before* child columns are constructed.
  // Child struct column null vectors appear *before* parent struct.
  auto struct_null_column_contents = struct_null_vectors[current_null_vector_index++]->release();

  auto struct_members = vector_of_columns{};
  struct_members.reserve(blueprint.num_children());

  std::transform(blueprint.child_begin(),
                 blueprint.child_end(),
                 std::back_inserter(struct_members),
                 [&flattened,
                  &current_index,
                  &struct_null_vectors,
                  &current_null_vector_index] (auto member) {
                   return member.type().id() == type_id::STRUCT
                     ? unflatten_struct(flattened, 
                                        current_index, 
                                        member, 
                                        struct_null_vectors, 
                                        current_null_vector_index)
                     : std::move(flattened[current_index++]);
                 });

  return make_structs_column(num_rows,
                             std::move(struct_members),
                             UNKNOWN_NULL_COUNT, // Do count?
                             std::move(*struct_null_column_contents.null_mask)); // TODO: stream, mr?
}

std::unique_ptr<cudf::table> unflatten(std::unique_ptr<cudf::table>&& flattened, // Adopted from. Must be table, not view.
                                       table_view blueprint,                   // For reconstruction.
                                       std::vector<std::unique_ptr<cudf::column>>&& struct_null_vectors) // && ?
{
  auto const n_struct_columns = std::count_if(blueprint.begin(),
                                              blueprint.end(),
                                              [](auto const& col) { return col.type().id() == type_id::STRUCT; });
  if (n_struct_columns == 0) 
  {
    return std::move(flattened); // Unchanged.
  }

  // There be struct columns.
  // Requires null vectors for all struct input columns.
  // TODO: Explore if blueprint's struct's has_nulls() should be used at all.
  //       At first glance, no. `groupby.aggregate()` might have filtered out nulls.
  CUDF_EXPECTS(n_struct_columns == static_cast<decltype(n_struct_columns)>(struct_null_vectors.size()), 
                "Cannot unflatten: Number of null vectors must match struct columns.");


  auto flattened_columns = flattened->release();
  auto current_idx = column_index_t{0};
  auto current_null_vector_idx = column_index_t{0};

  auto return_columns = vector_of_columns{};
  std::transform(blueprint.begin(),
                 blueprint.end(),
                 std::back_inserter(return_columns),
                 [&flattened_columns,
                 &current_idx,
                 &struct_null_vectors,
                 &current_null_vector_idx](auto blueprint_column) {
                   return blueprint_column.type().id() == type_id::STRUCT
                     ? unflatten_struct(flattened_columns, 
                                         current_idx, 
                                         blueprint_column, 
                                         struct_null_vectors, 
                                         current_null_vector_idx)
                     : std::move(flattened_columns[current_idx++]);
                 });
  
  return std::make_unique<cudf::table>(std::move(return_columns));
}

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
  
  auto unflattened = unflatten(std::move(flattened_table), 
                               input_table, 
                               std::move(nullability_vectors));

  std::cout << "Unflattened column: " << std::endl;
  print(unflattened->view());

  /*
  auto& null_cols  = std::get<3>(flattened);
  auto  gby_input  = std::vector<column_view>{output_table.begin(), output_table.end()};
  std::transform(null_cols.begin(), 
                 null_cols.end(), 
                 std::back_inserter(gby_input), 
                 [&](auto const& col){ return col->view(); });
  
  auto gby = groupby::groupby{table_view{gby_input}, null_policy::INCLUDE, sorted::NO};

  std::vector<std::unique_ptr<aggregation>> aggs;
  aggs.emplace_back(make_sum_aggregation());
  auto requests = std::vector<aggregation_request>{{aggregation_request{agg_values->view(), std::move(aggs)}}};
  */
}

}  // namespace cudf::test
