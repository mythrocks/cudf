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

#pragma once

#include <cudf/lists/list_device_view.cuh>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Construct rmm::device_vector<list_device_view> from
 *        specified column.
 */
/*
rmm::device_vector<cudf::list_device_view> list_vector_from_column(
  cudf::detail::lists_column_device_view const& lists_col_d_v,
  cudaStream_t stream
)
{
  // auto ptr_lists_column_device_view = column_device_view::create(lists_column.parent(), stream);
  // auto lists_col_d_v = cudf::detail::lists_column_device_view(*ptr_lists_column_device_view);
  auto n_rows = lists_col_d_v.size();

  auto vector = rmm::device_vector<cudf::list_device_view>(n_rows);

  thrust::for_each_n(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<size_type>(0),
    n_rows,
    [
      lists_col_d_v,
      output = vector.data().get()
    ] __device__ (size_type idx)
    {
      // if (lists_col_d_v.is_null(idx))
      // {}
      output[idx] = lists_col_d_v[idx];
    }

  );
  return vector;
}
*/

/**
 * @brief Construct rmm::device_vector<list_device_view> from
 *        specified column.
 */
rmm::device_vector<cudf::list_device_view> list_vector_from_column(
  cudf::lists_column_view const& lists_column,
  cudaStream_t stream
)
{
  auto ptr_lists_column_device_view = column_device_view::create(lists_column.parent(), stream);
  auto lists_col_d_v = cudf::detail::lists_column_device_view(*ptr_lists_column_device_view);
  auto n_rows = lists_column.size();

  auto vector = rmm::device_vector<cudf::list_device_view>(n_rows);

  thrust::for_each_n(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<size_type>(0),
    n_rows,
    [
      lists_col_d_v,
      output = vector.data().get()
    ] __device__ (size_type idx)
    {
      // if (lists_col_d_v.is_null(idx))
      // {}
      output[idx] = lists_col_d_v[idx];
    }

  );
  return vector;
}

struct list_child_constructor
{
  template <typename T,
            std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()() const
  {
    std::cout << "CALEB: Fixed width list_child_constructor<" << typeid(T).name() << ">" << std::endl;
    CUDF_FAIL("list_child_constructor unimplemented!");
  }

  template <typename T, std::enable_if_t<!cudf::is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()() const
  {
    std::cout << "CALEB: list_child_constructor<" << typeid(T).name() << std::endl;
    CUDF_FAIL("list_child_constructor unsupported!");
  }

};

} // namespace;

/**
 * @brief Scatters lists into a copy of the target column
 * according to a scatter map.
 *
 * The scatter is performed according to the scatter iterator such that row
 * `scatter_map[i]` of the output column is replaced by the source list-row.
 * All other rows of the output column equal corresponding rows of the target table.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * The caller must update the null mask in the output column.
 *
 * @tparam SourceIterator must produce list_view objects
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New lists column.
 */
template <typename MapIterator>
std::unique_ptr<column> scatter(
  column_view const& source,
  MapIterator scatter_map_begin,
  MapIterator scatter_map_end,
  column_view const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0
  )
{
    auto child_column_type = lists_column_view(target).child().type();

    std::cout << "CALEB: Inside scatter_list()!" << std::endl;

    // TODO: FIXME:
    //  source_vector and target_vector contain list_device_views that depend on
    //  lists_column_device_views being alive, for use.
    //  Must construct lists_column_device_views here, before calling list_vector_from_column().

    // TODO: Deep(er) checks that source and target have identical types.

    auto source_lists_column_view = lists_column_view(source); // Checks that this is a list column.
    // auto source_device_view = column_device_view::create(source, stream);
    // auto source_lists_column_device_view = cudf::detail::lists_column_device_view(*column_device_view::create(source, stream));
    // auto source_vector = list_vector_from_column(source_lists_column_device_view, stream);
    auto source_vector = list_vector_from_column(source_lists_column_view, stream);
    std::cout << "CALEB: Source vector size: " << source_vector.size() << std::endl;

    auto target_lists_column_view = lists_column_view(target); // Checks that target is a list column.
    // auto target_device_view = column_device_view::create(target, stream);
    // auto target_lists_column_device_view = cudf::detail::lists_column_device_view(*column_device_view::create(target, stream));
    // auto target_vector = list_vector_from_column(target_lists_column_device_view, stream);
    auto target_vector = list_vector_from_column(target_lists_column_view, stream);
    std::cout << "CALEB: Target vector size: " << target_vector.size() << std::endl;

    // Scatter.
    thrust::scatter(
      rmm::exec_policy(stream)->on(stream),
      source_vector.begin(),
      source_vector.end(),
      scatter_map_begin,
      target_vector.begin()
    );

    return cudf::type_dispatcher(
      child_column_type, 
      list_child_constructor{}
    );
}

} // namespace detail;
} // namespace lists;
} // namespace cudf;
