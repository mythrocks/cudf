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

#include <cuda_runtime.h>
#include <cudf/lists/list_device_view.cuh>
#include <cinttypes>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Construct rmm::device_vector<list_device_view> from
 *        specified column.
 */
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
      output[idx] = lists_col_d_v[idx];
    }

  );
  return vector;
}

struct list_child_constructor
{
  // Workaround for not being able to use a lambda.
  /*
  template <typename T>
  struct copy_child_column_elements
  {
    public:

    copy_child_column_elements(
      rmm::device_vector<cudf::list_device_view> const& list_vector,
      cudf::column_view const& list_offsets,
      cudf::mutable_column_view child_column)
      : d_list_device_view(list_vector.data().get()),
        d_offsets(list_offsets.template data<int32_t>()),
        d_child_column(child_column.template data<T>())
    {}

    void __device__ operator()(size_type const& list_row_index) const 
    {
      printf("CALEB: %d\n", list_row_index);
      auto start_offset = d_offsets[list_row_index];
      auto list_device_view = d_list_device_view[list_row_index];
      
      // Copy each element in the list-row to its position in the child column.
      // TODO: memcpy() instead?
      thrust::for_each_n(
        thrust::seq,
        thrust::make_counting_iterator<size_type>(0),
        list_device_view.size(),
        [
          start_offset, 
          list_device_view, 
          d_child_column = this->d_child_column] __device__ (auto list_element_idx)
        {
          d_child_column[start_offset + list_element_idx] = list_device_view.element<T>(list_element_idx);
        }
      );
    }

    private:

      cudf::list_device_view const * d_list_device_view;
      int32_t const* d_offsets;
      T* d_child_column;
  };
  */

  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    rmm::device_vector<cudf::list_device_view> const& list_vector, 
    cudf::column_view const& list_offsets,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) const
  {
    int32_t num_child_rows{};
    CUDA_TRY(cudaMemcpyAsync(&num_child_rows, list_offsets.data<int32_t>()+list_offsets.size()-1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

    auto child_column = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_to_id<T>()},
      num_child_rows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr
    );

    /*
    thrust::for_each_n(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<size_type>(0),
      num_child_rows,
      copy_child_column_elements<T>{list_vector, list_offsets, child_column->mutable_view()}
    );
    */

    thrust::for_each_n(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<size_type>(0),
      num_child_rows,
      [
        d_list_device_view = list_vector.data().get(),
        d_offsets          = list_offsets.template data<int32_t>(),
        d_child_column     = child_column->mutable_view().data<T>()
      ] __device__ (cudf::size_type row_index)
      {
        auto start_offset = d_offsets[row_index];
        // auto end_offset   = d_offsets[row_index + 1];
        auto list_device_view = d_list_device_view[row_index];

        thrust::for_each_n(
          thrust::seq,
          thrust::make_counting_iterator<size_type>(0),
          list_device_view.size(),
          [start_offset, list_device_view, d_child_column] __device__ (auto list_element_idx)
          {
            d_child_column[start_offset + list_element_idx] = list_device_view.element<T>(list_element_idx);
          }
        );
      }
    );

    // return std::make_unique<cudf::column>(list_offsets); // TODO: Replace with constructed child column.
    return std::make_unique<cudf::column>(child_column->view()); // TODO: Replace with constructed child column.
  }

  template <typename T> 
  std::enable_if_t<!cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    rmm::device_vector<cudf::list_device_view> const& list_vector, 
    cudf::column_view const& list_offsets,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) const
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

    // TODO: Deep(er) checks that source and target have identical types.

    auto source_lists_column_view = lists_column_view(source); // Checks that this is a list column.
    auto source_device_view = column_device_view::create(source, stream);
    auto source_lists_column_device_view = cudf::detail::lists_column_device_view(*source_device_view);
    auto source_vector = list_vector_from_column(source_lists_column_device_view, stream);
    std::cout << "CALEB: Source vector size: " << source_vector.size() << std::endl;

    auto target_lists_column_view = lists_column_view(target); // Checks that target is a list column.
    auto target_device_view = column_device_view::create(target, stream);
    auto target_lists_column_device_view = cudf::detail::lists_column_device_view(*target_device_view);
    auto target_vector = list_vector_from_column(target_lists_column_device_view, stream);
    std::cout << "CALEB: Target vector size: " << target_vector.size() << std::endl;

    // Scatter.
    thrust::scatter(
      rmm::exec_policy(stream)->on(stream),
      source_vector.begin(),
      source_vector.end(),
      scatter_map_begin,
      target_vector.begin()
    );

    std::cout << "CALEB: Post scatter: Target vector size: " << target.size() << std::endl;
    thrust::for_each(
      rmm::exec_policy(stream)->on(stream),
      // thrust::seq,
      target_vector.begin(),
      target_vector.end(),
      []__device__(auto list)
      {
        printf(" list(size:%" PRId32 ") [", list.size());
        for (int i(0); i<list.size(); ++i)
        {
          printf("%" PRId32, list.template element<int32_t>(i));
        }
        printf("]\n");
      }
    );

    auto list_size_begin = thrust::make_transform_iterator(target_vector.begin(), [] __device__(list_device_view l) { return l.size(); });
    auto offsets_column = cudf::strings::detail::make_offsets_child_column(
      list_size_begin,
      list_size_begin + target.size(),
      mr,
      stream
    );

    return cudf::type_dispatcher(
      child_column_type, 
      list_child_constructor{},
      target_vector,
      offsets_column->view(),
      mr,
      stream
    );
}

} // namespace detail;
} // namespace lists;
} // namespace cudf;
