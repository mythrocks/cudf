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

#include <cinttypes>
#include <cuda_runtime.h>
#include <cudf/types.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace lists {
namespace detail {

namespace {

// TODO: Rename to `unbound_child_view`.
//       Carries only positional information, unbound to list_view.
struct scattered_list_row
{

  enum label_t : bool {SOURCE, TARGET};

  using lists_column_device_view = cudf::detail::lists_column_device_view;
  using list_device_view = cudf::list_device_view;

  scattered_list_row() = default;

  CUDA_DEVICE_CALLABLE scattered_list_row(label_t scatter_source_label,
                                          cudf::detail::lists_column_device_view const& lists_column,
                                          size_type const& row_index)
    : _label{scatter_source_label},
      _row_index{row_index}
  {
    auto actual_list_row = list_device_view{lists_column, row_index};
    _size = actual_list_row.size();
  }

  CUDA_DEVICE_CALLABLE scattered_list_row(label_t scatter_source_label,
                                          size_type const& row_index,
                                          size_type const& size)
    : _label{scatter_source_label},
      _row_index{row_index},
      _size{size}
  {}

  CUDA_DEVICE_CALLABLE size_type size() const { return _size; }
  CUDA_DEVICE_CALLABLE label_t label() const { return _label; }
  CUDA_DEVICE_CALLABLE bool is_from_scatter_source() const { return label() == SOURCE; }
  CUDA_DEVICE_CALLABLE size_type row_index() const { return _row_index; }

  // TODO: Rename to `bind_to_lists_column`?
  CUDA_DEVICE_CALLABLE list_device_view to_list_device_view(
    lists_column_device_view const& scatter_source,
    lists_column_device_view const& scatter_target) const 
  {
    return list_device_view(_label == SOURCE? scatter_source : scatter_target, _row_index);
  }

  private:

    // Note: Cannot store reference to list column, because of storage in device_vector.
    // Only keep track of whether this list row came from the source or target of scatter.

    label_t _label {SOURCE}; // Whether this list row came from the scatter source or target. 
    size_type _row_index{};         // Row index in the Lists column.
    size_type _size{};              // Number of elements in *this* list row.
};

rmm::device_vector<scattered_list_row> list_vector_from_column(
  scattered_list_row::label_t label,
  cudf::detail::lists_column_device_view const& lists_column,
  cudaStream_t stream
)
{
  auto n_rows = lists_column.size();

  auto vector = rmm::device_vector<scattered_list_row>(n_rows);

  thrust::for_each_n(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<size_type>(0),
    n_rows,
    [
      label,
      lists_column,
      output = vector.data().get()
    ] __device__ (size_type row_index)
    {
      output[row_index] = scattered_list_row{label, lists_column, row_index};
    }
  );

  return vector;
}

struct list_child_constructor
{
  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    rmm::device_vector<scattered_list_row> const& list_vector, 
    cudf::column_view const& list_offsets,
    cudf::detail::lists_column_device_view const& source_lists,
    cudf::detail::lists_column_device_view const& target_lists,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) const
  {
    // Number of rows in child-column == last offset value.
    int32_t num_child_rows{};
    CUDA_TRY(cudaMemcpyAsync(&num_child_rows, 
                             list_offsets.data<int32_t>()+list_offsets.size()-1, 
                             sizeof(int32_t), 
                             cudaMemcpyDeviceToHost, 
                             stream));
    CUDA_TRY(cudaStreamSynchronize(stream));

    // Init child-column.
    auto child_column = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_to_id<T>()},
      num_child_rows,
      cudf::mask_state::UNALLOCATED, // TODO: Figure out child column null mask.
      stream,
      mr
    );

    // Function to copy child-values for specified index of scattered_list_row
    // to the child column.
    auto copy_child_values_for_list_index = [
      d_scattered_lists = list_vector.data().get(), // scattered_list_row*
      d_child_column    = child_column->mutable_view().data<T>(),
      d_offsets         = list_offsets.template data<int32_t>(),
      source_lists,
      target_lists
    ] __device__ (auto const& row_index) {

      auto scattered_list_row = d_scattered_lists[row_index];
      auto actual_list_row    = scattered_list_row.to_list_device_view(source_lists, target_lists);
      
      // Copy all elements in this list row, to "appropriate" offset in child-column.
      auto destination_start_offset = d_offsets[row_index];
      thrust::for_each_n(
        thrust::seq,
        thrust::make_counting_iterator<size_type>(0),
        actual_list_row.size(),
        [actual_list_row, d_child_column, destination_start_offset] __device__ (auto const& list_element_index)
        {
          d_child_column[destination_start_offset + list_element_index] =
            actual_list_row.template element<T>(list_element_index);
        }
      );
    };

    // For each list-row, copy underlying elements to the child column.
    thrust::for_each_n(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<size_type>(0),
      list_vector.size(),
      copy_child_values_for_list_index
    );

    return std::make_unique<column>(child_column->view());
  }

  /**
   * @brief Utility function to fetch the number of rows in a lists column's
   *        child column, given its offsets column.
   *        (This is simply the last value in the offsets column.)
   * 
   * @param list_offsets Offsets child of a lists column
   * @param stream The cuda-stream to synchronize on, when reading from device memory
   * @return int32_t The last element in the list_offsets column, indicating
   *         the number of rows in the lists-column's child.
   */
  static int32_t get_num_child_rows(cudf::column_view const& list_offsets, cudaStream_t stream)
  {
    // Number of rows in child-column == last offset value.
    int32_t num_child_rows{};
    CUDA_TRY(cudaMemcpyAsync(&num_child_rows, 
                             list_offsets.data<int32_t>()+list_offsets.size()-1, 
                             sizeof(int32_t), 
                             cudaMemcpyDeviceToHost, 
                             stream));
    CUDA_TRY(cudaStreamSynchronize(stream));  
    return num_child_rows;
}

  template <typename T> 
  std::enable_if_t<std::is_same<T, string_view>::value, std::unique_ptr<column>> operator()(
    rmm::device_vector<scattered_list_row> const& list_vector, 
    cudf::column_view const& list_offsets,
    cudf::detail::lists_column_device_view const& source_lists,
    cudf::detail::lists_column_device_view const& target_lists,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) const
  {
    int32_t num_child_rows{get_num_child_rows(list_offsets, stream)};

    auto string_views = rmm::device_vector<string_view>(num_child_rows);

    auto populate_string_views = [
      d_scattered_lists = list_vector.data().get(), // scattered_list_row*
      d_list_offsets    = list_offsets.template data<int32_t>(),
      d_string_views    = string_views.data().get(),
      source_lists,
      target_lists
    ] __device__ (auto const& row_index) {

      auto scattered_list_row    = d_scattered_lists[row_index];
      auto actual_list_row       = scattered_list_row.to_list_device_view(source_lists, target_lists);
      auto lists_column          = actual_list_row.get_column();
      auto lists_offsets_column  = lists_column.offsets();
      auto child_strings_column  = lists_column.child();
      auto string_offsets_column = child_strings_column.child(cudf::strings_column_view::offsets_column_index);
      auto string_chars_column   = child_strings_column.child(cudf::strings_column_view::chars_column_index);

      auto output_start_offset = d_list_offsets[row_index]; // Offset in `string_views` at which string_views are 
                                                            // to be written for this list row_index.
      auto input_list_start = lists_offsets_column.template element<int32_t>(scattered_list_row.row_index());

      thrust::for_each_n(
        thrust::seq,
        thrust::make_counting_iterator<size_type>(0),
        actual_list_row.size(),
        [
          output_start_offset,
          d_string_views,
          input_list_start,
          d_string_offsets = string_offsets_column.template data<int32_t>(),
          d_string_chars   = string_chars_column.template data<char>()
        ] __device__ (auto const& string_idx)
        {
          // auto string_offset     = output_start_offset + string_idx;
          auto string_start_idx  = d_string_offsets[input_list_start + string_idx];
          auto string_end_idx    = d_string_offsets[input_list_start + string_idx + 1];

          d_string_views[output_start_offset + string_idx] = 
            string_view{d_string_chars + string_start_idx, string_end_idx - string_start_idx};
        }
      );
    };

    thrust::for_each_n(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<size_type>(0),
      list_vector.size(),
      populate_string_views
    );

    // string_views should now have been populated with source and target references.

    auto string_offsets = cudf::strings::detail::child_offsets_from_string_vector(string_views, mr, stream);
    auto string_chars   = cudf::strings::detail::child_chars_from_string_vector(string_views, string_offsets->view().data<int32_t>(), 0, mr, stream);

    return cudf::make_strings_column(num_child_rows,
                                     std::move(string_offsets),
                                     std::move(string_chars),
                                     cudf::UNKNOWN_NULL_COUNT,
                                     {}, stream, mr);
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, list_view>::value, 
                   std::unique_ptr<column>> 
  operator() (
    rmm::device_vector<scattered_list_row> const& list_vector, 
    cudf::column_view const& list_offsets,
    cudf::detail::lists_column_device_view const& source_lists,
    cudf::detail::lists_column_device_view const& target_lists,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) const
  {
    auto num_child_rows = get_num_child_rows(list_offsets, stream);

    auto child_list_views = rmm::device_vector<scattered_list_row>(num_child_rows);

    // Function to convert from parent list_device_view instances to child list_device_views.
    // For instance, if a parent list_device_view has 3 elements, it should have 3 corresponding
    // child list_device_view instances.
    auto populate_child_list_views = [
      d_scattered_lists  = list_vector.data().get(),
      d_list_offsets     = list_offsets.template data<int32_t>(),
      d_child_list_views = child_list_views.data().get(),
      source_lists,
      target_lists
    ] __device__ (auto const& row_index) {

      auto scattered_row        = d_scattered_lists[row_index];
      auto label                = scattered_row.label();
      auto bound_list_row       = scattered_row.to_list_device_view(source_lists, target_lists);
      auto lists_offsets_column = bound_list_row.get_column().offsets();

      auto child_column         = bound_list_row.get_column().child();
      auto child_offsets        = child_column.child(cudf::lists_column_view::offsets_column_index);

      // For lists row at row_index,
      //   1. Number of entries in child_list_views == bound_list_row.size().
      //   2. Offset of the first child list_view   == d_list_offsets[row_index].
      auto output_start_offset  = d_list_offsets[row_index];
      auto input_list_start     = lists_offsets_column.template element<int32_t>(scattered_row.row_index());

      thrust::for_each_n(
        thrust::seq,
        thrust::make_counting_iterator<size_type>(0),
        bound_list_row.size(),
        [
          input_list_start,
          output_start_offset,
          label,
          d_child_list_views,
          d_child_offsets = child_offsets.template data<int32_t>()
        ] __device__ (auto const& child_list_index)
        {
          auto child_start_idx = d_child_offsets[input_list_start + child_list_index];
          auto child_end_idx   = d_child_offsets[input_list_start + child_list_index + 1];

          d_child_list_views[output_start_offset + child_list_index] = 
            scattered_list_row{label, child_start_idx, child_end_idx - child_start_idx};
        }
      );
    };

    thrust::for_each_n(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<size_type>(0),
      list_vector.size(),
      populate_child_list_views
    );

    // child_list_views should now have been populated, with source and target references.

    auto begin = thrust::make_transform_iterator(
      child_list_views.begin(), 
      [] __device__ (auto const& row) { return row.size(); }
    );

    // TODO: Implement recursion with type dispatch. Placeholder follows:

    return cudf::strings::detail::make_offsets_child_column(
      begin,
      begin + child_list_views.size(),
      mr,
      stream
    );

    /*
    auto child_offsets = cudf::strings::detail::make_offsets_child_column(
      begin,
      begin + child_list_views.size(),
      mr,
      stream
    );

    // TODO: Oh, so close! column_device_view::child() is __device__ only.
    return cudf::type_dispatcher(
      source_lists.child().type(),
      list_child_constructor{},
      child_list_views,
      child_offsets->view(),
      cudf::detail::lists_column_device_view(source_lists.child()),
      cudf::detail::lists_column_device_view(target_lists.child()),
      mr,
      stream
    );
    */
  }

  template <typename T>
  struct is_supported_child_type
  {
    static const bool value = cudf::is_fixed_width<T>()
                           || std::is_same<T, string_view>::value
                           || std::is_same<T, list_view>::value;
  };

  template <typename T> 
  std::enable_if_t<!is_supported_child_type<T>::value, std::unique_ptr<column>> operator()(
    rmm::device_vector<scattered_list_row> const& list_vector, 
    cudf::column_view const& list_offsets,
    cudf::detail::lists_column_device_view const& source_list,
    cudf::detail::lists_column_device_view const& target_list,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) const
  {
    CUDF_FAIL("list_child_constructor unsupported!");
  }
};

void debug_print(rmm::device_vector<scattered_list_row> const& vector, std::string const& msg = "")
{
    std::cout << msg << " Vector size: " << vector.size() << std::endl;
    thrust::for_each(
      thrust::device,
      vector.begin(),
      vector.end(),
      []__device__(auto list)
      {
        /*
        for (int i(0); i<list.size(); ++i)
        {
          printf("%" PRId32, list.template element<int32_t>(i));
        }
        printf("]\n");
        */
        printf(" list(size:%" PRId32 ") [%s] \n", list.size(), list.is_from_scatter_source()? "SOURCE" : "TARGET");
      }
    );    
}

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
    auto num_rows = target.size();

    if (num_rows == 0)
    {
      return cudf::empty_like(target);
    }

    auto child_column_type = lists_column_view(target).child().type();

    // TODO: Deep(er) checks that source and target have identical types.

    auto source_lists_column_view = lists_column_view(source); // Checks that this is a list column.
    auto source_device_view = column_device_view::create(source, stream);
    auto source_lists_column_device_view = cudf::detail::lists_column_device_view(*source_device_view);
    auto source_vector = list_vector_from_column(cudf::lists::detail::scattered_list_row::SOURCE, source_lists_column_device_view, stream);

    auto target_lists_column_view = lists_column_view(target); // Checks that target is a list column.
    auto target_device_view = column_device_view::create(target, stream);
    auto target_lists_column_device_view = cudf::detail::lists_column_device_view(*target_device_view);
    auto target_vector = list_vector_from_column(cudf::lists::detail::scattered_list_row::TARGET, target_lists_column_device_view, stream);

    // Scatter.
    thrust::scatter(
      rmm::exec_policy(stream)->on(stream),
      source_vector.begin(),
      source_vector.end(),
      scatter_map_begin,
      target_vector.begin()
    );

    auto list_size_begin = thrust::make_transform_iterator(target_vector.begin(), [] __device__(scattered_list_row l) { return l.size(); });
    auto offsets_column = cudf::strings::detail::make_offsets_child_column(
      list_size_begin,
      list_size_begin + target.size(),
      mr,
      stream
    );

    auto child_column = cudf::type_dispatcher( 
      child_column_type, 
      list_child_constructor{},
      target_vector,
      offsets_column->view(),
      source_lists_column_device_view,
      target_lists_column_device_view,
      mr,
      stream
    );

    rmm::device_buffer null_mask{0, stream, mr};
    if (target.has_nulls()) {
      null_mask = copy_bitmask(target, stream, mr);
    }

    return cudf::make_lists_column(
      num_rows,
      std::move(offsets_column),
      std::move(child_column),
      cudf::UNKNOWN_NULL_COUNT,
      std::move(null_mask),
      stream,
      mr
    );
}

} // namespace detail;
} // namespace lists;
} // namespace cudf;
