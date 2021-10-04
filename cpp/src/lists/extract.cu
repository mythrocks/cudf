/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/gather.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <limits>
#include <rmm/cuda_stream_view.hpp>

#include <thrust/detail/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {

std::unique_ptr<cudf::column> make_index_child_column(column_view const& indices,
                                                      rmm::cuda_stream_view stream)
{
  // New column, near identical to `indices`, except with null values replaced.
  // `segmented_gather()` on a null index should produce a null row.
  if (not indices.nullable()) { return std::make_unique<column>(indices, stream); }

  auto const indices_device_view = column_device_view::create(indices);
  auto const d_indices           = *indices_device_view;
  auto const null_index          = std::numeric_limits<size_type>::max();
  auto const null_replaced_iter_begin =
    cudf::detail::make_null_replacement_iterator(d_indices, null_index);
  auto index_child = cudf::make_numeric_column(
    data_type{type_id::INT32}, indices.size(), mask_state::UNALLOCATED, stream);
  thrust::copy(rmm::exec_policy(stream),
               null_replaced_iter_begin,
               null_replaced_iter_begin + indices.size(),
               index_child->mutable_view().begin<size_type>());
  return index_child;
}

}  // namespace

/**
 * @copydoc cudf::lists::extract_list_element
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             size_type index,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto const num_lists = lists_column.size();
  if (num_lists == 0) return empty_like(lists_column.child());

  auto index_child =  // [index, index, index, ..., index]
    make_numeric_column(data_type{type_id::INT32}, num_lists, mask_state::UNALLOCATED, stream);
  thrust::copy_n(rmm::exec_policy(stream),
                 thrust::make_constant_iterator(size_type{index}),
                 num_lists,
                 index_child->mutable_view().begin<size_type>());

  auto index_offsets =  // [0, 1, 2, 3, ... num_lists + 1]
    make_numeric_column(data_type{type_id::INT32}, num_lists + 1, mask_state::UNALLOCATED, stream);
  thrust::copy_n(rmm::exec_policy(stream),
                 cudf::detail::make_counting_transform_iterator(0, thrust::identity<size_type>{}),
                 num_lists + 1,
                 index_offsets->mutable_view().begin<size_type>());

  auto index_lists =  // [(index), (index), (index), ..., (index)]
    make_lists_column(num_lists, std::move(index_offsets), std::move(index_child), 0, {}, stream);

  auto extracted_lists =
    segmented_gather(lists_column, index_lists->view(), out_of_bounds_policy::NULLIFY, stream, mr);
  return std::move(extracted_lists->release().children[lists_column_view::child_column_index]);
}

/**
 * @copydoc cudf::lists::extract_list_element
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             column_view const& indices,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto const num_lists = lists_column.size();
  if (num_lists == 0) return empty_like(lists_column.child());

  CUDF_EXPECTS(indices.size() == num_lists,
               "Index column must have as many elements as lists column.");
  // TODO: Assert on index type?

  auto index_offsets =  // [0, 1, 2, 3, ... num_lists + 1]
    make_numeric_column(data_type{type_id::INT32}, num_lists + 1, mask_state::UNALLOCATED, stream);
  thrust::copy_n(rmm::exec_policy(stream),
                 cudf::detail::make_counting_transform_iterator(0, thrust::identity<size_type>{}),
                 num_lists + 1,
                 index_offsets->mutable_view().begin<size_type>());

  auto index_child =  // [indices[0], indices[1], indices[2], ..., indices[n-1]]
    make_index_child_column(indices, stream);

  auto index_lists =
    make_lists_column(num_lists, std::move(index_offsets), std::move(index_child), 0, {}, stream);

  auto extracted_lists =
    segmented_gather(lists_column, index_lists->view(), out_of_bounds_policy::NULLIFY, stream, mr);

  return std::move(extracted_lists->release().children[lists_column_view::child_column_index]);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::extract_list_element
 */
std::unique_ptr<column> extract_list_element(lists_column_view const& lists_column,
                                             size_type index,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::extract_list_element(lists_column, index, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_list_element(lists_column_view const& lists_column,
                                             column_view const& indices,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::extract_list_element(lists_column, indices, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
