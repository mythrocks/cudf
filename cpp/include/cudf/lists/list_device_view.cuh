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
#include <cudf/types.hpp>
#include <cudf/column/column_device_view.cuh>

/**
 * @file list_view.cuh
 * @brief Class definition for cudf::list_view.
 */

namespace cudf {

namespace detail {
    class lists_column_device_view;
}

/**
 * @brief A non-owning, immutable view of device data that represents
 * a list of elements of arbitrary type (including further nested lists).
 *
 */
class list_device_view {

    using lists_column_device_view = cudf::detail::lists_column_device_view;

  public:

    list_device_view() = default;

    CUDA_DEVICE_CALLABLE list_device_view(lists_column_device_view const* lists_column, size_type const& idx);

    ~list_device_view() = default;

    /**
     * @brief Fetches the offset in the list column's child that corresponds to
     * the element at the specified list index.
     *
     * Consider the following lists column:
     *  [
     *   [0,1,2],
     *   [3,4,5],
     *   [6,7,8]
     *  ]
     *
     * The list's internals would look like:
     *  offsets: [0, 3, 6, 9]
     *  child  : [0, 1, 2, 3, 4, 5, 6, 7, 8]
     *
     * The second list row (i.e. row_index=1) is [3,4,5].
     * The third element (i.e. idx=2) of the second list row is 5.
     *
     * The offset of this element as stored in the child column (i.e. 5)
     * may be fetched using this method.
     */
    CUDA_DEVICE_CALLABLE size_type element_offset(size_type idx) const;

    /**
     * @brief Fetches the element at the specified index, within the list row.
     *
     * @tparam The type of the list's element.
     * @param The index into the list row
     * @return The element at the specified index of the list row.
     */
    template <typename T>
    CUDA_DEVICE_CALLABLE T element(size_type idx) const;

    /**
     * @brief Checks whether element is null at specified index in the list row.
     */
    CUDA_DEVICE_CALLABLE bool is_null(size_type idx) const;

    /**
     * @brief Checks whether this list row is null.
     */
    CUDA_DEVICE_CALLABLE bool is_null() const;

    /**
     * @brief Fetches the number of elements in this list row.
     */
    CUDA_DEVICE_CALLABLE size_type size() const { return _size; }

    /**
     * @brief Fetches the lists_column_device_view that contains this list.
     */
    CUDA_DEVICE_CALLABLE lists_column_device_view const& get_column() const { return *lists_column; }

  private:

    lists_column_device_view const* lists_column; // TODO: FIXME: Ugly! Lifetime of device-view needs management.
    size_type _row_index{};  // Row index in the Lists column vector.
    size_type _size{};       // Number of elements in *this* list row.

    size_type begin_offset;  // Offset in list_column_device_view where this list begins.

};

namespace detail {

/**
 * @brief Given a column-device-view, an instance of this class provides a
 * wrapper on this compound column for list operations.
 * Analogous to list_column_view.
 */
class lists_column_device_view {

 public:
  lists_column_device_view() = delete;

  ~lists_column_device_view()                               = default;
  lists_column_device_view(lists_column_device_view const&) = default;
  lists_column_device_view(lists_column_device_view&&)      = default;

  lists_column_device_view(column_device_view const& underlying)
    : underlying(underlying)
  {
  }

  CUDA_HOST_DEVICE_CALLABLE cudf::size_type size() const
  {
    return underlying.size();
  }

  /**
   * @brief Fetches the list row at the specified index.
   * @param idx The index into the list column at which the list row
   * is to be fetched
   * @return list_device_view for the list row at the specified index.
   */
  CUDA_DEVICE_CALLABLE cudf::list_device_view operator[](size_type idx) const
  {
    return cudf::list_device_view{this, idx};
  }

  /**
   * @brief Fetches the offsets column of the underlying list column.
   */
  CUDA_DEVICE_CALLABLE column_device_view offsets() const { return underlying.child(0); }

  /**
   * @brief Fetches the child column of the underlying list column.
   */
  CUDA_DEVICE_CALLABLE column_device_view child() const { return underlying.child(1); }

  /**
   * @brief Indicates whether the list column is nullable.
   */
  CUDA_DEVICE_CALLABLE bool nullable() const { return underlying.nullable(); }

  /**
   * @brief Indicates whether the row (i.e. list) at the specified
   * index is null.
   */
  CUDA_DEVICE_CALLABLE bool is_null(size_type idx) const { return underlying.is_null(idx); }

 private:
  column_device_view underlying;
};

}  // namespace detail

CUDA_DEVICE_CALLABLE list_device_view::list_device_view(
  lists_column_device_view const* lists_column, size_type const& row_index)
  : lists_column(lists_column), _row_index(row_index)
{
  release_assert(row_index >= 0 && row_index < lists_column.size() && "row_index out of bounds");

  column_device_view const& offsets = lists_column->offsets();
  release_assert(row_index < offsets.size() && "row_index should not have exceeded offset size");

  begin_offset = offsets.element<size_type>(row_index);
  release_assert(begin_offset >= 0 && begin_offset < child().size() &&
                 "begin_offset out of bounds.");
  _size = offsets.element<size_type>(row_index + 1) - begin_offset;
}

CUDA_DEVICE_CALLABLE size_type list_device_view::element_offset(size_type idx) const
{
  release_assert(idx >= 0 && idx < size() && "idx out of bounds");
  release_assert(!is_null() && !is_null(idx) && "Cannot read null element.");
  return begin_offset + idx;
}

template <typename T>
CUDA_DEVICE_CALLABLE T list_device_view::element(size_type idx) const
{
  return lists_column->child().element<T>(element_offset(idx));
}

CUDA_DEVICE_CALLABLE bool list_device_view::is_null(size_type idx) const
{
  release_assert(idx >= 0 && idx < size() && "Index out of bounds.");
  auto element_offset = begin_offset + idx;
  return lists_column->child().is_null(element_offset);
}

CUDA_DEVICE_CALLABLE bool list_device_view::is_null() const
{
  return lists_column->is_null(_row_index);
}

}  // namespace cudf
