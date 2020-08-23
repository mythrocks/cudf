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

#include <cudf/types.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cstdio>
#include <cuda_runtime.h>

/**
 * @file list_device_view.cuh
 * @brief Class definition for cudf::list_device_view.
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

        CUDA_DEVICE_CALLABLE list_device_view(
            lists_column_device_view const& lists_column, 
            size_type const& idx);

        list_device_view(list_device_view const&) = default;
        list_device_view& operator=(list_device_view const&) = default;

        CUDA_DEVICE_CALLABLE bool operator == (list_device_view const& rhs) const;
        CUDA_DEVICE_CALLABLE bool operator != (list_device_view const& rhs) const
        {
            return !(*this == rhs);
        }

        template<typename T>
        CUDA_DEVICE_CALLABLE T element(size_type idx) const;

        // Check if element at index idx is null.
        CUDA_DEVICE_CALLABLE bool is_null(size_type idx) const;

        // Check if list itself is null.
        CUDA_DEVICE_CALLABLE bool is_null() const;

        CUDA_DEVICE_CALLABLE size_type size() const {return _size;}

    private:

        lists_column_device_view const& lists_column;
        size_type _row_index{}; // Row index in the Lists column vector.
        size_type _size{};      // Number of elements in *this* list row.

        size_type begin_offset; // Offset in list_column_device_view where this list begins.
};

namespace detail {

class lists_column_device_view 
{
    public:

        lists_column_device_view() = delete;

        ~lists_column_device_view() = default;
        lists_column_device_view(lists_column_device_view const&) = default;
        lists_column_device_view(lists_column_device_view &&) = default;
        
        CUDA_DEVICE_CALLABLE lists_column_device_view(column_device_view const& underlying)
            :underlying(underlying)
        {}

        CUDA_DEVICE_CALLABLE cudf::list_device_view operator[](size_type idx) const
        {
            return cudf::list_device_view{*this, idx};
        }

        CUDA_DEVICE_CALLABLE column_device_view offsets() const
        {
            return underlying.child(0);
        }

        CUDA_DEVICE_CALLABLE column_device_view child() const
        {
            return underlying.child(1);
        }

        CUDA_DEVICE_CALLABLE bool is_null(size_type idx) const
        {
            return underlying.is_null(idx);
        }

    private:

        column_device_view underlying;
};

} // namespace detail;

CUDA_DEVICE_CALLABLE list_device_view::list_device_view(lists_column_device_view const& lists_column, size_type const& row_index)
    : lists_column(lists_column), _row_index(row_index)
{
    // TODO: release_assert(idx in [0, offsets.size()));

    column_device_view const& offsets = lists_column.offsets();
    begin_offset = offsets.element<size_type>(row_index);
    _size = offsets.element<size_type>(row_index+1) - begin_offset;
}


template<typename T>
CUDA_DEVICE_CALLABLE T list_device_view::element(size_type idx) const
{
    // TODO: release_assert() for idx < size of list row.
    auto element_offset = begin_offset + idx;
    return lists_column.child().element<T>(element_offset); 
}

CUDA_DEVICE_CALLABLE bool list_device_view::is_null(size_type idx) const
{
    // TODO: release_assert() for idx < size of list row.
    auto element_offset = begin_offset + idx;
    return lists_column.child().is_null(element_offset);
}

CUDA_DEVICE_CALLABLE bool list_device_view::is_null() const
{
    return lists_column.is_null(_row_index);
}

template<bool has_nulls=true>
class list_element_equality_comparator
{
    public:
        CUDA_DEVICE_CALLABLE list_element_equality_comparator(
            list_device_view const& lhs, 
            list_device_view const& rhs,
            bool nulls_are_equal = true)
          : lhs(lhs), rhs(rhs), nulls_are_equal(nulls_are_equal)
        {}

        template <typename T,
                  std::enable_if_t<!std::is_same<T, cudf::struct_view>::value>* = nullptr>
        CUDA_DEVICE_CALLABLE bool operator()(size_type i)
        {
            printf("CALEB: list_element_equality_comparator::op()!\n");
            if (has_nulls)
            {
                bool lhs_is_null = lhs.is_null(i);
                bool rhs_is_null = rhs.is_null(i);
                if (lhs_is_null && rhs_is_null)
                {
                    return nulls_are_equal;
                }
                else if (lhs_is_null != rhs_is_null)
                {
                    return false;
                }
            }

           return lhs.element<T>(i) == rhs.element<T>(i);
        } 

        template <typename T,
                  std::enable_if_t<std::is_same<T, cudf::struct_view>::value>* = nullptr>
        CUDA_DEVICE_CALLABLE bool operator()(size_type i)
        {
            release_assert(false && "list_element_equality_comparator does not support STRUCT!");
            return false;
        }
    
    private:

        list_device_view const& lhs;
        list_device_view const& rhs;
        bool nulls_are_equal;
};

CUDA_DEVICE_CALLABLE bool list_device_view::operator == (list_device_view const& rhs) const
{
    printf("CALEB: list_device_view::operator ==()!\n");

    auto element_type{lists_column.child().type()};
    release_assert(rhs.lists_column.child().type() == element_type && "List-Element type mismatch!");

    if (is_null() && rhs.is_null())
    {
        printf("BOTH NULL! EQUAL!\n");
        return true;
    }

    if (size() != rhs.size())
    {
        printf("CALEB: list_device_view::operator ==()! Sizes are different! \n");
        return false;
    }

    if (element_type.id() == cudf::type_id::STRUCT)
    {
        printf("CALEB: list_device_view::operator ==()! TODO: Implement list<struct> \n");
        return false; // TODO: Implement nesting.
    }

    if (element_type.id() == cudf::type_id::LIST)
    {
        // List of lists.
        // Must compare each list that this list-element contains, against rhs's.

        lists_column_device_view lhs_lists_column{lists_column.child()};
        lists_column_device_view rhs_lists_column{rhs.lists_column.child()};
        for(size_type i{0}; i<size(); ++i)
        {
            if (lhs_lists_column[i+begin_offset] != rhs_lists_column[i+rhs.begin_offset])
            {
                return false;
            }
        }
        return true;
    }

    // Compare primitive elements.
    // For each i between the corresponding [begin,end) offsets of lhs and rhs,
    // type-dispatch the comparisons.
    
    list_element_equality_comparator<true> compare_eq{*this, rhs};

    for (size_type i{0}; i<size(); ++i)
    {
        if (!cudf::type_dispatcher(element_type, compare_eq, i)) {
            return false;
        }
    }

    return true;
}

}  // namespace cudf
