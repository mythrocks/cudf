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
// #include <cudf/column/column_device_view.cuh>
#include <cstdio>
#include <cuda_runtime.h>

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
class list_view {

    using lists_column_device_view = cudf::detail::lists_column_device_view;

    public:

        // TODO: Verify that this is necessary. Else, remove.
        __host__ __device__ list_view()
        {
            printf("CALEB: list_view::default_ctor()!\n");
        }

        list_view(lists_column_device_view const* device_column_view, size_type const& idx)
            : _p_lists_column_device_view(device_column_view), _element_index(idx)
        {
            printf("CALEB: list_view::non-default-ctor!\n");
            printf("\tCALEB: device_column_view == %s\n", (device_column_view)? "NON_NULL" : "NULL");
            printf("\tCALEB: index: %d\n", idx);
        }

        list_view(list_view const&) = default;
        list_view& operator=(list_view const&) = default;

        __device__ bool operator == (list_view const& rhs) const
        {
            printf("CALEB: list_view::operator ==()!\n");

            return false;
        }

    private:

        lists_column_device_view const* _p_lists_column_device_view{};
        size_type _element_index{};
};

/*
namespace detail {

class lists_column_device_view 
// TODO: Required for recursion.
// : private column_device_view 
{
    public:

        lists_column_device_view() = delete;

        ~lists_column_device_view() = default;
        lists_column_device_view(lists_column_device_view const&) = default;
        lists_column_device_view(lists_column_device_view &&) = default;
        
        lists_column_device_view(
            column_device_view const& child,    // Holds data.
            column_device_view const& offsets   // Holds list offsets.
        ) 
        : d_child(child), d_offsets(offsets)
        {}

        cudf::list_view operator[](size_type idx) const
        {
            return cudf::list_view{this, idx};
        }

    private:

        column_device_view d_child;
        column_device_view d_offsets;
};

} // namespace detail;

__device__ bool list_view::operator == (list_view const& rhs) const
{
    printf("CALEB: list_view::operator ==()!\n");

    return false;
}
*/


}  // namespace cudf
