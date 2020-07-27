/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include "cudf/null_mask.hpp"
#include "cudf/types.hpp"
#include "cudf/utilities/error.hpp"
#include "cudf/utilities/bit.hpp"
#include "rmm/device_buffer.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"
#include <cudf/detail/utilities/cuda.cuh>

namespace cudf { namespace { 
// Whacked wholesale from null_mask.cu. TODO: Put that thing back where it came from, or so help meee.
/**
 * @brief Convenience function to get offset word from a bitmask
 *
 * @see copy_offset_bitmask
 * @see offset_bitmask_and
 */
__device__ bitmask_type get_mask_offset_word(bitmask_type const *__restrict__ source,
                                             size_type destination_word_index,
                                             size_type source_begin_bit,
                                             size_type source_end_bit)
{
  size_type source_word_index = destination_word_index + word_index(source_begin_bit);
  bitmask_type curr_word      = source[source_word_index];
  bitmask_type next_word      = 0;
  if (word_index(source_end_bit) >
      word_index(source_begin_bit +
                 destination_word_index * detail::size_in_bits<bitmask_type>())) {
    next_word = source[source_word_index + 1];
  }
  return __funnelshift_r(curr_word, next_word, source_begin_bit);
}

/**
 * @brief Computes the bitwise AND of an array of bitmasks
 *
 * @param destination The bitmask to write result into
 * @param source Array of source mask pointers. All masks must be of same size
 * @param begin_bit Array of offsets into corresponding @p source masks.
 *                  Must be same size as source array
 * @param num_sources Number of masks in @p source array
 * @param source_size Number of bits in each mask in @p source
 * @param number_of_mask_words The number of words of type bitmask_type to copy
 */
__global__ void offset_bitmask_and(bitmask_type *__restrict__ destination,
                                   bitmask_type const *const *__restrict__ source,
                                   size_type const *__restrict__ begin_bit,
                                   size_type num_sources,
                                   size_type source_size,
                                   size_type number_of_mask_words)
{
  for (size_type destination_word_index = threadIdx.x + blockIdx.x * blockDim.x;
       destination_word_index < number_of_mask_words;
       destination_word_index += blockDim.x * gridDim.x) {
    bitmask_type destination_word = ~bitmask_type{0};  // All bits 1
    for (size_type i = 0; i < num_sources; i++) {
      destination_word &= get_mask_offset_word(
        source[i], destination_word_index, begin_bit[i], begin_bit[i] + source_size);
    }

    destination[destination_word_index] = destination_word;
  }
}

  // Bitwise AND of the masks
rmm::device_buffer bitmask_and(std::vector<bitmask_type const *> const &masks,
                               std::vector<size_type> const &begin_bits,
                               size_type mask_size,
                               cudaStream_t stream,
                               rmm::mr::device_memory_resource *mr)
{
  CUDF_EXPECTS(std::all_of(begin_bits.begin(), begin_bits.end(), [](auto b) { return b >= 0; }),
               "Invalid range.");
  CUDF_EXPECTS(mask_size > 0, "Invalid bit range.");
  CUDF_EXPECTS(std::all_of(masks.begin(), masks.end(), [](auto p) { return p != nullptr; }),
               "Mask pointer cannot be null");

  rmm::device_buffer dest_mask{};
  auto num_bytes = bitmask_allocation_size_bytes(mask_size);

  auto number_of_mask_words = num_bitmask_words(mask_size);

  dest_mask = rmm::device_buffer{num_bytes, stream, mr};

  rmm::device_vector<bitmask_type const *> d_masks(masks);
  rmm::device_vector<size_type> d_begin_bits(begin_bits);

  cudf::detail::grid_1d config(number_of_mask_words, 256);
  offset_bitmask_and<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
    static_cast<bitmask_type *>(dest_mask.data()),
    d_masks.data().get(),
    d_begin_bits.data().get(),
    d_masks.size(),
    mask_size,
    number_of_mask_words);

  CHECK_CUDA(stream);

  return dest_mask;
}

}}

#include <iostream>

namespace cudf
{

  namespace 
  {
    // Helper function to superimpose validity of parent struct
    // over all member fields (i.e. child columns).
    void superimpose_validity(
      rmm::device_buffer const& parent_null_mask, 
      size_type parent_null_count,
      std::vector<std::unique_ptr<column>>& children,
      cudaStream_t stream,
      rmm::mr::device_memory_resource* mr
    )
    {
      if (parent_null_mask.is_empty()) {
        // Struct is not nullable. Children do not need adjustment.
        // Bail.
        return;
      }

      std::for_each(
        children.begin(),
        children.end(),
        [&](std::unique_ptr<column>& p_child)
        {
          if (!p_child->nullable())
          {
            std::cout << "CALEB: Child is nullable!\n";
            p_child->set_null_mask(std::move(rmm::device_buffer{parent_null_mask, stream, mr})); 
            p_child->set_null_count(parent_null_count);
          }
          else {

            auto data_type{p_child->type()};
            auto num_rows{p_child->size()};

            std::cout << "CALEB: For column of type " << static_cast<int>(data_type.id()) << std::endl;

            // All this to reset the null mask. :/
            cudf::column::contents contents{p_child->release()};
            std::vector<bitmask_type const*> masks {
              reinterpret_cast<bitmask_type const*>(parent_null_mask.data()), 
              reinterpret_cast<bitmask_type const*>(contents.null_mask->data())};
            
            std::cout << "CALEB: Parent null mask: " << (parent_null_mask.data() == nullptr? "NULL" : "NON_NULL") << std::endl;
            std::cout << "CALEB: Child  null mask: " << (contents.null_mask->data() == nullptr? "NULL" : "NON_NULL") << std::endl;
            
            rmm::device_buffer new_child_mask = bitmask_and(masks, {0, 0}, num_rows, stream, mr);

            // Recurse for struct members.
            // Push down recomputed child mask to child columns of the current child.
            if (data_type.id() == cudf::type_id::STRUCT)
            {
              superimpose_validity(new_child_mask, UNKNOWN_NULL_COUNT, contents.children, stream, mr);
            }

            // Reconstitute the column.
            p_child.reset(
              new column(
                data_type,
                num_rows,
                std::move(*contents.data),
                std::move(new_child_mask),
                UNKNOWN_NULL_COUNT,
                std::move(contents.children)
              )
            );
          }
        }
      );
    }
  }

  /// Column factory that adopts child columns.
  std::unique_ptr<cudf::column> make_structs_column(
    size_type num_rows,
    std::vector<std::unique_ptr<column>>&& child_columns,
    size_type null_count,
    rmm::device_buffer&& null_mask,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
  {
      if (null_count > 0)
      {
        CUDF_EXPECTS(!null_mask.is_empty(), "Column with nulls must be nullable.");
      }

      CUDF_EXPECTS(std::all_of(child_columns.begin(), child_columns.end(), [&](auto const& i){return num_rows == i->size();}), 
        "Child columns must have the same number of rows as the Struct column.");

      superimpose_validity(null_mask, null_count, child_columns, stream, mr);

      return std::make_unique<column>(
        cudf::data_type{type_id::STRUCT},
        num_rows,
        rmm::device_buffer{0, stream, mr}, // Empty data buffer. Structs hold no data.
        null_mask,
        null_count,
        std::move(child_columns)
      );
  }

} // namespace cudf;
