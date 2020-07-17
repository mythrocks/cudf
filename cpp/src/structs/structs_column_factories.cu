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
#include "cudf/types.hpp"
#include "cudf/utilities/error.hpp"
#include "rmm/device_buffer.hpp"

namespace cudf
{

  std::unique_ptr<cudf::column> make_structs_column(
    size_type num_rows,
    std::vector<std::unique_ptr<column>> child_columns,
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
