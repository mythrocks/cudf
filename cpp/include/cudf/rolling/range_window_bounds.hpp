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

#pragma once

#include <cudf/scalar/scalar.hpp>

namespace cudf
{

struct range_window_bounds {
 public:

  /**
   * @brief Construct bounded window boundary.
   *
   * @param value Finite window boundary 
   * 
   */
  static range_window_bounds get(std::unique_ptr<scalar>&& value) // TODO: Make fixed_width_scalar?
  { return range_window_bounds(false, std::move(value)); }

  /**
   * @brief Construct unbounded window boundary.
   *
   * @return window_bounds
   */
  static range_window_bounds unbounded(data_type type);

  bool is_unbounded() const { return _is_unbounded; }

  scalar const& value() const { return *_value; }

  /**
   * @brief Rescale underlying scalar.
   * 
   * @param target_type 
   */
  void scale_to(data_type target_type, 
                rmm::cuda_stream_view stream = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

 private:

  const bool _is_unbounded{false};
  std::unique_ptr<scalar> _value{nullptr};

  range_window_bounds(bool is_unbounded_, std::unique_ptr<scalar>&& value_)
    : _is_unbounded{is_unbounded_}, _value{std::move(value_)}
  {
    assert_invariants();
  }

  void assert_invariants() const
  {
    CUDF_EXPECTS(_value.get(), "Range window scalar cannot be null.");
    CUDF_EXPECTS(_is_unbounded || _value->is_valid(), 
                 "Bounded Range window scalar must be valid.");
  }
};

} // namespace cudf;
