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

#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>

namespace cudf {
namespace {

template <typename, typename, typename = void>
struct is_range_scalable : std::false_type {};

template <typename From, typename To>
struct is_range_scalable<From, // Range Type.
                         To,   // OrderBy Type.
                         std::enable_if_t<    cudf::is_duration<From>() 
                                           && cudf::is_timestamp<To>(), void >> 
{
    using destination_duration = typename To::duration;
    using destination_period   = typename destination_duration::period;
    using source_period        = typename From::period;
    static constexpr bool value = cuda::std::ratio_less_equal<destination_period, source_period>::value;
};

template <typename OrderByColumnType>
struct range_scaler // A scalar_scaler, if you will.
{
  // SFINAE catch-all.
  template <typename RangeType, typename... Args>
  std::enable_if_t<!is_range_scalable<RangeType, OrderByColumnType>::value, 
    std::unique_ptr<scalar>> operator()(Args&&... args)
  {
    CUDF_FAIL("Unsupported RangeType!");
  }

  template <typename RangeType,
            std::enable_if_t<  is_timestamp<OrderByColumnType>()
                            && is_duration<RangeType>()
                            && is_range_scalable<RangeType, OrderByColumnType>::value, void> * = nullptr>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar,
                                     bool is_unbounded_range,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using order_by_column_duration_t = typename OrderByColumnType::duration;
    using rep_t = typename order_by_column_duration_t::rep;

    auto const& range_scalar_duration = static_cast< cudf::duration_scalar<RangeType> const& >(range_scalar);
    return std::unique_ptr<scalar>{
             new cudf::duration_scalar<order_by_column_duration_t>{
                 is_unbounded_range
                   ? order_by_column_duration_t{std::numeric_limits<rep_t>::max()} 
                   : order_by_column_duration_t{range_scalar_duration.value()},
                 true}};
  }
};

struct type_deducing_range_scaler
{
  template <typename OrderByColumnType>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar,
                                     bool is_unbounded_range,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    return cudf::type_dispatcher(range_scalar.type(),
                                 range_scaler<OrderByColumnType>{},
                                 range_scalar,
                                 is_unbounded_range,
                                 stream,
                                 mr);
  }
};

} // namespace;

void range_window_bounds::scale_to(data_type target_type, 
                                   rmm::cuda_stream_view stream, 
                                   rmm::mr::device_memory_resource* mr)
{
    scalar const& range_scalar = *_value;

    _value = std::move(cudf::type_dispatcher(target_type,
                                             type_deducing_range_scaler{},
                                             range_scalar,
                                             _is_unbounded,
                                             stream,
                                             mr));
    assert_invariants();
}

range_window_bounds range_window_bounds::unbounded(data_type type)
{
    return range_window_bounds(true, make_default_constructed_scalar(type));
}

} // namespace cudf;
