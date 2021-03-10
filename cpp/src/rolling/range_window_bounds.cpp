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
struct range_scaler // scalar_scaler;
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
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using order_by_column_duration = typename OrderByColumnType::duration;
    auto const& range_scalar_duration = static_cast< cudf::duration_scalar<RangeType> const& >(range_scalar);
    return std::unique_ptr<scalar>{new cudf::duration_scalar<order_by_column_duration>{range_scalar_duration.value(), true}};
  }
};

struct order_by_type_deducer
{
  template <typename OrderByColumnType>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    return cudf::type_dispatcher(range_scalar.type(),
                                 range_scaler<OrderByColumnType>{},
                                 range_scalar,
                                 stream,
                                 mr);
  }
};

} // namespace;

void range_window_bounds::scale_to(data_type target_type, 
                                   rmm::cuda_stream_view stream, 
                                   rmm::mr::device_memory_resource* mr)
{
    std::cout << "Scaling from " << static_cast<int32_t>(_value->type().id()) 
              << " to " << static_cast<int32_t>(target_type.id()) << std::endl;

    if (_is_unbounded) {
        std::cout << "Unbounded window. No scaling required!" << std::endl;
        // TODO: Must rewrite with "appropriate" default for target_type. Can fetch from range_scaler.
        return;
    }

    scalar const& range_scalar = *_value;

    _value = std::move(cudf::type_dispatcher(target_type,
                                 order_by_type_deducer{},
                                 range_scalar,
                                 stream,
                                 mr));
}

} // namespace cudf;
