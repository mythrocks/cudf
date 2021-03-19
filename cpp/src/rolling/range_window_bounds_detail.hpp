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

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>

namespace cudf {
namespace detail {

template <typename RangeType>
constexpr bool is_supported_range_type()
{
    return cudf::is_duration<RangeType>()
        || (std::is_integral<RangeType>::value && !cudf::is_boolean<RangeType>());
}

template <typename ColumnType>
constexpr bool is_supported_order_by_column_type()
{
    return cudf::is_timestamp<ColumnType>()
        || (std::is_integral<ColumnType>::value && !cudf::is_boolean<ColumnType>());
    ;
}

template <typename From, typename To, typename = void>
struct is_range_scalable : std::false_type {};

template <typename From, typename To>
struct is_range_scalable<From, // Range Type.
                         To,   // OrderBy Type.
                         std::enable_if_t<    cudf::is_duration<From>() 
                                           && cudf::is_timestamp<To>(), void >> 
{
    using to_duration = typename To::duration;
    using to_period   = typename to_duration::period;
    using from_period = typename From::period;
    static constexpr bool value = cuda::std::ratio_less_equal<to_period, from_period>::value;
};

template <typename From>
struct is_range_scalable<From,
                         From,
                         std::enable_if_t< std::is_integral<From>::value
                                        && !cudf::is_boolean<From>(), void>>
: std::true_type{};

template <typename OrderByColumnType>
struct range_scaler // A scalar_scaler, if you will.
{
  // SFINAE catch-all.
  template <typename RangeType, typename... Args>
  std::enable_if_t<!is_range_scalable<RangeType, OrderByColumnType>::value, 
    std::unique_ptr<scalar>> operator()(Args&&... args)
  {
    CUDF_FAIL("Unsupported range type for order by column!");
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
                   : order_by_column_duration_t{range_scalar_duration.value(stream)},
                 true,
                 stream,
                 mr}};
  }

  template <typename RangeType,
            std::enable_if_t<  std::is_same<OrderByColumnType, RangeType>::value
                            && std::is_integral<RangeType>::value
                            && !cudf::is_boolean<RangeType>(), void> * = nullptr>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar,
                                     bool is_unbounded_range,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using numeric_scalar = cudf::numeric_scalar<RangeType>;

    return std::unique_ptr<scalar>{new numeric_scalar {
      is_unbounded_range
        ? std::numeric_limits<RangeType>::max()
        : static_cast<numeric_scalar const&>(range_scalar).value(stream),
      true,
      stream,
      mr}};
  }
};

namespace
{
template <typename RepType>
struct range_comparable_value_fetcher
{
    template <typename RangeType, typename... Args>
    std::enable_if_t< !is_supported_range_type<RangeType>(), RepType >
    operator()(Args&&...) const 
    {
        CUDF_FAIL("Unsupported window range type!");
    }

    template <typename RangeType>
    std::enable_if_t< std::is_integral<RangeType>::value && !cudf::is_boolean<RangeType>(), RepType >
    operator()(scalar const& range_scalar, rmm::cuda_stream_view stream) const
    {
        return static_cast<numeric_scalar<RangeType>const&>(range_scalar).value(stream);
    }    
    
    template <typename RangeType>
    std::enable_if_t< cudf::is_duration<RangeType>(), RepType >
    operator()(scalar const& range_scalar, rmm::cuda_stream_view stream) const
    {
        return static_cast<duration_scalar<RangeType>const&>(range_scalar).value(stream).count();
    }
};

template <typename RepType>
bool rep_type_compatible_for_range_comparison(type_id id)
{
    return (id == type_id::DURATION_DAYS         && std::is_same<RepType, int32_t>())
        || (id == type_id::DURATION_SECONDS      && std::is_same<RepType, int64_t>())
        || (id == type_id::DURATION_MILLISECONDS && std::is_same<RepType, int64_t>())
        || (id == type_id::DURATION_MICROSECONDS && std::is_same<RepType, int64_t>())
        || (id == type_id::DURATION_NANOSECONDS  && std::is_same<RepType, int64_t>())
        || type_id_matches_device_storage_type<RepType>(id);
};

template <typename T, 
          std::enable_if_t<std::numeric_limits<T>::is_signed, void>* = nullptr>
void assert_non_negative(T const& value)
{
  CUDF_EXPECTS(value >= T{0}, "Range scalar must be >= 0.");
}

template <typename T, 
          std::enable_if_t<!std::numeric_limits<T>::is_signed, void>* = nullptr>
void assert_non_negative(T const& value)
{
  // Unsigned values are non-negative.
}

} // namespace <unnamed>;

template <typename RepType>
RepType range_comparable_value(range_window_bounds const& range_bounds,
                               rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
    auto const& range_scalar = range_bounds.range_scalar();
    CUDF_EXPECTS(rep_type_compatible_for_range_comparison<RepType>(range_scalar.type().id()), 
                    "Data type of window range scalar does not match output type.");
    auto comparable_value = cudf::type_dispatcher(range_scalar.type(),
                                 range_comparable_value_fetcher<RepType>{},
                                 range_scalar,
                                 stream);
    assert_non_negative(comparable_value);
    return comparable_value;
}

template <typename ScalarType, 
          typename value_type = typename ScalarType::value_type,
          std::enable_if_t<  is_supported_range_type<value_type>(), void >* = nullptr>
auto range_bounds(ScalarType const& scalar)
{
    return range_window_bounds::get(std::unique_ptr<ScalarType>{new ScalarType{scalar}});
}

} // namespace detail;
} // namespace cudf;
