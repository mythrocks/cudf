/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>

namespace cudf {
namespace detail {

/// Checks if the specified type is supported in a range_window_bounds.
template <typename RangeType>
constexpr bool is_supported_range_type()
{
  return cudf::is_duration<RangeType>() ||
         (std::is_integral<RangeType>::value && !cudf::is_boolean<RangeType>());
}

/// Checks if the specified type is a supported target type,
/// as an orderby column, for comparisons with a range_window_bounds scalar.
template <typename ColumnType>
constexpr bool is_supported_order_by_column_type()
{
  return cudf::is_timestamp<ColumnType>() ||
         (std::is_integral<ColumnType>::value && !cudf::is_boolean<ColumnType>());
  ;
}

/// Checks if a range bounds scalar of type `Range` has the same resolution
/// as an orderby column of type `OrderBy`.
template <typename Range, typename OrderBy, typename = void>
struct is_matching_resolution : std::false_type {
};

/// Checks if a duration range bounds scalar of type `Range` has the same resolution
/// as a timestamp orderby column of type `OrderBy`.
template <typename Range, typename OrderBy>
struct is_matching_resolution<
  Range,  
  OrderBy,
  std::enable_if_t<cudf::is_duration<Range>() && cudf::is_timestamp<OrderBy>(), void>> {
  using oby_duration          = typename OrderBy::duration;
  using oby_period            = typename oby_duration::period;
  using range_period          = typename Range::period;
  static constexpr bool value = cuda::std::ratio_equal<oby_period, range_period>::value;
};

/// Integral range scalars can only be used with orderby columns of exactly the same type.
template <typename T>
struct is_matching_resolution<
  T,
  T,
  std::enable_if_t<std::is_integral<T>::value && !cudf::is_boolean<T>(), void>>
  : std::true_type {
};

/* DELETEME!
template <typename OrderByColumnType>
struct range_scaler  // A scalar_scaler, if you will.
{
  // SFINAE catch-all.
  template <typename RangeType, typename... Args>
  std::enable_if_t<!is_range_scalable<RangeType, OrderByColumnType>::value, std::unique_ptr<scalar>>
  operator()(Args&&... args)
  {
    CUDF_FAIL("Unsupported range type for order by column!");
  }

  template <typename RangeType,
            std::enable_if_t<is_timestamp<OrderByColumnType>() && is_duration<RangeType>() &&
                               is_range_scalable<RangeType, OrderByColumnType>::value,
                             void>* = nullptr>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar,
                                     bool is_unbounded_range,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using order_by_column_duration_t = typename OrderByColumnType::duration;
    using rep_t                      = typename order_by_column_duration_t::rep;

    auto const& range_scalar_duration =
      static_cast<cudf::duration_scalar<RangeType> const&>(range_scalar);
    return std::unique_ptr<scalar>{new cudf::duration_scalar<order_by_column_duration_t>{
      is_unbounded_range ? order_by_column_duration_t{std::numeric_limits<rep_t>::max()}
                         : order_by_column_duration_t{range_scalar_duration.value(stream)},
      true,
      stream,
      mr}};
  }

  template <typename RangeType,
            std::enable_if_t<std::is_same<OrderByColumnType, RangeType>::value &&
                               std::is_integral<RangeType>::value && !cudf::is_boolean<RangeType>(),
                             void>* = nullptr>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar,
                                     bool is_unbounded_range,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using numeric_scalar = cudf::numeric_scalar<RangeType>;

    return std::unique_ptr<scalar>{new numeric_scalar{
      is_unbounded_range ? std::numeric_limits<RangeType>::max()
                         : static_cast<numeric_scalar const&>(range_scalar).value(stream),
      true,
      stream,
      mr}};
  }
};
*/

namespace {
template <typename RepType>
struct range_comparable_value_fetcher {
  template <typename RangeType, typename... Args>
  std::enable_if_t<!is_supported_range_type<RangeType>(), RepType> operator()(Args&&...) const
  {
    CUDF_FAIL("Unsupported window range type!");
  }

  template <typename RangeType>
  std::enable_if_t<std::is_integral<RangeType>::value && !cudf::is_boolean<RangeType>(), RepType>
  operator()(scalar const& range_scalar, rmm::cuda_stream_view stream) const
  {
    return static_cast<numeric_scalar<RangeType> const&>(range_scalar).value(stream);
  }

  template <typename RangeType>
  std::enable_if_t<cudf::is_duration<RangeType>(), RepType> operator()(
    scalar const& range_scalar, rmm::cuda_stream_view stream) const
  {
    return static_cast<duration_scalar<RangeType> const&>(range_scalar).value(stream).count();
  }
};

template <typename RepType>
bool rep_type_compatible_for_range_comparison(type_id id)
{
  return (id == type_id::DURATION_DAYS && std::is_same<RepType, int32_t>()) ||
         (id == type_id::DURATION_SECONDS && std::is_same<RepType, int64_t>()) ||
         (id == type_id::DURATION_MILLISECONDS && std::is_same<RepType, int64_t>()) ||
         (id == type_id::DURATION_MICROSECONDS && std::is_same<RepType, int64_t>()) ||
         (id == type_id::DURATION_NANOSECONDS && std::is_same<RepType, int64_t>()) ||
         type_id_matches_device_storage_type<RepType>(id);
};

template <typename T, std::enable_if_t<std::numeric_limits<T>::is_signed, void>* = nullptr>
void assert_non_negative(T const& value)
{
  CUDF_EXPECTS(value >= T{0}, "Range scalar must be >= 0.");
}

template <typename T, std::enable_if_t<!std::numeric_limits<T>::is_signed, void>* = nullptr>
void assert_non_negative(T const& value)
{
  // Unsigned values are non-negative.
}

/// Helper to check that the range-type mathes the orderby column type,
/// in resolution.
template <typename OrderByType>
struct range_resolution_checker
{
  template <typename RangeType,
            std::enable_if_t< detail::is_matching_resolution<RangeType, OrderByType>::value,
                              void >* = nullptr >
  void operator()() const
  {}

  template <typename RangeType,
            std::enable_if_t< !detail::is_matching_resolution<RangeType, OrderByType>::value,
                              void >* = nullptr >
  void operator()() const
  {
    CUDF_FAIL("Range type resolution must exactly match that of the OrderBy type.");
  }
};

}  // namespace

/**
 * @brief Fetch the value of the range_window_bounds scalar, for comparisons
 *        with an orderby column's rows.
 *
 * @tparam RepType The output type for the range scalar
 * @param range_bounds The range_window_bounds whose value is to be read
 * @param stream The CUDA stream for device memory operations
 * @return RepType Value of the range scalar
 */
template <typename RepType>
RepType range_comparable_value(range_window_bounds const& range_bounds,
                               rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  auto const& range_scalar = range_bounds.range_scalar();
  CUDF_EXPECTS(rep_type_compatible_for_range_comparison<RepType>(range_scalar.type().id()),
               "Data type of window range scalar does not match output type.");
  auto comparable_value = cudf::type_dispatcher(
    range_scalar.type(), range_comparable_value_fetcher<RepType>{}, range_scalar, stream);
  assert_non_negative(comparable_value);
  return comparable_value;
}

template <typename OrderByType>
void assert_matching_resolution(range_window_bounds const& range_bounds)
{
  cudf::type_dispatcher(range_bounds.range_scalar().type(),
                        range_resolution_checker<OrderByType>{});
}

}  // namespace detail
}  // namespace cudf
