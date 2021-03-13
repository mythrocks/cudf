#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <src/rolling/rolling_detail.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <chrono>
#include <cuda/std/ratio>
#include <vector>

// TODO: Add to rolling.hpp.
#include <cudf/rolling/range_window_bounds.hpp>

namespace cudf
{
namespace test
{

struct RangeWindowBoundsTest : public BaseFixture {};

namespace 
{
    template <typename ScalarType, 
              typename value_type = typename ScalarType::value_type,
              std::enable_if_t< cudf::is_duration<value_type>(), void >* = nullptr>
    auto range_bounds(ScalarType const& scalar)
    {
        return range_window_bounds::get(std::unique_ptr<ScalarType>{new ScalarType{scalar}});
    }

    /*
    template <typename ScalarType, 
              typename value_type = typename ScalarType::value_type,
              std::enable_if_t< cudf::is_duration<value_type>(), void >* = nullptr>
    auto get_range_comparable_value(ScalarType const& scalar)
    {
        return 
    }
    */

    /*
    template <typename T>
    T get_range_comparable_value(cudf::scalar const s)
    {
        return
    }
    */
}
 

TEST_F(RangeWindowBoundsTest, TimestampsAndDurations)
{
    using namespace cudf;

    {
      // Test that range_window_bounds specified in Days can be scaled down to seconds, milliseconds, etc.
      auto range_3_days = range_bounds(duration_scalar<duration_D>{3, true});
      EXPECT_TRUE(range_3_days.value().is_valid());
      EXPECT_TRUE(!range_3_days.is_unbounded());

      range_3_days.scale_to(data_type{type_id::TIMESTAMP_SECONDS});
      EXPECT_EQ(static_cast<duration_scalar<duration_s>const&>(range_3_days.value()).value().count(),
                3 * 24 * 60 * 60);

      // Unchanged.
      range_3_days.scale_to(data_type{type_id::TIMESTAMP_SECONDS});
      EXPECT_EQ(static_cast<duration_scalar<duration_s>const&>(range_3_days.value()).value().count(),
                3 * 24 * 60 * 60);

      // Finer.
      range_3_days.scale_to(data_type{type_id::TIMESTAMP_MILLISECONDS});
      EXPECT_EQ(static_cast<duration_scalar<duration_ms>const&>(range_3_days.value()).value().count(),
                3 * 24 * 60 * 60 * 1000);

      // Finer.
      range_3_days.scale_to(data_type{type_id::TIMESTAMP_MICROSECONDS});
      EXPECT_EQ(static_cast<duration_scalar<duration_us>const&>(range_3_days.value()).value().count(),
                int64_t{3} * 24 * 60 * 60 * 1000 * 1000);

      // Scale back up to days. Should fail because of loss of precision.
      EXPECT_THROW(range_3_days.scale_to(data_type{type_id::TIMESTAMP_DAYS}), cudf::logic_error);
    }

    {
      // Negative tests. Cannot scale from higher to lower precision.
      auto range_3M_ns = range_bounds(duration_scalar<duration_ns>{int64_t{3}*1000*1000, true});
      EXPECT_THROW(range_3M_ns.scale_to(data_type{type_id::TIMESTAMP_DAYS}), cudf::logic_error);
    }

    /*
    auto range = cudf::range_window_bounds::get(std::unique_ptr<scalar>{new cudf::duration_scalar<cudf::duration_s>{1, true}});
    std::cout << "Range.is_valid()? " << std::boolalpha << range.value().is_valid() << std::endl;

    auto unscaled_count = static_cast<cudf::duration_scalar<cudf::duration_D>const&>(range.value()).value().count();
    std::cout << "Unscaled count: " << unscaled_count << std::endl;

    range.scale_to(data_type{type_id::TIMESTAMP_MILLISECONDS});
    auto scaled_as_milliseconds = static_cast<cudf::duration_scalar<cudf::duration_ms>const&>(range.value()).value().count();
    std::cout << "Scaled count, as milliseconds: " << scaled_as_milliseconds << std::endl;

    auto unlimited_range = cudf::range_window_bounds::unbounded(data_type{type_id::DURATION_SECONDS});
    unlimited_range.scale_to(data_type{type_id::TIMESTAMP_MICROSECONDS});
    auto unlimited_range_as_milliseconds = static_cast<cudf::duration_scalar<cudf::duration_us>const&>(unlimited_range.value()).value().count();
    std::cout << "Scaled unlimited range count, as microseconds: " << unlimited_range_as_milliseconds << std::endl;
    */
}
 
TEST_F(RangeWindowBoundsTest, DELETEME)
{
    std::cout << "RangeWindowBoundsTest::Scalars!" << std::endl;

    {
        // auto duration_scalar = cudf::make_duration_scalar(cudf::data_type{cudf::type_id::DURATION_DAYS});
        auto one_day = cudf::duration_scalar<cudf::duration_D>{int32_t{1}, true};
        std::cout << "Num ticks in a day: " << one_day.count() << std::endl;

        auto seconds_in_a_day = cudf::duration_scalar<cudf::duration_s>{60*60*24, true};
        std::cout << "Num ticks in (seconds in a day): " << seconds_in_a_day.count() << std::endl;

        /*
        using namespace std::literals;
        auto now = std::chrono::system_clock::now();
        auto yesterday = std::chrono::system_clock::to_time_t(now - 24h);
        std::cout << "Time yesterday: " << std::put_time(std::localtime(&yesterday), "%F %T.\n") << std::endl;
        */

        auto day_timestamp = cudf::timestamp_scalar<cudf::timestamp_D>{one_day, true};
        std::cout << "Day timestamp has " << day_timestamp.ticks_since_epoch() << " ticks since epoch." << std::endl;
    }

    {
        auto duration_days = cudf::duration_D{5};
        std::cout << "Num ticks: " << duration_days.count() << std::endl;

        auto duration_seconds = cudf::duration_s{duration_days};
        std::cout << "Num ticks: " << duration_seconds.count() << std::endl;
    }

    {
        using days_duration = typename cudf::id_to_type<cudf::type_id::TIMESTAMP_DAYS>::duration; 
        using days_period = typename days_duration::period;
        using seconds_duration = typename cudf::id_to_type<cudf::type_id::TIMESTAMP_SECONDS>::duration; 
        using seconds_period = typename seconds_duration::period;
        
        std::cout << typeid(days_period).name() << std::endl;

        std::cout << "Ratio greater-equal? " << std::boolalpha << cuda::std::ratio_greater_equal<days_period, seconds_period>::value << std::endl;
        std::cout << "Reversed ratio greater-equal? " << std::boolalpha << cuda::std::ratio_greater_equal<seconds_period, days_period>::value << std::endl;
    }

    {
        using OrderByColumnType = cudf::timestamp_s;
        using RangeType  = cudf::duration_D;

        auto range = cudf::duration_scalar<RangeType>{1, true};
        std::cout << "Unscaled range-ticks: " << range.count() << std::endl;



        using OrderByDurationType = typename OrderByColumnType::duration;
        // auto scaled_range = cudf::duration_scalar<OrderByDurationType>{RangeType{range.value()}, true};
        auto scaled_range = cudf::duration_scalar<OrderByDurationType>{range.value(), true};
        std::cout << "Scaled range-ticks count: " << scaled_range.count() << std::endl;
    }
}

} // namespace test;
} // namespace cudf;
