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

struct NonTimestampRangeTest : public BaseFixture {};

/*
template <typename, typename, typename = void>
struct is_range_scalable_2 : std::false_type {};

template <typename From, typename To>
struct is_range_scalable_2<From, // Range Type.
                           To,   // OrderBy Type.
                           std::enable_if_t<    cudf::is_duration<From>() 
                                             && cudf::is_timestamp<To>(), void >> 
{
    using destination_duration = typename To::duration;
    using destination_period   = typename destination_duration::period;
    using source_period        = typename From::period;
    static constexpr bool value = cuda::std::ratio_less_equal<destination_period, source_period>::value;
};

TEST_F(NonTimestampRangeTest, SFINAETest)
{
    std::cout << std::boolalpha 
              << is_range_scalable_2<cudf::duration_D, cudf::timestamp_s>::value << std::endl
              << is_range_scalable_2<cudf::duration_s, cudf::timestamp_D>::value << std::endl
              << is_range_scalable_2<bool, bool>::value 
              << std::endl;
}
*/

TEST_F(NonTimestampRangeTest, RangeWindowTest)
{
    std::cout << "NonTimestampRangeTest::RangeWindowTest!" << std::endl;

    using namespace cudf;

    auto range = cudf::range_window_bounds::get(std::unique_ptr<scalar>{new cudf::duration_scalar<cudf::duration_D>{int32_t{1}, true}});
    std::cout << "Range.is_valid()? " << std::boolalpha << range.value().is_valid() << std::endl;

    auto unscaled_count = static_cast<cudf::duration_scalar<cudf::duration_D>const&>(range.value()).value().count();
    std::cout << "Unscaled count: " << unscaled_count << std::endl;

    range.scale_to(data_type{type_id::TIMESTAMP_SECONDS});
    auto scaled_as_seconds = static_cast<cudf::duration_scalar<cudf::duration_s>const&>(range.value()).value().count();
    std::cout << "Scaled count, as seconds: " << scaled_as_seconds << std::endl;
    auto mistakenly_scaled = static_cast<cudf::duration_scalar<cudf::duration_D>const&>(range.value()).value().count();
    std::cout << "Scaled count, as days: "    << mistakenly_scaled << std::endl;
    auto also_mistakenly_scaled = static_cast<cudf::detail::fixed_width_scalar<cudf::duration_D>const&>(range.value()).value().count();
    std::cout << "Scaled count, as fixed_wd: "<< also_mistakenly_scaled << std::endl;
}
 
TEST_F(NonTimestampRangeTest, Scalars)
{
    std::cout << "NonTimestampRangeTest::Scalars!" << std::endl;

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

}
}
