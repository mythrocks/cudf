#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <tests/strings/utilities.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/tabulate.h>

#include <chrono>
#include <random>
#include <thread>
#include <vector>

struct ContainsREBorkTest : public cudf::test::BaseFixture {};

void foo(std::string const& msg)
{
    std::cout << "tid:" << std::this_thread::get_id() << ":: msg: " << msg << std::endl;
}

void test(long thread_no)
{
    auto constexpr num_rows = 166'666'667;
    // auto constexpr num_rows = 166'666'667/8; // Fails  compute sanitizer.
    // auto constexpr num_rows = 166'666'667/16;   // Passes compute sanitizer.

    auto input = [&] {
        auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64}, num_rows);
        auto begin = col->mutable_view().begin<int64_t>();
        auto end = col->mutable_view().end<int64_t>();
        thrust::tabulate(thrust::device, begin, end, 
                        [thread_no, num_rows]__device__(auto i) { return thread_no * num_rows + i; });
        return col;
    }();

    // Cast to STRING.
    auto to_string = cudf::strings::from_integers(input->view());
    auto strings_cv = cudf::strings_column_view{to_string->view()};

    // Find matches.
    auto matches = cudf::strings::contains_re(strings_cv, "(.|\\n)*1(.|\\n)0(.|\\n)*");

    // Count matches.
    auto num_matches = thrust::count_if(thrust::device, matches->view().begin<bool>(), matches->view().end<bool>(), 
        []__device__(auto is_true){ return is_true; });

    std::cout << "[" << thread_no << "][tid=" << std::this_thread::get_id() << "] num_matches: " << num_matches << std::endl;
}

TEST_F(ContainsREBorkTest, Test2Threads)
{
    std::cout << "Is cuda_stream_per_thread enabled? ";
    #ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    std::cout << "YES." << std::endl;
    #else 
    std::cout << "NO." << std::endl;
    #endif
    std::cout << "Is per_thread_default? " << std::boolalpha << cudf::default_stream_value.is_per_thread_default() << std::endl;
    std::cout << "\n" << std::endl;

    for (int i{0}; i<1; ++i) {
        std::cout << "Attempt#" << i << std::endl;
        auto thread0 = std::thread{test, 0};
        auto thread1 = std::thread{test, 1};
        thread0.join();
        thread1.join();
        std::cout << std::endl;
        // std::cout << "Sleeping 5..." << std::endl;
        // using namespace std::chrono_literals;
        // std::this_thread::sleep_for(5000ms);
    }
}