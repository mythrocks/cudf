#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/dictionary/update_keys.hpp>

using namespace cudf::test::iterators;

namespace cudf::test {

using ints = fixed_width_column_wrapper<int32_t>;

struct HashGroupByNthElementTest : cudf::test::BaseFixture {
};

TEST_F(HashGroupByNthElementTest, Min)
{
  std::cout << "CALEB: Testing HashGroupByNthElementTest!" << std::endl;
  std::cout << "Nuther print." << std::endl;

  auto const grouping_keys = ints{0, 0, 0, 0, 1, 1, 1, 1};
  auto const agg_values    = ints{{0, 1, 2, 3, 4, 5, 6, 7}, nulls_at({0, 1, 2, 3})};

  auto aggs = std::vector<std::unique_ptr<cudf::groupby_aggregation>>{};
  aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  auto agg_request =
    cudf::groupby::aggregation_request{.values = agg_values, .aggregations = std::move(aggs)};
  auto agg_requests = std::vector<cudf::groupby::aggregation_request>{};
  agg_requests.push_back(std::move(agg_request));

  auto [grouped, results] =
    cudf::groupby::groupby(table_view{{grouping_keys}}).aggregate(agg_requests);
  std::cout << "Results: " << std::endl;
  print(*results[0].results[0]);
  std::cout << "Grouping keys: " << std::endl;
  print(grouped->get_column(0));
}

TEST_F(HashGroupByNthElementTest, FirstWithNulls)
{
  std::cout << "CALEB: Testing HashGroupByNthElementTest!" << std::endl;
  std::cout << "Nuther print." << std::endl;

  auto const grouping_keys = ints{0, 0, 0, 0, 1, 1, 1, 1};
  auto const agg_values    = ints{{10, 20, 30, 40, 50, 60, 70, 80}, nulls_at({0, 1, 2, 3})};

  auto aggs = std::vector<std::unique_ptr<cudf::groupby_aggregation>>{};
  aggs.push_back(
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0, null_policy::EXCLUDE));
  auto agg_request =
    cudf::groupby::aggregation_request{.values = agg_values, .aggregations = std::move(aggs)};
  auto agg_requests = std::vector<cudf::groupby::aggregation_request>{};
  agg_requests.push_back(std::move(agg_request));

  auto [grouped, results] =
    cudf::groupby::groupby(table_view{{grouping_keys}}).aggregate(agg_requests);
  std::cout << "Results: " << std::endl;
  print(*results[0].results[0]);
  std::cout << "Grouping keys: " << std::endl;
  print(grouped->get_column(0));
}

TEST_F(HashGroupByNthElementTest, LastWithNulls)
{
  std::cout << "CALEB: Testing HashGroupByNthElementTest!" << std::endl;
  std::cout << "Nuther print." << std::endl;

  auto const grouping_keys = ints{0, 0, 0, 0, 1, 1, 1, 1};
  auto const agg_values    = ints{{10, 20, 30, 40, 50, 60, 70, 80}, nulls_at({3})};

  auto aggs = std::vector<std::unique_ptr<cudf::groupby_aggregation>>{};
  aggs.push_back(
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-1, null_policy::EXCLUDE));
  auto agg_request =
    cudf::groupby::aggregation_request{.values = agg_values, .aggregations = std::move(aggs)};
  auto agg_requests = std::vector<cudf::groupby::aggregation_request>{};
  agg_requests.push_back(std::move(agg_request));

  auto [grouped, results] =
    cudf::groupby::groupby(table_view{{grouping_keys}}).aggregate(agg_requests);
  std::cout << "Results: " << std::endl;
  print(*results[0].results[0]);
  std::cout << "Grouping keys: " << std::endl;
  print(grouped->get_column(0));
}

TEST_F(HashGroupByNthElementTest, LastWithNulls)
{
  std::cout << "CALEB: Testing HashGroupByNthElementTest!" << std::endl;
  std::cout << "Nuther print." << std::endl;

  auto const grouping_keys = ints{0, 0, 0, 0, 1, 1, 1, 1};
  auto const agg_values    = ints{{10, 20, 30, 40, 50, 60, 70, 80}, nulls_at({3})};

  auto aggs = std::vector<std::unique_ptr<cudf::groupby_aggregation>>{};
  aggs.push_back(
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-1, null_policy::EXCLUDE));
  auto agg_request =
    cudf::groupby::aggregation_request{.values = agg_values, .aggregations = std::move(aggs)};
  auto agg_requests = std::vector<cudf::groupby::aggregation_request>{};
  agg_requests.push_back(std::move(agg_request));

  auto [grouped, results] =
    cudf::groupby::groupby(table_view{{grouping_keys}}).aggregate(agg_requests);
  std::cout << "Results: " << std::endl;
  print(*results[0].results[0]);
  std::cout << "Grouping keys: " << std::endl;
  print(grouped->get_column(0));
}

}  // namespace cudf::test