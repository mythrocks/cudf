/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <tests/strings/utilities.h>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/lists_column_view.hpp>

template <typename T>
class TypedScatterListsTest : public cudf::test::BaseFixture {
};
using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::DurationTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(TypedScatterListsTest, FixedWidthTypesNotBool);

class ScatterListsTest : public cudf::test::BaseFixture {
};

TEST_F(ScatterListsTest, ListsOfNullableFixedWidth)
{
    using namespace cudf::test;

    auto src_ints = fixed_width_column_wrapper<int32_t> {
        {9, 9, 9, 9, 8, 8, 8},
        {1, 1, 1, 0, 1, 1, 1}
    };

    auto p_src_list_column = cudf::make_lists_column(
        2,
        fixed_width_column_wrapper<cudf::size_type>{0, 4, 7}.release(),
        src_ints.release(),
        0,
        {}
    );
    auto src_list_column = *p_src_list_column;

    std::cout << "Scatter source: " << std::endl;
    print(src_list_column);

    auto target_list_column = lists_column_wrapper<int32_t>{
        {1, 1}, {2, 2}, {3, 3}
    };

    std::cout << "Scatter target: " << std::endl;
    print(target_list_column);

    auto scatter_map = fixed_width_column_wrapper<int32_t>{2, 0};

    std::cout << "Scatter map: " << std::endl;
    print(scatter_map);

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column}),
        scatter_map,
        cudf::table_view({target_list_column}));

    std::cout << "Scatter result: " << std::endl;
    print(ret->get_column(0));
}

TEST_F(ScatterListsTest, ListsOfFixedWidth)
{
    using namespace cudf::test;

    auto src_list_column = lists_column_wrapper<int32_t>{
        {9, 9, 9, 9}, {8, 8, 8}
    };

    std::cout << "Scatter source: " << std::endl;
    print(src_list_column);

    auto target_list_column = lists_column_wrapper<int32_t>{
        {1, 1}, {2, 2}, {3, 3}
    };

    std::cout << "Scatter target: " << std::endl;
    print(target_list_column);

    auto scatter_map = fixed_width_column_wrapper<int32_t>{2, 0};

    std::cout << "Scatter map: " << std::endl;
    print(scatter_map);

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column}),
        scatter_map,
        cudf::table_view({target_list_column}));

    std::cout << "Scatter result: " << std::endl;
    print(ret->get_column(0));
}

TEST_F(ScatterListsTest, ListsOfStrings)
{
    using namespace cudf::test;

    auto src_list_column = lists_column_wrapper<cudf::string_view> {
        {"all", "the", "leaves", "are", "brown"},
        {"california", "dreaming"}
    };

    auto target_list_column = lists_column_wrapper<cudf::string_view> {
        {"zero"},
        {"one", "one"},
        {"two", "two"}
    };

    auto scatter_map = fixed_width_column_wrapper<int32_t>{2, 0};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column}),
        scatter_map,
        cudf::table_view({target_list_column})
    );

    print(ret->get_column(0));
}

TEST_F(ScatterListsTest, ListsOfLists)
{
    using namespace cudf::test;

    auto src_list_column = lists_column_wrapper<int32_t> {
        { {1,1,1,1}, {2,2,2,2} },
        { {3,3,3,3}, {4,4,4,4} }
    };

    auto target_list_column = lists_column_wrapper<int32_t> {
        { {9,9}, {8,8}, {7,7} },
        { {6,6}, {}, {4,4} },
        { {3,3}, {2,2}, {1,1} }
    };

    auto scatter_map = fixed_width_column_wrapper<int32_t>{2, 0};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column}),
        scatter_map,
        cudf::table_view({target_list_column})
    );

    print(ret->get_column(0));
}

TEST_F(ScatterListsTest, TestScatter)
{
    using namespace thrust;

    std::vector<int> source {9, 8};
    std::vector<int> target {1, 2, 3};
    std::vector<int> scatter_map {1,0,1};

    scatter(host, source.begin(), source.end(), scatter_map.begin(), target.begin());

    thrust::copy(target.begin(), target.end(), std::ostream_iterator<int>(std::cout, "\n"));
}