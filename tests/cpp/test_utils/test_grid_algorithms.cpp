/* Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
 * 
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 * 
 *        http://www.apache.org/licenses/LICENSE-2.0
 * 
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */


// Include Libraries
#include <gtest/gtest.h>
#include <qdecomp/utils/grid_problem/cpp/grid_algorithms.cpp>


// Tested intervals
#define INTERVAL_TYPE std::tuple<long double, long double>
const std::vector<INTERVAL_TYPE> INTERVALS = {
    {0, 1},
    {-1, 0},
    {-0.8, 0},
    {0, 0.35},
    {-0.35, 0},
    {-1, 1},
    {-10, 10},
    {-47.9, 12.4},
    {33.33, 44.44},
    {-9, 2.00000000000001}
};
const std::vector<INTERVAL_TYPE> FEWER_INTERVALS = {
    {0, 1},
    {-0.8, 0},
    {0, 0.35},
    {-10, 10},
    {-47.9, 12.4},
    {-9, 2.00000000000001}
};


// Test 1D grid algorithm's solutions
class GridAlgorithms1DTests : public ::testing::TestWithParam<std::tuple<INTERVAL_TYPE, INTERVAL_TYPE>> {};

TEST_P(GridAlgorithms1DTests, TestSolutions) {
    auto [a_int, b_int] = GetParam();
    auto [a_min, a_max] = a_int;
    auto [b_min, b_max] = b_int;

    auto solutions = GridProblem1D(a_min, a_max, b_min, b_max);

    for (auto sol : solutions){
        EXPECT_GE(sol.to_long_double(), a_min);
        EXPECT_LE(sol.to_long_double(), a_max);
        EXPECT_GE(sol.sqrt2_conjugate().to_long_double(), b_min);
        EXPECT_LE(sol.sqrt2_conjugate().to_long_double(), b_max);
    }
}

INSTANTIATE_TEST_SUITE_P(
    GridAlgorithmsTests,
    GridAlgorithms1DTests,
    ::testing::Combine(
        ::testing::ValuesIn(INTERVALS),
        ::testing::ValuesIn(INTERVALS)
    )
);


// Test 2D grid algorithm's solutions
class GridAlgorithms2DTests : public ::testing::TestWithParam<
    std::tuple<INTERVAL_TYPE, INTERVAL_TYPE, INTERVAL_TYPE, INTERVAL_TYPE>
> {};

TEST_P(GridAlgorithms2DTests, TestSolutions) {
    auto [ax_int, ay_int, bx_int, by_int] = GetParam();
    auto [ax_min, ax_max] = ax_int;
    auto [ay_min, ay_max] = ay_int;
    auto [bx_min, bx_max] = bx_int;
    auto [by_min, by_max] = by_int;

    auto solutions = GridProblem2D(
        ax_min, ax_max, bx_min, bx_max,
        ay_min, ay_max, by_min, by_max
    );

    for (auto sol : solutions){
        EXPECT_GE(sol.to_Domega().real().to_long_double(), ax_min);
        EXPECT_LE(sol.to_Domega().real().to_long_double(), ax_max);
        EXPECT_GE(sol.to_Domega().imag().to_long_double(), ay_min);
        EXPECT_LE(sol.to_Domega().imag().to_long_double(), ay_max);

        EXPECT_GE(sol.to_Domega().real().sqrt2_conjugate().to_long_double(), bx_min);
        EXPECT_LE(sol.to_Domega().real().sqrt2_conjugate().to_long_double(), bx_max);
        EXPECT_GE(sol.to_Domega().imag().sqrt2_conjugate().to_long_double(), by_min);
        EXPECT_LE(sol.to_Domega().imag().sqrt2_conjugate().to_long_double(), by_max);
    }
}

INSTANTIATE_TEST_SUITE_P(
    GridAlgorithmsTests,
    GridAlgorithms2DTests,
    ::testing::Combine(
        ::testing::ValuesIn(INTERVALS),
        ::testing::ValuesIn(FEWER_INTERVALS),
        ::testing::ValuesIn(FEWER_INTERVALS),
        ::testing::ValuesIn(INTERVALS)
    )
);
