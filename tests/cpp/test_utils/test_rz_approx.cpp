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
#include <vector>
#include <gtest/gtest.h>
#include <qdecomp/utils/grid_problem/cpp/rz_approx.cpp>


// Useful definitions
struct RZ_APPROX_INPUT {
    long double theta;
    long double ellipse[2][2];
    long double point[2];
    long double bbox1[2][2];
    long double bbox2[2][2];
    long double epsilon;
};

const std::vector<RZ_APPROX_INPUT> TEST_CASES = {
    {0.5, {{1, 0}, {0, 1}}, {0, 0}, {{0.96867957926246406, 0.96908066999737796}, {-0.25126070974413917, -0.24353071516762309}}, {{-1.4736257582079002, 1.4736257582079002}, {-0.71743893521430069, 0.71743893521430069}}, 0.01},  // Theta = 0.5, epsilon = 0.01
    {2.0943951023931953, {{1, 0}, {0, 1}}, {0.50000000000000011, -0.8660254037844386}, {{0.49999990527724025, 0.50000009468942652}, {-0.86602541272131472, -0.86602539478982754}}, {{-0.59627070522758074, 0.59627070522758074}, {-1.688965210318714, 1.688965210318714}}, 0.00001},  // Theta = 2pi/3, epsilon = 0.00001
    {6, {{1, 0}, {0, 1}}, {-0.98999249660044542, -0.14112000805986721}, {{-0.98999252538189675, -0.98999246121904394}, {-0.14113448557246364, -0.14110552960647074}}, {{-17.737597536427572, 17.737597536427572}, {-0.060966544098781855, 0.060966544098781855}}, 0.0001}  // Theta = 6, epsilon = 0.0001
};


// Test the rz_approx function
class RzApproxTests : public ::testing::TestWithParam<RZ_APPROX_INPUT> {};

TEST_P(RzApproxTests, TestSolutions) {
    auto params = GetParam();
    auto [theta, ellipse, point, bbox1, bbox2, epsilon] = params;

    auto [u, t] = rz_approx(theta, ellipse, point, bbox1, bbox2, epsilon);

    /*
     * To pass this test, we check that max(singular values of U_approx - U_target) < epsilon with
     * U_approx = [[u, tdag],
     *             [t, udag]]
     * U_target = Rz(theta)
     *          = [[cos(theta/2) - 1.j sin(theta/2),  0],
     *             [0,                                cos(theta/2) + 1.j sin(theta/2)]]
     * 
     * The eigenvalues of a matrix A = [[a,     b],
     *                                  [bdag,  adag]]
     * are given by real(a) ± sqrt( imag(a)^2 + (b * bdag) ).
     * The singular values are given by abs(eigenvalues).
     */

     auto a_real = u.real().to_long_double() - std::cos(theta/2);
     auto a_imag = u.imag().to_long_double() + std::sin(theta/2);
     auto b_real = t.real().to_long_double();
     auto b_imag = t.complex_conjugate().imag().to_long_double();

     auto singular_values = std::vector<long double>{
         std::abs(a_real + std::sqrt( std::pow(a_imag, 2) + std::pow(b_real, 2) + std::pow(b_imag, 2) )),
         std::abs(a_real - std::sqrt( std::pow(a_imag, 2) + std::pow(b_real, 2) + std::pow(b_imag, 2) ))
     };

     EXPECT_LT(*std::max_element(singular_values.begin(), singular_values.end()), epsilon);
}

INSTANTIATE_TEST_SUITE_P(
    RzApproxSuite,
    RzApproxTests,
    ::testing::ValuesIn(TEST_CASES)
);
