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

#include <utility>

#include <qdecomp/rings/cpp/Rings.hpp>
#include <qdecomp/utils/grid_problem/cpp/rz_approx.cpp>


extern "C" {
    struct Domega_struct {
        long long int a = 0; unsigned int la = 0;
        long long int b = 0; unsigned int lb = 0;
        long long int c = 0; unsigned int lc = 0;
        long long int d = 0; unsigned int ld = 0;
    };

    struct Result_struct {
        Domega_struct u;
        Domega_struct t;
    };

    Result_struct rz_approx_helper(
        double theta,
        const double* ellipse,  // Size 2x2
        const double* point,  // Size 2
        const double* bbox1,  // Size 2x2
        const double* bbox2,  // Size 2x2
        double epsilon
    ) {  // Using long double is not cross-platform safe using ctype in Python
        // Convert inputs to appropriate data structures
        std::array<std::array<long double, 2>, 2> ellipse_arr = {{
            {static_cast<long double>(ellipse[0]), static_cast<long double>(ellipse[1])},
            {static_cast<long double>(ellipse[2]), static_cast<long double>(ellipse[3])}
        }};
        std::array<long double, 2> point_arr = {
            {static_cast<long double>(point[0]), static_cast<long double>(point[1])}
        };
        std::array<std::array<long double, 2>, 2> bbox1_arr = {{
            {static_cast<long double>(bbox1[0]), static_cast<long double>(bbox1[1])},
            {static_cast<long double>(bbox1[2]), static_cast<long double>(bbox1[3])}
        }};
        std::array<std::array<long double, 2>, 2> bbox2_arr = {{
            {static_cast<long double>(bbox2[0]), static_cast<long double>(bbox2[1])},
            {static_cast<long double>(bbox2[2]), static_cast<long double>(bbox2[3])}
        }};

        // Find the solution
        std::pair<Domega<long long int>, Domega<long long int>> solution = rz_approx(
            theta,
            ellipse_arr,
            point_arr,
            bbox1_arr,
            bbox2_arr,
            epsilon
        );

        // Data structure to return the solution
        Result_struct result;

        // Store the solution in the data structure
        result.u.a = solution.first.a().num();
        result.u.b = solution.first.b().num();
        result.u.c = solution.first.c().num();
        result.u.d = solution.first.d().num();

        result.u.la = solution.first.a().denom();
        result.u.lb = solution.first.b().denom();
        result.u.lc = solution.first.c().denom();
        result.u.ld = solution.first.d().denom();

        result.t.a = solution.second.a().num();
        result.t.b = solution.second.b().num();
        result.t.c = solution.second.c().num();
        result.t.d = solution.second.d().num();

        result.t.la = solution.second.a().denom();
        result.t.lb = solution.second.b().denom();
        result.t.lc = solution.second.c().denom();
        result.t.ld = solution.second.d().denom();

        return result;
    }
}
