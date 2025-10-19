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

#include "..\..\rings\cpp\Rings.hpp"
#include "rz_approx.cpp"


/**
 * @brief Cast a vector of double to a vector of long double
 * 
 * @param source The source vector of double
 * @param destination The destination vector of long double
 * @param size The size of the vector
 */
void cast_double_to_longdouble_vector(const double* source, long double* destination, unsigned short size) {
    for (unsigned short i = 0; i < size; ++i) {
        destination[i] = static_cast<long double>(source[i]);
    }
}

/**
 * @brief Cast a 2x2 matrix of double to a matrix of long double
 * 
 * @param source The source matrix of double
 * @param destination The destination matrix of long double
 */
void cast_double_to_longdouble_2by2(const double source[2][2], long double destination[2][2]) {
    unsigned short size = 2;
    for (unsigned short i = 0; i < size; ++i) {
    for (unsigned short j = 0; j < size; ++j) {
        destination[i][j] = static_cast<long double>(source[i][j]);
    }
    }
}


extern "C" {
    struct Domega_struct {
        long long int a = 0; unsigned short la = 0;
        long long int b = 0; unsigned short lb = 0;
        long long int c = 0; unsigned short lc = 0;
        long long int d = 0; unsigned short ld = 0;
    };

    struct Result_struct {
        Domega_struct u;
        Domega_struct t;
    };

    Result_struct rz_approx_helper(
        double theta,
        double ellipse[2][2],
        double point[2],
        double bbox1[2][2],
        double bbox2[2][2],
        double epsilon
    ) {
        // Cast the input parameters to long double
        long double theta_long = static_cast<long double>(theta);
        long double ellipse_long[2][2];
        long double point_long[2];
        long double bbox1_long[2][2];
        long double bbox2_long[2][2];
        long double epsilon_long = static_cast<long double>(epsilon);

        cast_double_to_longdouble_2by2(ellipse, ellipse_long);
        cast_double_to_longdouble_vector(point, point_long, 2);
        cast_double_to_longdouble_2by2(bbox1, bbox1_long);
        cast_double_to_longdouble_2by2(bbox2, bbox2_long);

        // Data structure to return the solution
        Result_struct result;

        // Find the solution
        std::pair<Domega, Domega> solution = rz_approx(
            theta_long,
            ellipse_long,
            point_long,
            bbox1_long,
            bbox2_long,
            epsilon_long
        );

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
