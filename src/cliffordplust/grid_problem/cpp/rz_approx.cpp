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
#include <cmath>

#include "..\..\rings\cpp\Rings.hpp"
#include "..\..\diophantine\cpp\diophantine_equation.cpp"
#include "grid_algorithms.cpp"


void multiply_bbox(long double bbox[2][2], long double factor) {
    if (factor >= 0) {
        bbox[0][0] *= factor;
        bbox[0][1] *= factor;
        bbox[1][0] *= factor;
        bbox[1][1] *= factor;
    } else {
        long double temp = bbox[0][0];
        bbox[0][0] = bbox[0][1] * factor;
        bbox[0][1] = temp * factor;

        temp = bbox[1][0];
        bbox[1][0] = bbox[1][1] * factor;
        bbox[1][1] = temp * factor;
    }
}

bool is_inside_ellipse(const long double ellipse[2][2], const long double point[2],
    const long double offset[2]) {
    long double x = point[0] - offset[0];
    long double y = point[1] - offset[1];
    return (ellipse[0][0] * x * x + 2 * ellipse[0][1] * x * y + ellipse[1][1] * y * y ) <= 1;
}

/**
 * @brief Approximate a Rz rotation by Clifford+T matrix
 * 
 * This function approximates a Rz rotation by a Clifford+T matrix. The function returns two
 * numbers that are the diagonal and anti-diagonal elements (up to a complex conjugate) of the
 * matrix. Some processing, carried out in Python, must be done before calling this function. It
 * returns some inputs of the function such as the ellipse, the point and the bounding boxes.
 * 
 * Note: for the BBOX, the first index refers to the coordinate and the second one refers to the
 * minimum or maximum bound, i.e. bbox[1][0] is the minimum bound of the y coordinate.
 * 
 * @param theta The angle of the Rz rotation
 * @param ellipse The smallest ellipse englobing the disk slice
 * @param point The center point of the ellipse
 * @param bbox1 The bounding box of the ellipse
 * @param bbox2 The bounding box of the transformed unit disk
 * @param epsilon The precision of the approximation
 * @return A pair containing the diagonal and anti-diagonal elements, respectively, of the matrix
 *  approximating the Rz rotation
 */
std::pair<Domega, Domega> rz_approx(
    long double theta,
    long double ellipse[2][2],
    long double point[2],
    long double bbox1[2][2],
    long double bbox2[2][2],
    long double epsilon
) {
    // Point on the unit circle
    long double z[2] = {std::cos(theta / 2), std::sin(theta / 2)};
    
    // Usefull variables
    unsigned short int n = 0;  // Iteration
    bool odd;  // True if n is odd
    Domega constant(0, 0, 0, 0, 0, 0, 0, 0);  // Constant used to calculate u

    // Solve the problem
    while (true) {
        odd = (bool)(n & 1);

        if (odd) { constant = Dsqrt2(0, 0, 1, (n >> 1) + 1).to_Domega(); }
        else { constant = Domega(0, 0, 0, 0, 0, 0, 1, n >> 1); }

        long double A[2][2] = {{bbox1[0][0], bbox1[0][1]}, {bbox1[1][0], bbox1[1][1]}};  // Bbox
        long double B[2][2] = {{bbox2[0][0], bbox2[0][1]}, {bbox2[1][0], bbox2[1][1]}};  // Transformed bbox

        long double sqrt2_n = std::pow(2, n >> 1);
        if (odd) { sqrt2_n *= std::sqrt(2); }    
        multiply_bbox(A, sqrt2_n);
        multiply_bbox(B, sqrt2_n);

        GridProblem2D gp(A[0][0], A[0][1], B[0][0], B[0][1], A[1][0], A[1][1], B[1][0], B[1][1]);
        int n_candidate = 0;
        for (const auto& candidate : gp) {
            n_candidate++;

            if ( n == 0 or (candidate.a() - candidate.c()) & 1 or (candidate.b() - candidate.d()) & 1 ) {
                Domega u = candidate.to_Domega() * constant;
                Dsqrt2 re = u.real();
                Dsqrt2 im = u.imag();

                long double u_tuple[2] = {re.to_long_double(), im.to_long_double()};
                if (! is_inside_ellipse(ellipse, u_tuple, point)) { continue; }  // True if the candidate is not in the ellipse
                if ( (re.pow(2) + im.pow(2)).to_long_double() > 1) { continue; }  // True if the candidate is not in the unit disk
                long double dst = re.to_long_double() * z[0] - im.to_long_double() * z[1];  // Distance of the point from the center of the ellipse
                if ( dst < 1 - std::pow(epsilon, static_cast<long double>(2)) / 2 ) { continue; }  // True if the candidate is not in the slice
                
                // At this point, the candidate solves the grid problem and is in the slice
                Domega t(0, 0, 0, 0, 0, 0, 0, 0);  // Create a Domega object to store the solution
                Dsqrt2 xi = Dsqrt2(1, 0, 0, 0) - (u * u.complex_conjugate()).to_Dsqrt2();

                if (xi != Dsqrt2(0, 0, 0, 0)) {  // If xi == 0, the solution to the diophantine equation is t = 0
                    t = solve_xi_eq_ttdag_in_d(xi);  // Solve the diophantine equation
                    if ( t == Domega(0, 0, 0, 0, 0, 0, 0, 0) ) { continue; }  // True if the solution does not exist for the diophantine equation
                }

                // Return the solution
                return std::make_pair(u, t);
            }
        }
        n++;
    }
}
