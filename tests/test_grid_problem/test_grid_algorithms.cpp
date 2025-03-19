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

#include <iostream>
#include <cmath>

#include "..\..\src\cliffordplust\grid_problem\cpp\grid_algorithms.cpp"


void test_grid_problem_1D_class() {
    std::cout << "Testing the GridProblem1D class ..." << std::endl;
    for (long long int amin=-10; amin<11; amin++) {
    for (long long int amax=amin; amax<11; amax++) {
    for (long long int bmin=-10; bmin<11; bmin++) {
    for (long long int bmax=bmin; bmax<11; bmax++) {
        // std::cout << "Testing the GridAlgorithm1D class for A = [" << amin << ", " << amax
        //     << "] and B = [" << bmin << ", " << bmax << "] ...";

        GridProblem1D gp(amin, amax, bmin, bmax);
        for (auto sol : gp) {
            double alpha_f = sol.to_float();
            double alpha_conj_f = sol.sqrt2_conjugate().to_float();

            if ((alpha_f >= amin) and (alpha_f <= amax) and
                (alpha_conj_f >= bmin) and (alpha_conj_f <= bmax)) {
                continue;
            } else {
                throw std::runtime_error(
                    "Error in GridProblem1D. The solution " + sol.to_string()
                    + " (or its conjugate) is not in the interval A (B)."
                );
            }
        }
        // std::cout << " OK!" << std::endl;
    }
    }
    }
    }
    std::cout << "Successfully tested the GridProblem1D class!" << std::endl << std::endl;
}

void test_grid_problem_2D_class() {
    std::cout << "Testing the GridProblem2D class ..." << std::endl;

    const double SQRT2 = std::sqrt(2);
    const double SQRT2_INV = 1.0 / SQRT2;

    for (long long int axmax=0; axmax<11; axmax++) {
    for (long long int bxmax=0; bxmax<11; bxmax++) {
    for (long long int aymax=0; aymax<11; aymax++) {
    for (long long int bymax=0; bymax<11; bymax++) {
        // std::cout << "Testing the GridAlgorithm2D class for Ax = [0, " << axmax << "], Bx = [0, " <<
        // bxmax << "], Ay = [0, " << aymax << "] and By = [0, " << bxmax << "] ...";

        // Instantiate the grid problems
        GridProblem1D gp1(0, axmax, 0, bxmax);
        GridProblem1D gp2(0, aymax, 0, bymax);

        GridProblem1D gp1_(-SQRT2_INV, axmax-SQRT2_INV, SQRT2_INV, bxmax+SQRT2_INV);
        GridProblem1D gp2_(-SQRT2_INV, aymax-SQRT2_INV, SQRT2_INV, bymax+SQRT2_INV);

        GridProblem2D gp(0, axmax, 0, bxmax, 0, aymax, 0, bymax);

        // Number of solutions
        int n1 = 0, n2 = 0;
        int n1_ = 0, n2_ = 0;
        for (auto sol : gp1) { n1++; }
        for (auto sol : gp2) { n2++; }
        for (auto sol : gp1_) { n1_++; }
        for (auto sol : gp2_) { n2_++; }

        // Total number of solutions
        int n = 0;
        for (auto sol : gp) { n++; }

        // Check if the number of solutions is correct
        if (n != (n1 * n2 + n1_ * n2_)) {
            throw std::runtime_error(
                "Error in GridProblem2D. The number of solutions is incorrect. Got " +
                std::to_string(n) + " solutions instead of " + std::to_string(n1 * n2 + n1_ * n2_) +
                " solutions."
            );
        }
        // std::cout << " OK!" << std::endl;
    }
    }
    }
    }
    std::cout << "Successfully tested the GridProblem2D class!" << std::endl << std::endl;
}

int main() {
    test_grid_problem_1D_class();
    test_grid_problem_2D_class();

    return 0;
}
