
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

#include <vector>
#include <tuple>

#include "..\rings_cpp\Rings.hpp"
#include "diophantine_equation.hpp"
#include "diophantine_equation.cpp"


extern "C" {
    struct Domega_struct {
        bool has_solution = false;
        int a = 0, la = 0;
        int b = 0, lb = 0;
        int c = 0, lc = 0;
        int d = 0, ld = 0;
    };

    Domega_struct solve_xi_eq_ttdag_in_d_helper(int a, int la, int b, int lb) {
        // Data structure to return the solution
        Domega_struct res;

        // Passed this if statement, if solution == 0, then there is no solution.
        if (a == 0 and b == 0) {
            res.has_solution = true;
            return res;
        }

        // Create the Dsqrt2 object of xi
        Dsqrt2 xi(a, la, b, lb);

        // Compute the solution
        Domega solution = solve_xi_eq_ttdag_in_d(xi);

        // Check if there is a solution
        if (solution == 0) {
            res.has_solution = false;
            return res;
        }

        // Store the solution in the data structure
        res.has_solution = true;

        res.a = solution.a().num();
        res.b = solution.b().num();
        res.c = solution.c().num();
        res.d = solution.d().num();
        
        res.la = solution.a().denom();
        res.lb = solution.b().denom();
        res.lc = solution.c().denom();
        res.ld = solution.d().denom();

        return res;
    }
}
