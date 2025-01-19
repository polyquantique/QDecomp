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

#ifndef DIOPHANTINE_HPP
#define DIOPHANTINE_HPP

#include <vector>
#include <tuple>

#include "..\rings_cpp\Rings.hpp"


bool is_square(int n);
int solve_usquare_eq_a_mod_p(int a, int p);

std::vector<std::tuple<int, int>> int_fact(int n);
std::vector<std::tuple<int, Zsqrt2, int>> xi_fact(Zsqrt2 xi);
Zsqrt2 pi_fact_into_xi(int pi);
Zomega xi_i_fact_into_ti(int pi, Zsqrt2 xi_i);

Zomega solve_xi_sim_ttdag_in_z(Zsqrt2 xi);
Domega solve_xi_eq_ttdag_in_d(Dsqrt2 xi);


#endif // DIOPHANTINE_HPP
