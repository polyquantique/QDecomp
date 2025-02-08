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

#include "..\..\rings\cpp\Rings.hpp"


/**
 * @brief Determine wether a number is a square or not.
 * 
 * @param n The number to check.
 * @return true if n is a square, false otherwise.
 */
bool is_square(int n);

/**
 * @brief Solve the Diophantine equation u^2 = -a (mod p).
 * 
 * This function is used to solve the equation u^2 = -a (mod p).
 * This is equivalent to solve the equation u^2 = q * p - a.
 * 
 * @param a The value of a.
 * @param p The value of p.
 * @return The solution u.
 */
int solve_usquare_eq_a_mod_p(int a, int p);


/**
 * @brief Find the factorization of a number.
 * 
 * @param n The number to factorize.
 * @return The factorization of n as a vector of tuples (p, m) where p is a prime number and m is
 *  the exponent of p in the factorization of n.
 */
std::vector<std::tuple<int, int>> int_fact(int n);

/**
 * @brief Find the factorization, up to a prime, of a number in Z[\u221A2].
 * 
 * @param xi The number to factorize in Z[\u221A2].
 * @return The factorization of xi as a vector of tuples (p, xi_i, m) where p is an integer prime
 *  number associated to xi_i, xi_i is a prime of xi in Z[\u221A2], and m is the exponent of xi_i in
 *  the factorization of n.
 */
std::vector<std::tuple<int, Zsqrt2, int>> xi_fact(Zsqrt2 xi);

/**
 * @brief Find the factorization of a prime integer in Z[\u221A2].
 * 
 * @param pi The prime number to factorize in Z[\u221A2].
 * @return A prime factor of pi in Z[\u221A2]. The \u221A2 conjugate is also a prime factor of pi.
 */
Zsqrt2 pi_fact_into_xi(int pi);

/**
 * @brief Find the factorization of a prime element of Z[\u221A2] in Z[\u03C9].
 * 
 * @param pi The integer prime number associated to xi_i. pi = xi * xi.sqrt2_conjugate(). Passing it
 *  as an argument is more efficient than recalculating it.
 * @param xi_i A prime number in Z[\u221A2] to factorize.
 * @return A prime factor of xi_i in Z[\u03C9]. The complex conjugate is also a prime factor of xi_i.
 */
Zomega xi_i_fact_into_ti(int pi, Zsqrt2 xi_i);


/**
 * @brief Solve the Diophantine equation xi ~ t * t^\u2020.
 * 
 * This function is used to solve the equation xi ~ t * t^\u2020 for t where \u2020 denotes the
 * complex conjugate. xi is an element of Z[\u221A2] and t is an element of Z[\u03C9]. This function
 * returns the first solution of the equation. If no solution exists, it returns Zomega(0, 0, 0, 0).
 * 
 * @param xi The value of xi.
 * @return The solution t, or Zomega(0, 0, 0, 0) if no solution exists.
 */
Zomega solve_xi_sim_ttdag_in_z(Zsqrt2 xi);

/**
 * @brief Solve the Diophantine equation xi = t * t^\u2020.
 * 
 * This function is used to solve the equation xi ~ t * t^\u2020 for t where \u2020 denotes the
 * complex conjugate. xi is an element of D[\u221A2] and t is an element of D[\u03C9]. This function
 * returns the first solution of the equation. If no solution exists, it returns
 * Domega(0, 0, 0, 0, 0, 0, 0, 0).
 * 
 * @param xi The value of xi.
 * @return The solution t, or Domega(0, 0, 0, 0, 0, 0, 0, 0) if no solution exists.
 */
Domega solve_xi_eq_ttdag_in_d(Dsqrt2 xi);


#endif // DIOPHANTINE_HPP
