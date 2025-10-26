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

#ifndef TONELLI_HPP
#define TONELLI_HPP


/**
 * @brief Compute the modular exponentiation of a base raised to an exponent modulo a number.
 * 
 * This function computes the expression (base^exp) % mod efficiently.
 * 
 * @param base The base of the exponentiation.
 * @param exp The exponent to which the base is raised.
 * @param mod The modulus to which the result is reduced.
 * @return The result of (base^exp) % mod.
 */
template <typename T = long long int>
T mod_pow(T const &base, T const &exp, T const &mod);

/**
 * @brief Compute the Legendre symbol (a/p) using Euler's criterion.
 * 
 * The Legendre symbol is defined as follows:
 *  - (a/p) = 0 if a is divisible by p
 *  - (a/p) = 1 if a is a quadratic residue modulo p (i.e., there exists an integer x such that x^2 = a (mod p))
 *  - (a/p) = -1 if a is not a quadratic residue modulo p
 * 
 * @param a The integer for whic the Legendre symbol is computed.
 * @param p A prime number.
 * @return The Legendre symbol (a/p).
 */
template <typename T = long long int>
int legendre_symbol(T const &a, T const &p);

/**
 * @brief Tonelli-Shanks algorithm to find the smallest square root of a modulo p.
 * 
 * The problem solved by this function is stated as follows:
 * Given a prime number p and an integer a, find an integer t such that r^2 = a (mod p).
 * If no solution exists, a Runtime error is raised.
 * 
 * @param a The integer for which the square root is computed.
 * @param p A prime number.
 * @return The smallest non-negative integer r such that r^2 = a (mod p).
 * @throws std::runtime_error if a is not a quadratic residue module p, i.e. no solution exists.
 */
template <typename T = long long int>
T tonelli_shanks_algo(T const &a, T const &p);


#include "tonelli_shanks.tpp"

#endif // TONELLI_HPP
