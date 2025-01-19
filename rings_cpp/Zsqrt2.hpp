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

#ifndef ZSQRT2_HPP
#define ZSQRT2_HPP

#include <string>

#include "Rings.hpp"


/**
 * @class Zsqrt2
 * @brief A class representing elements of the Z[\u221A2] ring.
 * 
 * This class represents Z[\u221A2] elements with 2 elements from the D ring, the 1 and \u221A2 coefficients.
 * 
 * The class provides basic arithmetic operations, comparison operators, and a method to raise the number to a power.
 */
class Zsqrt2 {
    private:
        int _p;  ///< The integer coefficient of the ring element
        int _q;  ///< The \u221A2 coefficient of the ring element

    public:
        /**
         * @brief Construct a new Z[\u221A2] object.
         * 
         * @param p The numerator of the integer coefficient.
         * @param q The numerator of the \u221A2 coefficient.
         */
        Zsqrt2(int p, int q);


        /**
         * @brief Get the integer coefficient of the ring element.
         * 
         * @return int The integer coefficient.
         */
        int p() const;

        /**
         * @brief Get the \u221A2 coefficient of the ring element.
         * 
         * @return int The \u221A2 coefficient.
         */
        int q() const;


        /**
         * @brief Get the \u221A2 conjugate of the number.
         * 
         * @return Zsqrt2 The \u221A2 conjugate of the number.
         */
        Zsqrt2 sqrt2_conjugate() const;


        /**
         * @brief Check if the number is an integer.
         *
         * @return true If the number is an integer.
         */
        bool is_int() const;


        /**
         * @brief Convert the number in the ring D[\u03C9].
         * 
         * @return Domega The number in the ring D[\u03C9].
         */
        Domega to_Domega() const;

        /**
         * @brief Convert the number in the ring Z[\u03C9].
         * 
         * @return Zomega The number in the ring Z[\u03C9].
         */
        Zomega to_Zomega() const;

        /**
         * @brief Convert the number in the ring D[\u221A2].
         * 
         * @return Dsqrt2 The number in the ring D[\u221A2].
         */
        Dsqrt2 to_Dsqrt2() const;

        /**
         * @brief Convert the number in the ring D.
         * 
         * @return D The number in the ring D.
         * @throw std::runtime_error if the number is not in D.
         */
        D to_D() const;

        /**
         * @brief Convert the number to an integer.
         * 
         * @return int The integer.
         * @throw std::runtime_error if the number is not an integer.
         */
        int to_int() const;

        /**
         * @brief Convert the number to a float.
         * 
         * @return float The float.
         */
        float to_float() const;


        /**
         * @brief Check if the number is equal to another Z[\u221A2] object.
         * 
         * @param other The other Z[\u221A2] object.
         * @return true If the numbers are equal.
         * @return false If the numbers are not equal.
         */
        bool operator==(const Zsqrt2& other) const;

        /**
         * @brief Check if the number is equal to an integer.
         * 
         * @param other The integer.
         * @return true If the numbers are equal.
         * @return false If the numbers are not equal.
         */
        bool operator==(const int& other) const;

        /**
         * @brief Check if the number is not equal to another Z[\u221A2] object.
         * 
         * @param other The other Z[\u221A2] object.
         * @return true If the numbers are not equal.
         * @return false If the numbers are equal.
         */
        bool operator!=(const Zsqrt2& other) const;

        /**
         * @brief Check if the number is not equal to an integer.
         * 
         * @param other The integer.
         * @return true If the numbers are not equal.
         * @return false If the numbers are equal.
         */
        bool operator!=(const int& other) const;

        /**
         * @brief Check if the number is similar to another Z[\u221A2] object.
         * 
         * Two numbers in Z[\u221A2] are similar if they only differ by a unit.
         * 
         * @param other The other Z[\u221A2] object.
         * @return true If the number is similar to the other number.
         * @return false If the number is not similar to the other number.
         */
        bool operator||(const Zsqrt2& other) const;


        /**
         * @brief Add a number in Z[\u221A2] to the number.
         * 
         * @param other The other Z[\u221A2] object.
         * @return Zsqrt2 The result of the addition.
         */
        Zsqrt2 operator+(const Zsqrt2& other) const;

        /**
         * @brief Negate the number.
         * 
         * @return Zsqrt2 The negated number.
         */
        Zsqrt2 operator-() const;

        /**
         * @brief Subtract another Z[\u221A2] object from the number.
         * 
         * @param other The other Z[\u221A2] object.
         * @return Zsqrt2 The result of the subtraction.
         */
        Zsqrt2 operator-(const Zsqrt2& other) const;

        /**
         * @brief Multiply the number by another Z[\u221A2] object.
         * 
         * @param other The other Z[\u221A2] object.
         * @return Zsqrt2 The result of the multiplication.
         */
        Zsqrt2 operator*(const Zsqrt2& other) const;


        /**
         * @brief Raise the number to a power.
         * 
         * @param n The exponent.
         * @return Zsqrt2 The result of the exponentiation.
         * @throw std::invalid_argument if the exponent is negative.
         */
        Zsqrt2 pow(int n) const;


        /**
         * @brief Get a string representation of the number.
         * 
         * @return std::string The string representation of the number.
         */
        std::string to_string() const;

        /**
         * @brief Print the number.
         */
        void print() const;
};


/**
 * @brief Perform the Euclidean division of two Z[\u221A2] objects.
 * 
 * This function returns q and r such that num = q * div + r.
 * 
 * @param num The numerator.
 * @param div The denominator.
 * @return std::tuple<Zsqrt2, Zsqrt2> The quotient and the remainder.
 */
std::tuple<Zsqrt2, Zsqrt2> euclidean_div(const Zsqrt2& num, const Zsqrt2& div);


#endif  // ZSQRT2_HPP
