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

#ifndef DSQRT2_HPP
#define DSQRT2_HPP

#include <string>

#include "Rings.hpp"


/**
 * @class Dsqrt2
 * @brief A class representing elements of the D[\u221A2] ring.
 * 
 * This class represents D[\u221A2] elements with 2 elements from the D ring, the 1 and \u221A2 coefficients.
 * 
 * The class provides basic arithmetic operations, comparison operators, and a method to raise the number to a power.
 */
class Dsqrt2 {
    private:
        D _p;  ///< The integer coefficient of the ring element
        D _q;  ///< The \u221A2 coefficient of the ring element

    public:
        /**
         * @brief Construct a new D[\u221A2] object.
         * 
         * @param p The numerator of the integer coefficient.
         * @param lp The denominator's power of 2 of the integer coefficient.
         * @param q The numerator of the \u221A2 coefficient.
         * @param lq The denominator's power of 2 of the \u221A2 coefficient.
         */
        Dsqrt2(long long int p, unsigned short lp, long long int q, unsigned short lq);

        /**
         * @brief Construct a new D[\u221A] object.
         * 
         * @param p The integer coefficient.
         * @param q The \u221A2 coefficient.
         */
        Dsqrt2(D p, D q);


        /**
         * @brief Get the integer coefficient of the ring element.
         * 
         * @return const D& The integer coefficient.
         */
        const D& p() const;

        /**
         * @brief Get the \u221A2 coefficient of the ring element.
         * 
         * @return const D& The \u221A2 coefficient.
         */
        const D& q() const;


        /**
         * @brief Get the \u221A2 conjugate of the number.
         * 
         * @return Dsqrt2 The \u221A2 conjugate of the number.
         */
        Dsqrt2 sqrt2_conjugate() const;


        /**
         * @brief Check if the number is in the ring Z[\u221A2].
         *
         * @return true If the number is in the ring Z[\u221A2].
         */
        bool is_Zsqrt2() const;

        /**
         * @brief Check if the number is in the ring D.
         *
         * @return true If the number is in the ring D.
         */
        bool is_D() const;

        /**
         * @brief Check if the number is an integer.
         *
         * @return true If the number is an integer.
         */
        bool is_int() const;


        /**
         * @brief Convert the number in the ring D[\u03C9].
         * 
         * @return Zomega The number in the ring D[\u03C9].
         */
        Domega to_Domega() const;

        /**
         * @brief Convert the number in the ring Z[\u03C9].
         * 
         * @return Zomega The number in the ring Z[\u03C9].
         * @throw std::runtime_error if the number is not in Z[\u03C9].
         */
        Zomega to_Zomega() const;

        /**
         * @brief Convert the number in the ring Z[\u221A2].
         * 
         * @return Zsqrt2 The number in the ring Z[\u221A2].
         * @throw std::runtime_error if the number is not in Z[\u221A2].
         */
        Zsqrt2 to_Zsqrt2() const;

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
        long long int to_int() const;

        /**
         * @brief Convert the number to a float.
         * 
         * @return float The float.
         */
        float to_float() const;

        /**
         * @brief Convert the number to a long double.
         * 
         * @return long double The double.
         */
        long double to_long_double() const;

        
        /**
         * @brief Check if the number is equal to another D[\221A2] object.
         * 
         * @param other The other D[\u221A2] object.
         * @return true If the numbers are equal.
         * @return false If the numbers are not equal.
         */
        bool operator==(const Dsqrt2& other) const;

        /**
         * @brief Check if the number is equal to an integer.
         * 
         * @param other The integer.
         * @return true If the numbers are equal.
         * @return false If the numbers are not equal.
         */
        bool operator==(const long long int& other) const;

        /**
         * @brief Check if the number is not equal to another D[\u221A2] object.
         * 
         * @param other The other D[\u221A2] object.
         * @return true If the numbers are not equal.
         * @return false If the numbers are equal.
         */
        bool operator!=(const Dsqrt2& other) const;

        /**
         * @brief Check if the number is not equal to an integer.
         * 
         * @param other The integer.
         * @return true If the numbers are not equal.
         * @return false If the numbers are equal.
         */
        bool operator!=(const long long int& other) const;


        /**
         * @brief Add a number in D[\u221A2] to the number.
         * 
         * @param other The other D[\u221A2] object.
         * @return Dsqrt2 The result of the addition.
         */
        Dsqrt2 operator+(const Dsqrt2& other) const;
        
        /**
         * @brief Negate the number.
         * 
         * @return Dsqrt2 The negated number.
         */
        Dsqrt2 operator-() const;

        /**
         * @brief Subtract another D[\u221A2] object from the number.
         * 
         * @param other The other D[\u221A2] object.
         * @return Dsqrt2 The result of the subtraction.
         */
        Dsqrt2 operator-(const Dsqrt2& other) const;

        /**
         * @brief Multiply the number by another D[\u221A2] object.
         * 
         * @param other The other D[\u221A2] object.
         * @return Dsqrt2 The result of the multiplication.
         */
        Dsqrt2 operator*(const Dsqrt2& other) const;


        /**
         * @brief Raise the number to a power.
         * 
         * @param n The exponent.
         * @return Dsqrt2 The result of the exponentiation.
         */
        Dsqrt2 pow(unsigned short n) const;


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

#endif // DSQRT2_HPP
