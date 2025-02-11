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

#ifndef ZOMEGA_HPP
#define ZOMEGA_HPP

#include <string>

#include "Rings.hpp"


/**
 * @class Zomega
 * @brief A class representing elements of the Z[\u03C9] ring.
 * 
 * This class represents Z[\u03C9]  elements as four integers, the four powers of \u039C.
 * 
 * The class provides basic arithmetic operations, comparison operators, and a method to raise the number to a power.
 */
class Zomega {
    private:
        long long int _a;  ///< \u03C9^3 coefficient of the ring element
        long long int _b;  ///< \u03C9^2 coefficient of the ring element
        long long int _c;  ///< \u03C9^1 coefficient of the ring element
        long long int _d;  ///< \u03C9^0 coefficient of the ring element

    public:
        /**
         * @brief Construct a new Z[\u03C9] object.
         * 
         * @param a The numerator of the \u03C9^3 coefficient.
         * @param b The numerator of the \u03C9^2 coefficient.
         * @param c The numerator of the \u03C9^1 coefficient.
         * @param d The numerator of the \u03C9^0 coefficient.
         */
        Zomega(long long int a, long long int b, long long int c, long long int d);


        /**
         * @brief Get the \u03C9^3 coefficient of the ring element.
         * 
         * @return int The \u03C9^3 coefficient.
         */
        long long int a() const;

        /**
         * @brief Get the \u03C9^2 coefficient of the ring element.
         * 
         * @return int The \u03C9^2 coefficient.
         */
        long long int b() const;

        /**
         * @brief Get the \u03C9^1 coefficient of the ring element.
         * 
         * @return int The \u03C9^1 coefficient.
         */
        long long int c() const;

        /**
         * @brief Get the \u03C9^0 coefficient of the ring element.
         * 
         * @return int The \u03C9^0 coefficient.
         */
        long long int d() const;


        /**
         * @brief Get the coefficient of the ring element.
         *
         * This method returns the coefficient of the ring element at the given index.
         * It is useful to access the coefficients of the ring element.
         * 
         * @param i The index of the coefficient.
         * @return int The coefficient.
         * @throw std::invalid_argument if the index is not between 0 and 3.
         */
        long long int operator[](unsigned short i) const;


        /**
         * @brief Get the real part of the number.
         * 
         * @return Zsqrt2 The real part of the number.
         */
        Zsqrt2 real() const;

        /**
         * @brief Get the imaginary part of the number.
         * 
         * @return Zsqrt2 The imaginary part of the number.
         */
        Zsqrt2 imag() const;


        /**
         * @brief Get the \u221A2 conjugate of the number.
         * 
         * @return Zomega The \u221A2 conjugate of the number.
         */
        Zomega sqrt2_conjugate() const;

        /**
         * @brief Get the complex conjugate of the number.
         * 
         * @return Zomega The complex conjugate of the number.
         */
        Zomega complex_conjugate() const;


        /**
         * @brief Check if the number is in the ring Z[\u221A2].
         *
         * @return true If the number is in the ring Z[\u221A2].
         */
        bool is_Zsqrt2() const;

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
         * @brief Convert the number in the ring D[\u221A2].
         * 
         * @return Dsqrt2 The number in the ring D[\u221A2].
         * @throw std::runtime_error if the number is not in D[\u221A2].
         */
        Dsqrt2 to_Dsqrt2() const;

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
         * @brief Check if the number is equal to another Z[\u03C9] object.
         * 
         * @param other The other Z[\u03C9] object.
         * @return true If the numbers are equal.
         * @return false If the numbers are not equal.
         */
        bool operator==(const Zomega& other) const;

        /**
         * @brief Check if the number is equal to an integer.
         * 
         * @param other The integer.
         * @return true If the numbers are equal.
         * @return false If the numbers are not equal.
         */
        bool operator==(const long long int& other) const;

        /**
         * @brief Check if the number is not equal to another Z[\u03C9] object.
         * 
         * @param other The other Z[\u03C9] object.
         * @return true If the numbers are not equal.
         * @return false If the numbers are equal.
         */
        bool operator!=(const Zomega& other) const;

        /**
         * @brief Check if the number is not equal to an integer.
         * 
         * @param other The integer.
         * @return true If the numbers are not equal.
         * @return false If the numbers are equal.
         */
        bool operator!=(const long long int& other) const;


        /**
         * @brief Add a number in Z[\u03C9] to the number.
         * 
         * @param other The other Z[\u03C9] object.
         * @return Zomega The result of the addition.
         */
        Zomega operator+(const Zomega& other) const;

        /**
         * @brief Add an integer to the number.
         * 
         * @param other The integer.
         * @return Zomega The result of the addition.
         */
        Zomega operator+(const long long int& other) const;

        /**
         * @brief Negate the number.
         * 
         * @return Zomega The negated number.
         */
        Zomega operator-() const;

        /**
         * @brief Subtract another Z[\u03C9] object from the number.
         * 
         * @param other The other Z[\u03C9] object.
         * @return Zomega The result of the subtraction.
         */
        Zomega operator-(const Zomega& other) const;

        /**
         * @brief Subtract an integer from the number.
         * 
         * @param other The integer.
         * @return Zomega The result of the subtraction.
         */
        Zomega operator-(const long long int& other) const;

        /**
         * @brief Multiply the number by another Z[\u03C9] object.
         * 
         * @param other The other Z[\u03C9] object.
         * @return Zomega The result of the multiplication.
         */
        Zomega operator*(const Zomega& other) const;

        /**
         * @brief Multiply the number by an integer.
         * 
         * @param other The integer.
         * @return Zomega The result of the multiplication.
         */
        Zomega operator*(const long long int& other) const;


        /**
         * @brief Raise the number to a power.
         * 
         * @param n The power.
         * @return Zomega The result of the power.
         */
        Zomega pow(unsigned short n) const;


        /**
         * @brief Convert the number to a string.
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
 * @brief Perform the Euclidean division of two Z[\u03C9] objects.
 * 
 * This function returns q and r such that num = q * div + r.
 * 
 * @param num The numerator.
 * @param div The denominator.
 * @return std::tuple<Zomega, Zomega> The quotient and the remainder.
 */
std::tuple<Zomega, Zomega> euclidean_div(const Zomega& num, const Zomega& div);

/**
 * @brief Compute the greatest common divisor of two Z[\u03C9] numbers.
 * 
 * @param x The first Z[\u03C9] number.
 * @param b The second Z[\u03C9] number.
 * @return Zomega The greatest common divisor of the two numbers.
 */
Zomega gcd(const Zomega& x, const Zomega& y);

#endif  // ZOMEGA_HPP
