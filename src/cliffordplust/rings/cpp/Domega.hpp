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

#ifndef DOMEGA_HPP
#define DOMEGA_HPP

#include <string>
#include <boost/multiprecision/cpp_int.hpp> // Include the Boost Multiprecision library header

#include "Rings.hpp"


/**
 * @class Domega
 * @brief A class representing elements of the D[\u03C9] ring.
 * 
 * This class represents D[\u03C9] elements with 4 elements from the D ring, the four powers of \u039C.
 * 
 * The class provides basic arithmetic operations, comparison operators, and a method to raise the number to a power.
 * 
 * \tparam T The type of the coefficients. Default is long long int.
 */
template <typename T = long long int>
class Domega {
    private:
        D<T> _a;  ///< \u03C9^3 coefficient of the ring element
        D<T> _b;  ///< \u03C9^2 coefficient of the ring element
        D<T> _c;  ///< \u03C9^1 coefficient of the ring element
        D<T> _d;  ///< \u03C9^0 coefficient of the ring element

    public:
        /**
         * @brief Construct a new D[\u03C9] object.
         * 
         * @param a The numerator of the \u03C9^3 coefficient.
         * @param la The denominator's power of 2 of the \u03C9^3 coefficient.
         * @param b The numerator of the \u03C9^2 coefficient.
         * @param lb The denominator's power of 2 of the \u03C9^2 coefficient.
         * @param c The numerator of the \u03C9^1 coefficient.
         * @param lc The denominator's power of 2 of the \u03C9^1 coefficient.
         * @param d The numerator of the \u03C9^0 coefficient.
         * @param ld The denominator's power of 2 of the \u03C9^0 coefficient.
         */
        Domega(
            T a, unsigned int la,
            T b, unsigned int lb,
            T c, unsigned int lc,
            T d, unsigned int ld
        );

        /**
         * @brief Construct a new D[\u03C9] object.
         * 
         * @param a The \u03C9^3 coefficient.
         * @param b The \u03C9^2 coefficient.
         * @param c The \u03C9^1 coefficient.
         * @param d The \u03C9^0 coefficient.
         */
        Domega(D<T> a, D<T> b, D<T> c, D<T> d);


        /**
         * @brief Get the \u03C9^3 coefficient of the ring element.
         * 
         * @return const D& The \u03C9^3 coefficient.
         */
        const D<T>& a() const;

        /**
         * @brief Get the \u03C9^2 coefficient of the ring element.
         * 
         * @return const D& The \u03C9^2 coefficient.
         */
        const D<T>& b() const;

        /**
         * @brief Get the \u03C9^1 coefficient of the ring element.
         * 
         * @return const D& The \u03C9^1 coefficient.
         */
        const D<T>& c() const;

        /**
         * @brief Get the \u03C9^0 coefficient of the ring element.
         * 
         * @return const D& The \u03C9^0 coefficient.
         */
        const D<T>& d() const;
        
        /**
         * @brief Get the coefficient of the ring element.
         *
         * This method returns the coefficient of the ring element at the given index.
         * It is useful to access the coefficients of the ring element.
         * 
         * @param i The index of the coefficient.
         * @return const D& The coefficient.
         * @throw std::invalid_argument if the index is not between 0 and 3.
         */
        const D<T>& operator[](unsigned int i) const;


        /**
         * @brief Get the real part of the number.
         * 
         * @return Dsqrt2 The real part of the number.
         */
        Dsqrt2<T> real() const;

        /**
         * @brief Get the imaginary part of the number.
         * 
         * @return Dsqrt2 The imaginary part of the number.
         */
        Dsqrt2<T> imag() const;


        /**
         * @brief Get the \u221A2 conjugate of the number.
         * 
         * @return Domega The \u221A2 conjugate of the number.
         */
        Domega<T> sqrt2_conjugate() const;

        /**
         * @brief Get the complex conjugate of the number.
         * 
         * @return Domega The complex conjugate of the number.
         */
        Domega<T> complex_conjugate() const;


        /**
         * @brief Get the smallest denominator exponent.
         * 
         * This method returns the smallest denominator exponent of the coefficients of the number.
         * 
         * @return int The smallest denominator exponent.
         */
        unsigned int sde() const;


        /**
         * @brief Check if the number is in the ring Z[\u03C9].
         *
         * @return true If the number is in the ring Z[\u03C9].
         */
        bool is_Zomega() const;

        /**
         * @brief Check if the number is in the ring D[\u221A2].
         *
         * @return true If the number is in the ring D[\u221A2].
         */
        bool is_Dsqrt2() const;

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
         * @brief Convert the number in the ring Z[\u03C9].
         * 
         * @return Zomega The number in the ring Z[\u03C9].
         * @throw std::runtime_error if the number is not in Z[\u03C9].
         */
        Zomega<T> to_Zomega() const;

        /**
         * @brief Convert the number in the ring D[\u221A2].
         * 
         * @return Dsqrt2 The number in the ring D[\u221A2].
         * @throw std::runtime_error if the number is not in D[\u221A2].
         */
        Dsqrt2<T> to_Dsqrt2() const;

        /**
         * @brief Convert the number in the ring Z[\u221A2].
         * 
         * @return Zsqrt2 The number in the ring Z[\u221A2].
         * @throw std::runtime_error if the number is not in Z[\u221A2].
         */
        Zsqrt2<T> to_Zsqrt2() const;

        /**
         * @brief Convert the number in the ring D.
         * 
         * @return D The number in the ring D.
         * @throw std::runtime_error if the number is not in D.
         */
        D<T> to_D() const;

        /**
         * @brief Convert the number to an integer.
         * 
         * @return int The integer.
         * @throw std::runtime_error if the number is not an integer.
         */
        T to_int() const;


        /**
         * @brief Check if the number is equal to another D[\u03C9] object.
         * 
         * @param other The other D[\u03C9] object.
         * @return true If the numbers are equal.
         * @return false If the numbers are not equal.
         */
        bool operator==(const Domega<T>& other) const;

        /**
         * @brief Check if the number is equal to an integer.
         * 
         * @param other The integer.
         * @return true If the numbers are equal.
         * @return false If the numbers are not equal.
         */
        bool operator==(const T& other) const;

        /**
         * @brief Check if the number is not equal to another D[\u03C9] object.
         * 
         * @param other The other D[\u03C9] object.
         * @return true If the numbers are not equal.
         * @return false If the numbers are equal.
         */
        bool operator!=(const Domega<T>& other) const;

        /**
         * @brief Check if the number is not equal to an integer.
         * 
         * @param other The integer.
         * @return true If the numbers are not equal.
         * @return false If the numbers are equal.
         */
        bool operator!=(const T& other) const;


        /**
         * @brief Add a number in D[\u03C9] to the number.
         * 
         * @param other The other D[\u03C9] object.
         * @return Domega The result of the addition.
         */
        Domega<T> operator+(const Domega<T>& other) const;

        /**
         * @brief Add an integer to the number.
         * 
         * @param other The integer.
         * @return Domega The result of the addition.
         */
        Domega<T> operator+(const T& other) const;

        /**
         * @brief Negate the number.
         * 
         * @return Domega The negated number.
         */
        Domega<T> operator-() const;

        /**
         * @brief Subtract another D[\u03C9] object from the number.
         * 
         * @param other The other D[\u03C9] object.
         * @return Domega The result of the subtraction.
         */
        Domega<T> operator-(const Domega<T>& other) const;

        /**
         * @brief Subtract an integer from the number.
         * 
         * @param other The integer.
         * @return Domega The result of the subtraction.
         */
        Domega<T> operator-(const T& other) const;

        /**
         * @brief Multiply the number by another D[\u03C9] object.
         * 
         * @param other The other D[\u03C9] object.
         * @return Domega The result of the multiplication.
         */
        Domega<T> operator*(const Domega<T>& other) const;

        /**
         * @brief Multiply the number by an integer.
         * 
         * @param other The integer.
         * @return Domega The result of the multiplication.
         */
        Domega<T> operator*(const T& other) const;


        /**
         * @brief Raise the number to a power.
         * 
         * @param n The power.
         * @return Domega The result of the power.
         */
        Domega<T> pow(unsigned int n) const;


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

#endif // DOMEGA_HPP
