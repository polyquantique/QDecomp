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

#ifndef D_HPP
#define D_HPP

#include <string>


/**
 * @class D
 * @brief A class representing elements of the D ring.
 * 
 * This class represents D elements as a pair of integers, the numerator and the denominator's power of 2.
 * The denominator exponent must be positive, and the fraction is reduced to its simplest form.
 * 
 * The class provides basic arithmetic operations, comparison operators, and a method to raise the number to a power.
 */
class D {
    private:
        int _num;  ///< The numerator of the fraction.
        int _denom;  ///< The denominator's power of 2.

        /**
         * @brief Reduce the fraction to its simplest form.
         */
        void _reduce();

    public:
        /**
         * @brief Construct a new D object.
         * 
         * @param num The numerator of the fraction.
         * @param denom The denominator's power of 2.
         * @throw std::invalid_argument if the denominator is negative.
         */
        D(int num, int denom);
        
        /**
         * @brief Get the numerator of the fraction.
         * 
         * @return int The numerator of the fraction.
         */
        int num() const;

        /**
         * @brief Get the denominator's power of 2.
         * 
         * @return int The denominator's power of 2.
         */
        int denom() const;


        /**
         * @brief Check if the number is an integer.
         * 
         * @return true If the number is an integer.
         * @return false If the number is not an integer.
         */
        bool is_int() const;


        /**
         * @brief Check if the number is equal to another D object.
         * 
         * @param other The other D object.
         * @return true If the numbers are equal.
         * @return false If the numbers are not equal.
         */
        bool operator==(const D& other) const;

        /**
         * @brief Check if the number is equal to an integer.
         * 
         * @param other The integer.
         * @return true If the number is equal to the integer.
         * @return false If the number is not equal to the integer.
         */
        bool operator==(const int& other) const;

        /**
         * @brief Check if the number is not equal to another D object.
         * 
         * @param other The other D object.
         * @return true If the numbers are not equal.
         * @return false If the numbers are equal.
         */
        bool operator!=(const D& other) const;

        /**
         * @brief Check if the number is not equal to an integer.
         * 
         * @param other The integer.
         * @return true If the number is not equal to the integer.
         * @return false If the number is equal to the integer.
         */
        bool operator!=(const int& other) const;

        /**
         * @brief Check if the number is less than another D object.
         * 
         * @param other The other D object.
         * @return true If the number is less than the other number.
         * @return false If the number is not less than the other number.
         */
        bool operator<(const D& other) const;

        /**
         * @brief Check if the number is less than an integer.
         * 
         * @param other The integer.
         * @return true If the number is less than the integer.
         * @return false If the number is not less than the integer.
         */
        bool operator<(const int& other) const;

        /**
         * @brief Check if the number is less than or equal to another D object.
         * 
         * @param other The other D object.
         * @return true If the number is less than or equal to the other number.
         * @return false If the number is not less than or equal to the other number.
         */
        bool operator<=(const D& other) const;

        /**
         * @brief Check if the number is less than or equal to an integer.
         * 
         * @param other The integer.
         * @return true If the number is less than or equal to the integer.
         * @return false If the number is not less than or equal to the integer.
         */
        bool operator<=(const int& other) const;

        /**
         * @brief Check if the number is greater than another D object.
         * 
         * @param other The other D object.
         * @return true If the number is greater than the other number.
         * @return false If the number is not greater than the other number.
         */
        bool operator>(const D& other) const;

        /**
         * @brief Check if the number is greater than an integer.
         * 
         * @param other The integer.
         * @return true If the number is greater than the integer.
         * @return false If the number is not greater than the integer.
         */
        bool operator>(const int& other) const;

        /**
         * @brief Check if the number is greater than or equal to another D object.
         * 
         * @param other The other D object.
         * @return true If the number is greater than or equal to the other number.
         * @return false If the number is not greater than or equal to the other number.
         */
        bool operator>=(const D& other) const;

        /**
         * @brief Check if the number is greater than or equal to an integer.
         * 
         * @param other The integer.
         * @return true If the number is greater than or equal to the integer.
         * @return false If the number is not greater than or equal to the integer.
         */
        bool operator>=(const int& other) const;


        /**
         * @brief Add another D object to the number.
         * 
         * @param other The other D object.
         * @return D The result of the addition.
         */
        D operator+(const D& other) const;

        /**
         * @brief Add an integer to the number.
         * 
         * @param other The integer.
         * @return D The result of the addition.
         */
        D operator+(const int& other) const;

        /**
         * @brief Negate the number.
         * 
         * @return D The negated number.
         */
        D operator-() const;

        /**
         * @brief Subtract another D object from the number.
         * 
         * @param other The other D object.
         * @return D The result of the subtraction.
         */
        D operator-(const D& other) const;

        /**
         * @brief Subtract an integer from the number.
         * 
         * @param other The integer.
         * @return D The result of the subtraction.
         */
        D operator-(const int& other) const;

        /**
         * @brief Multiply the number by another D object.
         * 
         * @param other The other D object.
         * @return D The result of the multiplication.
         */
        D operator*(const D& other) const;

        /**
         * @brief Multiply the number by an integer.
         * 
         * @param other The integer.
         * @return D The result of the multiplication.
         */
        D operator*(const int& other) const;

        /**
         * @brief Raise the number to a power.
         * 
         * @param n The power.
         * @return D The result of the operation.
         * @throw std::invalid_argument If the exponent is negative.
         */
        D pow(int n) const;


        /**
         * @brief Get a string representation of the number.
         * 
         * @return std::string The string representation of the number.
         */
        std::string to_string() const;

        /**
         * @brief Print the number to the standard output.
         */
        void print() const;

        /**
         * @brief Convert the number to a float.
         * 
         * @return float The float representation of the number.
         */
        float to_float() const;
};

#endif // D_HPP
