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

#include <string>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <boost/multiprecision/cpp_int.hpp> // Include the Boost Multiprecision library header

#include "Rings.hpp"

using namespace boost::multiprecision; // Use the Boost Multiprecision namespace


template <typename T>
Zsqrt2<T>::Zsqrt2(T p, T q) : _p(p), _q(q) {}


template <typename T>
T Zsqrt2<T>::p() const {return _p;}

template <typename T>
T Zsqrt2<T>::q() const {return _q;}


template <typename T>
Zsqrt2<T> Zsqrt2<T>::sqrt2_conjugate() const {return Zsqrt2<T>(_p, -_q);}


template <typename T>
bool Zsqrt2<T>::is_int() const {return _q == 0;}


template <typename T>
Domega<T> Zsqrt2<T>::to_Domega() const {return Domega<T>(-_q, 0, 0, 0, _q, 0, _p, 0);}

template <typename T>
Zomega<T> Zsqrt2<T>::to_Zomega() const {return Zomega<T>(-_q, 0, _q, _p);}

template <typename T>
Dsqrt2 Zsqrt2<T>::to_Dsqrt2() const {return Dsqrt2(_p, 0, _q, 0);}

template <typename T>
D Zsqrt2<T>::to_D() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }

    return D(_p, 0);
}

template <typename T>
T Zsqrt2<T>::to_int() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }
    
    return _p;
}

template <typename T>
float Zsqrt2<T>::to_float() const {
    return static_cast<float>(_p) + static_cast<float>(_q) * static_cast<float>(std::sqrt(2.0));
}

template <typename T>
long double Zsqrt2<T>::to_long_double() const {
    return static_cast<long double>(_p) + static_cast<long double>(_q) * std::sqrt(static_cast<long double>(2.0));
}


template <typename T>
bool Zsqrt2<T>::operator==(const Zsqrt2& other) const {return (_p == other._p) and (_q == other._q);}

template <typename T>
bool Zsqrt2<T>::operator==(const T& other) const {return (_p == other) and (_q == 0);}

template <typename T>
bool Zsqrt2<T>::operator!=(const Zsqrt2& other) const {return !(*this == other);}

template <typename T>
bool Zsqrt2<T>::operator!=(const T& other) const {return !(*this == other);}

template <typename T>
bool Zsqrt2<T>::operator||(const Zsqrt2& other) const {
    Zsqrt2<T> r1(0, 0);
    Zsqrt2<T> r2(0, 0);
    std::tie(std::ignore, r1) = euclidean_div(*this, other);
    std::tie(std::ignore, r2) = euclidean_div(other, *this);
    return (r1 == 0) and (r2 == 0);
}

template <typename T>
Zsqrt2<T> Zsqrt2<T>::operator+(const Zsqrt2& other) const {return Zsqrt2<T>(_p + other._p, _q + other._q);}

template <typename T>
Zsqrt2<T> Zsqrt2<T>::operator-() const {return Zsqrt2<T>(-_p, -_q);}

template <typename T>
Zsqrt2<T> Zsqrt2<T>::operator-(const Zsqrt2& other) const {return *this + (-other);}

template <typename T>
Zsqrt2<T> Zsqrt2<T>::operator*(const Zsqrt2& other) const {
    return Zsqrt2<T>(
        (_p * other._p) + (2 * _q * other._q),
        (_p * other._q) + (_q * other._p)
    );
}


template <typename T>
Zsqrt2<T> Zsqrt2<T>::pow(unsigned int n) const {
    Zsqrt2<T> nth_power = *this;
    Zsqrt2<T> result(1, 0);

    while (n) {
        if (n & 1) {result = result * nth_power;}
        nth_power = nth_power * nth_power;
        n >>= 1;
    }

    return result;
}


template <typename T>
void Zsqrt2<T>::unit_reduce() {
    Zsqrt2<T> n1 = *this;
    Zsqrt2<T> n2(0, 0);

    // If the sign of p and q is different, reduce the number by multiplying by lambda = 1 + sqrt(2)
    if (std::signbit(n1.p()) xor std::signbit(n1.q())) {
        do {
            n1 = n1 * Zsqrt2<T>(1, 1);
        } while (std::signbit(n1.p()) xor std::signbit(n1.q()));

        // Recover the number with the opposite sign
        n2 = n1 * Zsqrt2<T>(-1, 1);

    // If not, reduce the number by multiplying by lambda**-1 = -1 + sqrt(2)
    } else {
        do {
            n1 = n1 * Zsqrt2<T>(-1, 1);
        } while (std::signbit(n1.p()) == std::signbit(n1.q()));
        
        // Recover the number with the opposite sign
        n2 = n1 * Zsqrt2<T>(1, 1);
    }

    // Return the best number, the one with the smallest coefficients p and q
    T merit1 = std::llabs(n1.p()) + std::llabs(n1.q());
    T merit2 = std::llabs(n2.p()) + std::llabs(n2.q());
    if (merit1 < merit2) {
        *this = n1;
    } else {
        *this = n2;
    }
}


template <typename T>
std::string Zsqrt2<T>::to_string() const {
    return std::to_string(_p) + " + " + std::to_string(_q) + "\u221A2";
}

template <typename T>
void Zsqrt2<T>::print() const {std::cout << to_string() << std::endl;}


/// Functions in the Z[\u221A2] ring
template <typename T>
std::tuple<Zsqrt2, Zsqrt2> euclidean_div(const Zsqrt2& num, const Zsqrt2& div) {
    Zsqrt2<T> num_ = num * div.sqrt2_conjugate();
    T den_ = (div * div.sqrt2_conjugate()).p();

    T a = static_cast<T>(std::round(static_cast<float>(num_.p()) / static_cast<float>(den_)));
    T b = static_cast<T>(std::round(static_cast<float>(num_.q()) / static_cast<float>(den_)));

    Zsqrt2<T> q(a, b);
    Zsqrt2<T> r = num - q * div;

    return {q, r};
}
