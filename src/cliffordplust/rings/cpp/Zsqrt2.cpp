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

#include "Rings.hpp"


Zsqrt2::Zsqrt2(long long int p, long long int q) : _p(p), _q(q) {}

long long int Zsqrt2::p() const {return _p;}
long long int Zsqrt2::q() const {return _q;}

Zsqrt2 Zsqrt2::sqrt2_conjugate() const {return Zsqrt2(_p, -_q);}

bool Zsqrt2::is_int() const {return _q == 0;}

Domega Zsqrt2::to_Domega() const {return Domega(-_q, 0, 0, 0, _q, 0, _p, 0);}

template <typename T>
Zomega<T> Zsqrt2::to_Zomega() const {return Zomega<T>(-_q, 0, _q, _p);}

Dsqrt2 Zsqrt2::to_Dsqrt2() const {return Dsqrt2(_p, 0, _q, 0);}

D Zsqrt2::to_D() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }

    return D(_p, 0);
}

long long int Zsqrt2::to_int() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }
    
    return _p;
}

float Zsqrt2::to_float() const {
    return static_cast<float>(_p) + static_cast<float>(_q) * static_cast<float>(std::sqrt(2.0));
}

long double Zsqrt2::to_long_double() const {
    return static_cast<long double>(_p) + static_cast<long double>(_q) * std::sqrt(static_cast<long double>(2.0));
}

bool Zsqrt2::operator==(const Zsqrt2& other) const {return (_p == other._p) and (_q == other._q);}
bool Zsqrt2::operator==(const long long int& other) const {return (_p == other) and (_q == 0);}
bool Zsqrt2::operator!=(const Zsqrt2& other) const {return !(*this == other);}
bool Zsqrt2::operator!=(const long long int& other) const {return !(*this == other);}

bool Zsqrt2::operator||(const Zsqrt2& other) const {
    Zsqrt2 r1(0, 0);
    Zsqrt2 r2(0, 0);
    std::tie(std::ignore, r1) = euclidean_div(*this, other);
    std::tie(std::ignore, r2) = euclidean_div(other, *this);
    return (r1 == 0) and (r2 == 0);
}

Zsqrt2 Zsqrt2::operator+(const Zsqrt2& other) const {return Zsqrt2(_p + other._p, _q + other._q);}
Zsqrt2 Zsqrt2::operator-() const {return Zsqrt2(-_p, -_q);}
Zsqrt2 Zsqrt2::operator-(const Zsqrt2& other) const {return *this + (-other);}

Zsqrt2 Zsqrt2::operator*(const Zsqrt2& other) const {
    return Zsqrt2(
        (_p * other._p) + (2 * _q * other._q),
        (_p * other._q) + (_q * other._p)
    );
}


Zsqrt2 Zsqrt2::pow(unsigned short n) const {
    Zsqrt2 nth_power = *this;
    Zsqrt2 result(1, 0);

    while (n) {
        if (n & 1) {result = result * nth_power;}
        nth_power = nth_power * nth_power;
        n >>= 1;
    }

    return result;
}


void Zsqrt2::unit_reduce() {
    Zsqrt2 n1 = *this;
    Zsqrt2 n2(0, 0);

    // If the sign of p and q is different, reduce the number by multiplying by lambda = 1 + sqrt(2)
    if (std::signbit(n1.p()) xor std::signbit(n1.q())) {
        do {
            n1 = n1 * Zsqrt2(1, 1);
        } while (std::signbit(n1.p()) xor std::signbit(n1.q()));

        // Recover the number with the opposite sign
        n2 = n1 * Zsqrt2(-1, 1);

    // If not, reduce the number by multiplying by lambda**-1 = -1 + sqrt(2)
    } else {
        do {
            n1 = n1 * Zsqrt2(-1, 1);
        } while (std::signbit(n1.p()) == std::signbit(n1.q()));
        
        // Recover the number with the opposite sign
        n2 = n1 * Zsqrt2(1, 1);
    }

    // Return the best number, the one with the smallest coefficients p and q
    long long int merit1 = std::llabs(n1.p()) + std::llabs(n1.q());
    long long int merit2 = std::llabs(n2.p()) + std::llabs(n2.q());
    if (merit1 < merit2) {
        *this = n1;
    } else {
        *this = n2;
    }
}


std::string Zsqrt2::to_string() const {
    return std::to_string(_p) + " + " + std::to_string(_q) + "\u221A2";
}

void Zsqrt2::print() const {std::cout << to_string() << std::endl;}


/// Functions in the Z[\u221A2] ring
std::tuple<Zsqrt2, Zsqrt2> euclidean_div(const Zsqrt2& num, const Zsqrt2& div) {
    Zsqrt2 num_ = num * div.sqrt2_conjugate();
    long long int den_ = (div * div.sqrt2_conjugate()).p();

    long long int a = static_cast<long long int>(std::round(static_cast<float>(num_.p()) / static_cast<float>(den_)));
    long long int b = static_cast<long long int>(std::round(static_cast<float>(num_.q()) / static_cast<float>(den_)));

    Zsqrt2 q(a, b);
    Zsqrt2 r = num - q * div;

    return {q, r};
}
