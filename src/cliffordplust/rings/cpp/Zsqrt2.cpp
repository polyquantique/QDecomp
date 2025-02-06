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


Zsqrt2::Zsqrt2(int p, int q) : _p(p), _q(q) {}

int Zsqrt2::p() const {return _p;}
int Zsqrt2::q() const {return _q;}

Zsqrt2 Zsqrt2::sqrt2_conjugate() const {return Zsqrt2(_p, -_q);}

bool Zsqrt2::is_int() const {return _q == 0;}

Domega Zsqrt2::to_Domega() const {return Domega(-_q, 0, 0, 0, _q, 0, _p, 0);}
Zomega Zsqrt2::to_Zomega() const {return Zomega(-_q, 0, _q, _p);}
Dsqrt2 Zsqrt2::to_Dsqrt2() const {return Dsqrt2(_p, 0, _q, 0);}

D Zsqrt2::to_D() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }

    return D(_p, 0);
}

int Zsqrt2::to_int() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }
    
    return _p;
}

float Zsqrt2::to_float() const {return _p + _q * std::sqrt(2.0);}


bool Zsqrt2::operator==(const Zsqrt2& other) const {return (_p == other._p) and (_q == other._q);}
bool Zsqrt2::operator==(const int& other) const {return (_p == other) and (_q == 0);}
bool Zsqrt2::operator!=(const Zsqrt2& other) const {return !(*this == other);}
bool Zsqrt2::operator!=(const int& other) const {return !(*this == other);}

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


Zsqrt2 Zsqrt2::pow(int n) const {
    if (n < 0) {
        throw std::invalid_argument("The exponent must be positive. Got " + std::to_string(n));
    }

    Zsqrt2 nth_power = *this;
    Zsqrt2 result(1, 0);

    while (n) {
        if (n & 1) {result = result * nth_power;}
        nth_power = nth_power * nth_power;
        n >>= 1;
    }

    return result;
}


std::string Zsqrt2::to_string() const {
    return std::to_string(_p) + " + " + std::to_string(_q) + "\u221A2";
}

void Zsqrt2::print() const {std::cout << to_string() << std::endl;}


/// Functions in the Z[\u221A2] ring
std::tuple<Zsqrt2, Zsqrt2> euclidean_div(const Zsqrt2& num, const Zsqrt2& div) {
    Zsqrt2 num_ = num * div.sqrt2_conjugate();
    int den_ = (div * div.sqrt2_conjugate()).p();

    int a = std::round(static_cast<float>(num_.p()) / den_);
    int b = std::round(static_cast<float>(num_.q()) / den_);

    Zsqrt2 q(a, b);
    Zsqrt2 r = num - q * div;

    return {q, r};
}
