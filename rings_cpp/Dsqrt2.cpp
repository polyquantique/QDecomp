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

#include "Rings.hpp"


Dsqrt2::Dsqrt2(int p, int lp, int q, int lq) : _p(p, lp), _q(q, lq) {
    if (lp < 0 or lq < 0) {
        throw std::invalid_argument("Denominator must be positive. Got " + 
            std::to_string(lp) + ", " + std::to_string(lq)
        );
    }
}

Dsqrt2::Dsqrt2(D p, D q) : _p(p), _q(q) {}


const D& Dsqrt2::p() const {return _p;}
const D& Dsqrt2::q() const {return _q;}

Dsqrt2 Dsqrt2::sqrt2_conjugate() const {return Dsqrt2(_p, -_q);}

bool Dsqrt2::is_Zsqrt2() const {return _p.is_int() and _q.is_int();}
bool Dsqrt2::is_D() const {return _q == 0;}
bool Dsqrt2::is_int() const {return _p.is_int() and _q == 0;}

Domega Dsqrt2::to_Domega() const {return Domega(-_q, D(0, 0), _q, _p);}

Zomega Dsqrt2::to_Zomega() const {
    if (! is_Zsqrt2()) {
        throw std::runtime_error("The number to convert is not in Dsqrt2. Got " + to_string());
    }
    
    return Zomega(-_q.num(), 0, _q.num(), _p.num());
}

Zsqrt2 Dsqrt2::to_Zsqrt2() const {
    if (! is_Zsqrt2()) {
        throw std::runtime_error("The number to convert is not in Dsqrt2. Got " + to_string());
    }

    return Zsqrt2(_p.num(), _q.num());
}

D Dsqrt2::to_D() const {
    if (! is_D()) {
        throw std::runtime_error("The number to convert is not in D. Got " + to_string());
    }

    return _p;
}

int Dsqrt2::to_int() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }
    
    return _p.num();
}

float Dsqrt2::to_float() const {return _p.to_float() + _q.to_float() * std::sqrt(2.0);}


bool Dsqrt2::operator==(const Dsqrt2& other) const {return (_p == other._p) and (_q == other._q);}
bool Dsqrt2::operator==(const int& other) const {return (_p == other) and (_q == 0);}
bool Dsqrt2::operator!=(const Dsqrt2& other) const {return !(*this == other);}
bool Dsqrt2::operator!=(const int& other) const {return !(*this == other);}

Dsqrt2 Dsqrt2::operator+(const Dsqrt2& other) const {return Dsqrt2(_p + other._p, _q + other._q);}
Dsqrt2 Dsqrt2::operator-() const {return Dsqrt2(-_p, -_q);}
Dsqrt2 Dsqrt2::operator-(const Dsqrt2& other) const {return *this + (-other);}

Dsqrt2 Dsqrt2::operator*(const Dsqrt2& other) const {
    return Dsqrt2(
        (_p * other._p) + (_q * other._q * 2),
        (_p * other._q) + (_q * other._p)
    );
}


Dsqrt2 Dsqrt2::pow(int n) const {
    if (n < 0) {
        throw std::invalid_argument("Exponent must be positive. Got " + std::to_string(n));
    }

    Dsqrt2 nth_power = *this;
    Dsqrt2 result(1, 0, 0, 0);

    while (n) {
        if (n & 1) {result = result * nth_power;}
        nth_power = nth_power * nth_power;
        n >>= 1;
    }

    return result;
}


std::string Dsqrt2::to_string() const {return _p.to_string() + " + " + _q.to_string() + "\u221A2";}
void Dsqrt2::print() const {std::cout << to_string() << std::endl;}
