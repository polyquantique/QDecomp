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

#include "Domega.hpp"


Zomega::Zomega(int a, int b, int c, int d) : _a(a), _b(b), _c(c), _d(d) {}

const int Zomega::a() const {return _a;}
const int Zomega::b() const {return _b;}
const int Zomega::c() const {return _c;}
const int Zomega::d() const {return _d;}

int Zomega::operator[](int i) const {
    switch (i) {
        case 0: return _a;
        case 1: return _b;
        case 2: return _c;
        case 3: return _d;
        default:
            throw std::invalid_argument("Index must be between 0 and 3. Got " + std::to_string(i));
    }
}

Zsqrt2 Zomega::real() const {
    int sqrt2_part = _c - _a;
    if (sqrt2_part & 1) {
        throw std::runtime_error("The real part of " + to_string() + " is not in Zsqrt2.");
    }

    return Zsqrt2(_d, sqrt2_part >> 1);
}

Zsqrt2 Zomega::imag() const {
    int sqrt2_part = _c + _a;
    if (sqrt2_part & 1) {
        throw std::runtime_error("The imaginary part of " + to_string() + " is not in Zsqrt2.");
    }
    return Zsqrt2(_b, sqrt2_part >> 1);
}

Zomega Zomega::sqrt2_conjugate() const {return Zomega(-_a, _b, -_c, _d);}
Zomega Zomega::complex_conjugate() const {return Zomega(-_c, -_b, -_a, _d);}

bool Zomega::is_Zsqrt2() const {return _b == 0 and _a == -_c;}
bool Zomega::is_int() const {return _a == 0 and _b == 0 and _c == 0;}

Domega Zomega::to_Domega() const {return Domega(_a, 0, _b, 0, _c, 0, _d, 0);}

Dsqrt2 Zomega::to_Dsqrt2() const {
    if (! is_Zsqrt2()) {
        throw std::runtime_error("The number to convert is not in Dsqrt2. Got " + to_string());
    }

    return Dsqrt2(_d, 0, _c, 0);
}

Zsqrt2 Zomega::to_Zsqrt2() const {
    if (! is_Zsqrt2()) {
        throw std::runtime_error("The number to convert is not in Zsqrt2. Got " + to_string());
    }
    
    return Zsqrt2(_d, _c);
}

D Zomega::to_D() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not in D. Got " + to_string());
    }

    return D(_d, 0);
}

int Zomega::to_int() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }

    return _d;
}

Zomega Zomega::operator==(const Zomega& other) const {
    return (_a == other._a) and (_b == other._b) and (_c == other._c) and (_d == other._d);
}

Zomega Zomega::operator!=(const Zomega& other) const {return !(*this == other);}


Zomega Zomega::operator+(const Zomega& other) const {
    return Zomega(_a + other._a, _b + other._b, _c + other._c, _d + other._d);
}

Zomega Zomega::operator+(const int& other) const {return Zomega(_a, _b, _c, _d + other);}
Zomega Zomega::operator-() const {return Zomega(-_a, -_b, -_c, -_d);}
Zomega Zomega::operator-(const Zomega& other) const {return *this + (-ohter);}
Zomega Zomega::operator-(const int& other) const {return *this + (-other);}

Zomega Zomega::operator*(const Zomega& other) const {
    int a =  (_a * other._d) + (_b * other._c) + (_c * other._b) + (_d * other._a);
    int b = -(_a * other._a) + (_b * other._d) + (_c * other._c) + (_d * other._b);
    int c = -(_a * other._b) - (_b * other._a) + (_c * other._d) + (_d * other._c);
    int d = -(_a * other._c) - (_b * other._b) - (_c * other._a) + (_d * other._d);
    
    return Zomega(a, b, c, d);
}

Zomega Zomega::operator*(const int& other) const {
    return Zomega(_a * other, _b * other, _c * other, _d * other);
}

Zomega Zoemga::pow(int n) const {
    if (n < 0) {
        throw std::invalid_argument("The exponent must be positive. Got " + std::to_string(n));
    }

    Zomega nth_power = *this;
    Zomega result(0, 0, 0, 1);

    while (n) {
        if (n & 1) {result = result * nth_power;}
        nth_power = nth_power * nth_power;
        n >>= 1;
    }

    return result;
}

std::string Zomega::to_string() const {
    return std::to_string(_a) + " \u03C9^3 + " + std::to_string(_b) + " \u03C9^2 + " + std::to_string(_c) + " \u03C9 + " + std::to_string(_d);
}

void Zomega::print() const {std::cout << to_string() << std::endl;}
