/* Copyright 2022-2023 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
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


Domega::Domega(int a, int la, int b, int lb, int c, int lc, int d, int ld)
    : _a(a, la), _b(b, lb), _c(c, lc), _d(d, ld) {
    if (la < 0 or lb < 0 or lc < 0 or ld < 0) {
        throw std::invalid_argument("Denominator must be positive. Got" + 
            std::to_string(la) + ", " + std::to_string(lb) + ", " + 
            std::to_string(lc) + ", " + std::to_string(ld)
        );
    }
}

Domega::Domega(D a, D b, D c, D d) : _a(a), _b(b), _c(c), _d(d) {}


const D& Domega::a() const {return _a;}
const D& Domega::b() const {return _b;}
const D& Domega::c() const {return _c;}
const D& Domega::d() const {return _d;}

const D& Domega::operator[](int i) const {
    switch (i) {
        case 0: return _a;
        case 1: return _b;
        case 2: return _c;
        case 3: return _d;
        default:
            throw std::invalid_argument("Index must be between 0 and 3. Got " + std::to_string(i));
    }
} 


// Dsqrt2 Domega::real() const {return Dsqrt2(d_, (_c - _a) * D(1, 1));}
// Dsqrt2 Domega::imag() const {return Dsqrt2(b_, (_c + _a) * D(1, 1));}


Domega Domega::sqrt2_conjugate() const {return Domega(-_a, _b, -_c, _d);}
Domega Domega::complex_conjugate() const {return Domega(-_c, -_b, -_a, _d);}


int Domega::sde() const {
    int sde_ = 0;
    int coeffs[4];
    int coeffs_temp[4];

    if (is_D() and _d == 0) {throw std::runtime_error("The sde of zero is undefined.");}

    if (! is_Zomega()) {  // At least one of the coefficients is not an integer.
        int k_max = std::max(std::max(_a.denom(), _b.denom()), std::max(_c.denom(), _d.denom()));
        for (int i=1; i<4; i++) {
            coeffs[i] = operator[](i).num() << (k_max - operator[](i).denom());
        }
        sde_ = 2 * k_max;
    } else {
        for (int i=1; i<4; i++) {coeffs[i] = operator[](i).num();}

        while (!(coeffs[0] & 1) or !(coeffs[1] & 1) or !(coeffs[2] & 1) or !(coeffs[3] & 1)) {
            for (int i=0; i<4; i++) {coeffs[i] >>= 1;}
            sde_ -= 2;
        }
    }
    
    while ( ((coeffs[0]&1) == (coeffs[2]&1)) and ((coeffs[1]&1) == (coeffs[3]&1)) ) {
        for (int i=0; i<4; i++) {
            coeffs_temp[0] = (coeffs[1] - coeffs[3]) >> 1;
            coeffs_temp[1] = (coeffs[2] + coeffs[0]) >> 1;
            coeffs_temp[2] = (coeffs[1] + coeffs[3]) >> 1;
            coeffs_temp[3] = (coeffs[2] - coeffs[0]) >> 1;

            for (int i=0; i<4; i++) {coeffs[i] = coeffs_temp[i];}
            sde_--;
        }
    }

    return sde_;
}


bool Domega::is_Zomega() const {
    for (int i=0; i<4; i++) {
        if ( !(operator[](i).is_int()) ) {return false;}
    }
    return true;
}

bool Domega::is_Dsqrt2() const {return _b == 0 and _a == -_c;}
bool Domega::is_Zsqrt2() const {return is_Zomega() and is_Dsqrt2();}
bool Domega::is_D() const {return _a == 0 and _b == 0 and _c == 0;}
bool Domega::is_int() const {return is_D() and _d.is_int();}


// Zomega Domega::to_Zomega() const {
//     if (! is_Zomega()) {
//         throw std::runtime_error("The number to convert is not in Zomega. Got " + to_string());
//     }
// 
//     return Zomega(_a.num(), _b.num(), _c.num(), _d.num());
// }
// 
// Dsqrt2 Domega::to_Dsqrt2() const {
//     if (! is_Dsqrt2()) {
//         throw std::runtime_error("The number to convert is not in Dsqrt2. Got " + to_string());
//     }
//     
//     return Dsqrt2(_d, _c);
// }
// 
// Zsqrt2 Domega::to_Zsqrt2() const {
//     if (! is_Zsqrt2()) {
//         throw std::runtime_error("The number to convert is not in Zsqrt2. Got " + to_string());
//     }
// 
//     return Zsqrt2(_d.num(), _c.num());
// }

D Domega::to_D() const {
    if (! is_D()) {
        throw std::runtime_error("The number to convert is not in D. Got " + to_string());
    }

    return _d;
}

int Domega::to_int() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }
    
    return _d.num();
}


bool Domega::operator==(const Domega& other) const {
    return (_a == other._a) and (_b == other._b) and (_c == other._c) and (_d == other._d);
}

bool Domega::operator!=(const Domega& other) const {return !(*this == other);}


Domega Domega::operator+(const Domega& other) const {
    return Domega(_a + other._a, _b + other._b, _c + other._c, _d + other._d);
}

Domega Domega::operator+(const int& other) const {return Domega(_a, _b, _c, _d + other);}
Domega Domega::operator-() const {return Domega(-_a, -_b, -_c, -_d);}
Domega Domega::operator-(const Domega& other) const {return *this + (-other);}
Domega Domega::operator-(const int& other) const {return *this + (-other);}

Domega Domega::operator*(const Domega& other) const {
    D a =  (_a * other._d) + (_b * other._c) + (_c * other._b) + (_d * other._a);
    D b = -(_a * other._a) + (_b * other._d) + (_c * other._c) + (_d * other._b);
    D c = -(_a * other._b) - (_b * other._a) + (_c * other._d) + (_d * other._c);
    D d = -(_a * other._c) - (_b * other._b) - (_c * other._a) + (_d * other._d);

    return Domega(a, b, c, d);
}

Domega Domega::operator*(const int& other) const {return Domega(_a * other, _b * other, _c * other, _d * other);}

Domega Domega::pow(int n) const {
    if (n < 0) {
        throw std::invalid_argument("The exponent must be positive. Got " + std::to_string(n));
    }

    Domega nth_power = *this;
    Domega result(0, 0, 0, 0, 0, 0, 0, 1);

    while (n) {
        if (n & 1) {result = result * nth_power;}
        nth_power = nth_power * nth_power;
        n >>= 1;
    }

    return result;
}


std::string Domega::to_string() const {
    return _a.to_string() + " \u03C9^3 + " + _b.to_string() + " \u03C9^2 + " + _c.to_string() + " \u03C9 + " + _d.to_string();
}

void Domega::print() const {std::cout << to_string() << std::endl;}
