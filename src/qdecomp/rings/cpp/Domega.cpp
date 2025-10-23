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
#include <boost/multiprecision/cpp_int.hpp> // Include the Boost Multiprecision library header

#include "Rings.hpp"

using namespace boost::multiprecision; // Use the Boost Multiprecision namespace


template <typename T>
Domega<T>::Domega(
    T a, unsigned int la,
    T b, unsigned int lb,
    T c, unsigned int lc,
    T d, unsigned int ld
) : _a(a, la), _b(b, lb), _c(c, lc), _d(d, ld) {}

template <typename T>
Domega<T>::Domega(D<T> a, D<T> b, D<T> c, D<T> d) : _a(a), _b(b), _c(c), _d(d) {}


template <typename T>
const D<T>& Domega<T>::a() const {return _a;}

template <typename T>
const D<T>& Domega<T>::b() const {return _b;}

template <typename T>
const D<T>& Domega<T>::c() const {return _c;}

template <typename T>
const D<T>& Domega<T>::d() const {return _d;}


template <typename T>
const D<T>& Domega<T>::operator[](unsigned int i) const {
    switch (i) {
        case 0: return _a;
        case 1: return _b;
        case 2: return _c;
        case 3: return _d;
        default:
            throw std::invalid_argument("Index must be between 0 and 3. Got " + std::to_string(i));
    }
} 


template <typename T>
Dsqrt2<T> Domega<T>::real() const {return Dsqrt2<T>(_d, (_c - _a) * D<T>(1, 1));}

template <typename T>
Dsqrt2<T> Domega<T>::imag() const {return Dsqrt2<T>(_b, (_c + _a) * D<T>(1, 1));}


template <typename T>
Domega<T> Domega<T>::sqrt2_conjugate() const {return Domega<T>(-_a, _b, -_c, _d);}

template <typename T>
Domega<T> Domega<T>::complex_conjugate() const {return Domega<T>(-_c, -_b, -_a, _d);}


template <typename T>
int Domega<T>::sde() const {
    int sde_ = 0;
    T coeffs[4];
    T coeffs_temp[4];

    if (is_D() and _d == 0) {throw std::runtime_error("The sde of zero is undefined.");}

    if (! is_Zomega()) {  // At least one of the coefficients is not an integer.
        int k_max = std::max(std::max(_a.denom(), _b.denom()), std::max(_c.denom(), _d.denom()));
        for (unsigned int i=0; i<4; i++) {
            coeffs[i] = operator[](i).num() << (k_max - operator[](i).denom());
        }
        sde_ = 2 * k_max;
    } else {
        for (unsigned int i=0; i<4; i++) {coeffs[i] = operator[](i).num();}

        while (!(coeffs[0] & 1) and !(coeffs[1] & 1) and !(coeffs[2] & 1) and !(coeffs[3] & 1)) {
            for (unsigned int i=0; i<4; i++) {
                coeffs[i] >>= 1;}
            sde_ -= 2;
        }
    }
    
    while ( ((coeffs[0]&1) == (coeffs[2]&1)) and ((coeffs[1]&1) == (coeffs[3]&1)) ) {
        for (unsigned int i=0; i<4; i++) {
            coeffs_temp[0] = (coeffs[1] - coeffs[3]) >> 1;
            coeffs_temp[1] = (coeffs[2] + coeffs[0]) >> 1;
            coeffs_temp[2] = (coeffs[1] + coeffs[3]) >> 1;
            coeffs_temp[3] = (coeffs[2] - coeffs[0]) >> 1;

            for (unsigned int j=0; i<4; i++) {coeffs[j] = coeffs_temp[j];}
            sde_--;
        }
    }

    return sde_;
}


template <typename T>
bool Domega<T>::is_Zomega() const {
    for (unsigned int i=0; i<4; i++) {
        if ( !(operator[](i).is_int()) ) {return false;}
    }
    return true;
}

template <typename T>
bool Domega<T>::is_Dsqrt2() const {return _b == 0 and _a == -_c;}

template <typename T>
bool Domega<T>::is_Zsqrt2() const {return is_Zomega() and is_Dsqrt2();}

template <typename T>
bool Domega<T>::is_D() const {return _a == 0 and _b == 0 and _c == 0;}

template <typename T>
bool Domega<T>::is_int() const {return is_D() and _d.is_int();}


template <typename T>
Zomega<T> Domega<T>::to_Zomega() const {
    if (! is_Zomega()) {
        throw std::runtime_error("The number to convert is not in Zomega. Got " + to_string());
    }

    return Zomega<T>(_a.num(), _b.num(), _c.num(), _d.num());
}

template <typename T>
Dsqrt2<T> Domega<T>::to_Dsqrt2() const {
    if (! is_Dsqrt2()) {
        throw std::runtime_error("The number to convert is not in Dsqrt2. Got " + to_string());
    }
    
    return Dsqrt2<T>(_d, _c);
}

template <typename T>
Zsqrt2<T> Domega<T>::to_Zsqrt2() const {
    if (! is_Zsqrt2()) {
        throw std::runtime_error("The number to convert is not in Zsqrt2. Got " + to_string());
    }

    return Zsqrt2(_d.num(), _c.num());
}

template <typename T>
D<T> Domega<T>::to_D() const {
    if (! is_D()) {
        throw std::runtime_error("The number to convert is not in D. Got " + to_string());
    }

    return _d;
}

template <typename T>
T Domega<T>::to_int() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }
    
    return _d.num();
}


template <typename T>
bool Domega<T>::operator==(const Domega& other) const {
    return (_a == other._a) and (_b == other._b) and (_c == other._c) and (_d == other._d);
}

template <typename T>
bool Domega<T>::operator==(const T& other) const {return is_int() and (_d == other);}

template <typename T>
bool Domega<T>::operator!=(const Domega& other) const {return !(*this == other);}

template <typename T>
bool Domega<T>::operator!=(const T& other) const {return !(*this == other);}


template <typename T>
Domega<T> Domega<T>::operator+(const Domega& other) const {
    return Domega<T>(_a + other._a, _b + other._b, _c + other._c, _d + other._d);
}

template <typename T>
Domega<T> Domega<T>::operator+(const T& other) const {return Domega<T>(_a, _b, _c, _d + other);}

template <typename T>
Domega<T> Domega<T>::operator-() const {return Domega<T>(-_a, -_b, -_c, -_d);}

template <typename T>
Domega<T> Domega<T>::operator-(const Domega& other) const {return *this + (-other);}

template <typename T>
Domega<T> Domega<T>::operator-(const T& other) const {return *this + (-other);}

template <typename T>
Domega<T> Domega<T>::operator*(const Domega& other) const {
    D<T> a =  (_a * other._d) + (_b * other._c) + (_c * other._b) + (_d * other._a);
    D<T> b = -(_a * other._a) + (_b * other._d) + (_c * other._c) + (_d * other._b);
    D<T> c = -(_a * other._b) - (_b * other._a) + (_c * other._d) + (_d * other._c);
    D<T> d = -(_a * other._c) - (_b * other._b) - (_c * other._a) + (_d * other._d);

    return Domega<T>(a, b, c, d);
}

template <typename T>
Domega<T> Domega<T>::operator*(const T& other) const {return Domega<T>(_a * other, _b * other, _c * other, _d * other);}

template <typename T>
Domega<T> Domega<T>::pow(unsigned int n) const {
    Domega<T> nth_power = *this;
    Domega<T> result(0, 0, 0, 0, 0, 0, 1, 0);

    while (n) {
        if (n & 1) {result = result * nth_power;}
        nth_power = nth_power * nth_power;
        n >>= 1;
    }

    return result;
}


template <typename T>
std::string Domega<T>::to_string() const {
    return _a.to_string() + " \u03C9^3 + " + _b.to_string() + " \u03C9^2 + " + _c.to_string() + " \u03C9 + " + _d.to_string();
}

template <typename T>
void Domega<T>::print() const {std::cout << to_string() << std::endl;}
