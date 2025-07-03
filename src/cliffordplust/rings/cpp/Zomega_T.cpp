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
#include <sstream>
#include <boost/multiprecision/cpp_int.hpp> // Include the Boost Multiprecision library header

#include "Rings.hpp"

using namespace boost::multiprecision; // Use the Boost Multiprecision namespace


template <typename T>
Zomega<T>::Zomega(T a, T b, T c, T d)
: _a(a), _b(b), _c(c), _d(d) {}

template <typename T>
T Zomega<T>::a() const {return _a;}

template <typename T>
T Zomega<T>::b() const {return _b;}

template <typename T>
T Zomega<T>::c() const {return _c;}

template <typename T>
T Zomega<T>::d() const {return _d;}


template <typename T>
T Zomega<T>::operator[](unsigned short i) const {
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
Zsqrt2<T> Zomega<T>::real() const {
    T sqrt2_part = _c - _a;
    if (sqrt2_part & 1) {
        throw std::runtime_error("The real part of " + to_string() + " is not in Zsqrt2.");
    }

    return Zsqrt2<T>(_d, sqrt2_part >> 1);
}

template <typename T>
Zsqrt2<T> Zomega<T>::imag() const {
    T sqrt2_part = _c + _a;
    if (sqrt2_part & 1) {
        throw std::runtime_error("The imaginary part of " + to_string() + " is not in Zsqrt2.");
    }
    return Zsqrt2<T>(_b, sqrt2_part >> 1);
}


template <typename T>
Zomega<T> Zomega<T>::sqrt2_conjugate() const {return Zomega<T>(-_a, _b, -_c, _d);}

template <typename T>
Zomega<T> Zomega<T>::complex_conjugate() const {return Zomega<T>(-_c, -_b, -_a, _d);}


template <typename T>
bool Zomega<T>::is_Zsqrt2() const {return _b == 0 and _a == -_c;}

template <typename T>
bool Zomega<T>::is_int() const {return _a == 0 and _b == 0 and _c == 0;}

template <typename T>
bool Zomega<T>::is_real() const {return _b == 0 and _a == -_c;}


template <typename T>
Domega<T> Zomega<T>::to_Domega() const {return Domega<T>(_a, 0, _b, 0, _c, 0, _d, 0);}

template <typename T>
Dsqrt2<T> Zomega<T>::to_Dsqrt2() const {
    if (! is_Zsqrt2()) {
        throw std::runtime_error("The number to convert is not in Dsqrt2. Got " + to_string());
    }

    return Dsqrt2<T>(_d, 0, _c, 0);
}

template <typename T>
Zsqrt2<T> Zomega<T>::to_Zsqrt2() const {
    if (! is_Zsqrt2()) {
        throw std::runtime_error("The number to convert is not in Zsqrt2. Got " + to_string());
    }
    
    return Zsqrt2<T>(_d, _c);
}

template <typename T>
D<T> Zomega<T>::to_D() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not in D. Got " + to_string());
    }

    return D<T>(_d, 0);
}

template <typename T>
T Zomega<T>::to_int() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }

    return _d;
}


template <typename T>
bool Zomega<T>::operator==(const Zomega<T>& other) const {
    return (_a == other._a) and (_b == other._b) and (_c == other._c) and (_d == other._d);
}

template <typename T>
bool Zomega<T>::operator==(const T& other) const {return is_int() and (_d == other);}

template <typename T>
bool Zomega<T>::operator!=(const Zomega<T>& other) const {return !(*this == other);}

template <typename T>
bool Zomega<T>::operator!=(const T& other) const {return !(*this == other);}


template <typename T>
Zomega<T> Zomega<T>::operator+(const Zomega<T>& other) const {
    return Zomega<T>(_a + other._a, _b + other._b, _c + other._c, _d + other._d);
}

template <typename T>
Zomega<T> Zomega<T>::operator+(const T& other) const {return Zomega<T>(_a, _b, _c, _d + other);}

template <typename T>
Zomega<T> Zomega<T>::operator-() const {return Zomega<T>(-_a, -_b, -_c, -_d);}

template <typename T>
Zomega<T> Zomega<T>::operator-(const Zomega<T>& other) const {return *this + (-other);}

template <typename T>
Zomega<T> Zomega<T>::operator-(const T& other) const {return *this + (-other);}

template <typename T>
Zomega<T> Zomega<T>::operator*(const Zomega<T>& other) const {
    T a =  (_a * other._d) + (_b * other._c) + (_c * other._b) + (_d * other._a);
    T b = -(_a * other._a) + (_b * other._d) + (_c * other._c) + (_d * other._b);
    T c = -(_a * other._b) - (_b * other._a) + (_c * other._d) + (_d * other._c);
    T d = -(_a * other._c) - (_b * other._b) - (_c * other._a) + (_d * other._d);
    
    return Zomega<T>(a, b, c, d);
}

template <typename T>
Zomega<T> Zomega<T>::operator*(const T& other) const {
    return Zomega<T>(_a * other, _b * other, _c * other, _d * other);
}


template <typename T>
Zomega<T> Zomega<T>::pow(unsigned short n) const {
    Zomega<T> nth_power = *this;
    Zomega<T> result(0, 0, 0, 1);

    while (n) {
        if (n & 1) {result = result * nth_power;}
        nth_power = nth_power * nth_power;
        n >>= 1;
    }

    return result;
}


template <typename T>
// std::string Zomega<T>::to_string() const {
//     return std::to_string(_a) + " \u03C9^3 + " + std::to_string(_b) + " \u03C9^2 + " + std::to_string(_c) + " \u03C9 + " + std::to_string(_d);
// }
std::string Zomega<T>::to_string() const {
    std::ostringstream oss;
    oss << _a << " \u03C9^3 + " << _b << " \u03C9^2 + " << _c << " \u03C9 + " << _d;
    return oss.str();
}


template <typename T>
void Zomega<T>::print() const {std::cout << to_string() << std::endl;}


/// Functions in the Z[\u03C9] ring
template <typename t_in, typename t_out>
Zomega<t_out> cast_Zomega(const Zomega<t_in>& element) {
    return Zomega<t_out>(
        static_cast<t_out>(element.a()),
        static_cast<t_out>(element.b()),
        static_cast<t_out>(element.c()),
        static_cast<t_out>(element.d())
    );
}


template <typename T>
std::tuple<Zomega<T>, Zomega<T>> euclidean_div(const Zomega<T>& num, const Zomega<T>& div) {
    // Convert the denominator into an integer, and apply the same transformation to the numerator
    Zomega<T> coef = div.sqrt2_conjugate() * div.complex_conjugate() * div.sqrt2_conjugate().complex_conjugate();

    T denom = (div * coef).to_int();
    Zomega<T> numer = num * coef;

    // Perform the division
    Zomega<T> q = Zomega<T>(
        static_cast<T>(std::round(static_cast<float>(numer.a()) / static_cast<float>(denom))),
        static_cast<T>(std::round(static_cast<float>(numer.b()) / static_cast<float>(denom))),
        static_cast<T>(std::round(static_cast<float>(numer.c()) / static_cast<float>(denom))),
        static_cast<T>(std::round(static_cast<float>(numer.d()) / static_cast<float>(denom)))
    );

    return {q, num - q * div};
}


template <typename T>
Zomega<T> gcd(const Zomega<T>& x, const Zomega<T>& y) {
    Zomega<cpp_int> a = cast_Zomega<T, cpp_int>(y);
    Zomega<cpp_int> b = cast_Zomega<T, cpp_int>(x);
    
    Zomega<cpp_int> r(0, 0, 0, 0);
    do {
        std::tie(std::ignore, r) = euclidean_div(a, b);
        a = b;
        b = r;
    } while (b != 0);

    return cast_Zomega<cpp_int, T>(a);
}
 