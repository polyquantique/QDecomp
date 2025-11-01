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

using namespace boost::multiprecision; // Use the Boost Multiprecision namespace


template <typename T>
Dsqrt2<T>::Dsqrt2(
    T p, unsigned int lp,
    T q, unsigned int lq
) : _p(p, lp), _q(q, lq) {}

template <typename T>
Dsqrt2<T>::Dsqrt2(D<T> p, D<T> q) : _p(p), _q(q) {}


template <typename T>
const D<T>& Dsqrt2<T>::p() const {return _p;}

template <typename T>
const D<T>& Dsqrt2<T>::q() const {return _q;}


template <typename T>
Dsqrt2<T> Dsqrt2<T>::sqrt2_conjugate() const {return Dsqrt2<T>(_p, -_q);}


template <typename T>
bool Dsqrt2<T>::is_Zsqrt2() const {return _p.is_int() and _q.is_int();}

template <typename T>
bool Dsqrt2<T>::is_D() const {return _q == 0;}

template <typename T>
bool Dsqrt2<T>::is_int() const {return _p.is_int() and _q == 0;}


template <typename T>
Domega<T> Dsqrt2<T>::to_Domega() const {return Domega<T>(-_q, D<T>(0, 0), _q, _p);}

template <typename T>
Zomega<T> Dsqrt2<T>::to_Zomega() const {
    if (! is_Zsqrt2()) {
        throw std::runtime_error("The number to convert is not in Zomega. Got " + to_string());
    }
    
    return Zomega<T>(-_q.num(), 0, _q.num(), _p.num());
}

template <typename T>
Zsqrt2<T> Dsqrt2<T>::to_Zsqrt2() const {
    if (! is_Zsqrt2()) {
        throw std::runtime_error("The number to convert is not in Zsqrt2. Got " + to_string());
    }

    return Zsqrt2<T>(_p.num(), _q.num());
}

template <typename T>
D<T> Dsqrt2<T>::to_D() const {
    if (! is_D()) {
        throw std::runtime_error("The number to convert is not in D. Got " + to_string());
    }

    return _p;
}

template <typename T>
T Dsqrt2<T>::to_int() const {
    if (! is_int()) {
        throw std::runtime_error("The number to convert is not an integer. Got " + to_string());
    }
    
    return _p.num();
}

template <typename T>
float Dsqrt2<T>::to_float() const {
    return _p.to_float() + _q.to_float() * static_cast<float>(std::sqrt(2.0));
}

template <typename T>
long double Dsqrt2<T>::to_long_double() const {
    return _p.to_long_double() + _q.to_long_double() * std::sqrt(static_cast<long double>(2.0));
}


template <typename T>
bool Dsqrt2<T>::operator==(const Dsqrt2<T>& other) const {return (_p == other._p) and (_q == other._q);}

template <typename T>
bool Dsqrt2<T>::operator==(const T& other) const {return (_p == other) and (_q == 0);}

template <typename T>
bool Dsqrt2<T>::operator!=(const Dsqrt2<T>& other) const {return !(*this == other);}

template <typename T>
bool Dsqrt2<T>::operator!=(const T& other) const {return !(*this == other);}

template <typename T>
Dsqrt2<T> Dsqrt2<T>::operator+(const Dsqrt2<T>& other) const {return Dsqrt2<T>(_p + other._p, _q + other._q);}

template <typename T>
Dsqrt2<T> Dsqrt2<T>::operator-() const {return Dsqrt2<T>(-_p, -_q);}

template <typename T>
Dsqrt2<T> Dsqrt2<T>::operator-(const Dsqrt2<T>& other) const {return *this + (-other);}

template <typename T>
Dsqrt2<T> Dsqrt2<T>::operator*(const Dsqrt2<T>& other) const {
    return Dsqrt2<T>(
        (_p * other._p) + (_q * other._q * 2),
        (_p * other._q) + (_q * other._p)
    );
}


template <typename T>
Dsqrt2<T> Dsqrt2<T>::sqrt2_multiply(const unsigned int n) const {
    if (n & 1) {
        return Dsqrt2<T>(0, 0, 1, 0) * (*this).sqrt2_multiply(n - 1);
    } else {  // Multiply by 2**(l/2)
        unsigned int power = n >> 1;

        unsigned int decrease_p_power_by = std::min(_p.denom(), power);
        T new_p = _p.num() * (1 << (power - decrease_p_power_by));

        unsigned int decrease_q_power_by = std::min(_q.denom(), power);
        T new_q = _q.num() * (1 << (power - decrease_q_power_by));

        return Dsqrt2<T>(new_p, _p.denom() - decrease_p_power_by, new_q, _q.denom() - decrease_q_power_by);
    }
}


template <typename T>
Dsqrt2<T> Dsqrt2<T>::pow(unsigned int n) const {
    Dsqrt2<T> nth_power = *this;
    Dsqrt2<T> result(1, 0, 0, 0);

    while (n) {
        if (n & 1) {result = result * nth_power;}
        nth_power = nth_power * nth_power;
        n >>= 1;
    }

    return result;
}


template <typename T>
std::string Dsqrt2<T>::to_string() const {return _p.to_string() + " + " + _q.to_string() + "\u221A2";}

template <typename T>
void Dsqrt2<T>::print() const {std::cout << to_string() << std::endl;}


/// Functions in the D[\u221A2] ring
template <typename t_in, typename t_out>
Dsqrt2<t_out> cast_Dsqrt2(const Dsqrt2<t_in>& element) {
    return Dsqrt2<t_out>(
        static_cast<t_out>(element.p().num()), element.p().denom(),
        static_cast<t_out>(element.q().num()), element.q().denom()
    );
}
