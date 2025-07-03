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
#include <cmath>
#include <iostream>
#include <boost/multiprecision/cpp_int.hpp> // Include the Boost Multiprecision library header

#include "D.hpp"

using namespace boost::multiprecision; // Use the Boost Multiprecision namespace


template <typename T>
D<T>::D(T num, unsigned int denom) : _num(num), _denom(denom) {_reduce();}

template <typename T>
void D<T>::_reduce() {
    while (!(_num & 1) and (_denom > 0)) {
        _num >>= 1;
        _denom--;
    }
}


template <typename T>
T D<T>::num() const {return _num;}

template <typename T>
unsigned int D<T>::denom() const {return _denom;}


template <typename T>
bool D<T>::is_int() const {return _denom == 0;}


template <typename T>
bool D<T>::operator==(const D& other) const {return (_num == other._num) and (_denom == other._denom);}

template <typename T>
bool D<T>::operator==(const T& other) const {return (_num == other) and (_denom == 0);}

template <typename T>
bool D<T>::operator!=(const D& other) const {return !(*this == other);}

template <typename T>
bool D<T>::operator!=(const T& other) const {return !(*this == other);}

template <typename T>
bool D<T>::operator<(const D& other) const {return (_num * (1 << other._denom)) < (other._num * (1 << _denom));}

template <typename T>
bool D<T>::operator<(const T& other) const {return _num < (other << _denom);}

template <typename T>
bool D<T>::operator<=(const D& other) const {return (*this < other) or (*this == other);}

template <typename T>
bool D<T>::operator<=(const T& other) const {return (*this < other) or (*this == other);}

template <typename T>
bool D<T>::operator>(const D& other) const {return !(*this <= other);}

template <typename T>
bool D<T>::operator>(const T& other) const {return !(*this <= other);}

template <typename T>
bool D<T>::operator>=(const D& other) const {return !(*this < other);}

template <typename T>
bool D<T>::operator>=(const T& other) const {return !(*this < other);}


template <typename T>
D<T> D<T>::operator+(const D& other) const {
    if (_denom >= other._denom) {
        T new_num = _num + (other._num << (_denom - other._denom));
        return D(new_num, _denom);
    } else {
        T new_num = other._num + (_num << (other._denom - _denom));
        return D(new_num, other._denom);
    }
}

template <typename T>
D<T> D<T>::operator+(const T& other) const {return D(_num + (other << _denom), _denom);}

template <typename T>
D<T> D<T>::operator-() const {return D(-_num, _denom);}

template <typename T>
D<T> D<T>::operator-(const D& other) const {return *this + (-other);}

template <typename T>
D<T> D<T>::operator-(const T& other) const {return *this + (-other);}

template <typename T>
D<T> D<T>::operator*(const D& other) const {return D(_num * other._num, _denom + other._denom);}

template <typename T>
D<T> D<T>::operator*(const T& other) const {return D(_num * other, _denom);}


template <typename T>
D<T> D<T>::pow(unsigned int n) const {
    return D(static_cast<T>(std::pow(_num, n)), _denom * n);
}


template <typename T>
std::string D<T>::to_string() const {return std::to_string(_num) + "/2^" + std::to_string(_denom);}

template <typename T>
void D<T>::print() const {std::cout << to_string() << std::endl;}

template <typename T>
float D<T>::to_float() const {return static_cast<float>(_num) / (1 << _denom);}

template <typename T>
long double D<T>::to_long_double() const {
    return static_cast<long double>(_num) / ((unsigned int) 1 << _denom);
}
