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

#include "D.hpp"


D::D(long long int num, unsigned short denom) : _num(num), _denom(denom) {_reduce();}

void D::_reduce() {
    while (!(_num & 1) and (_denom > 0)) {
        _num >>= 1;
        _denom--;
    }
}


long long int D::num() const {return _num;}
unsigned short D::denom() const {return _denom;}


bool D::is_int() const {return _denom == 0;}


bool D::operator==(const D& other) const {return (_num == other._num) and (_denom == other._denom);}
bool D::operator==(const long long int& other) const {return (_num == other) and (_denom == 0);}

bool D::operator!=(const D& other) const {return !(*this == other);}
bool D::operator!=(const long long int& other) const {return !(*this == other);}

bool D::operator<(const D& other) const {return (_num * (1 << other._denom)) < (other._num * (1 << _denom));}
bool D::operator<(const long long int& other) const {return _num < (other << _denom);}

bool D::operator<=(const D& other) const {return (*this < other) or (*this == other);}
bool D::operator<=(const long long int& other) const {return (*this < other) or (*this == other);}

bool D::operator>(const D& other) const {return !(*this <= other);}
bool D::operator>(const long long int& other) const {return !(*this <= other);}

bool D::operator>=(const D& other) const {return !(*this < other);}
bool D::operator>=(const long long int& other) const {return !(*this < other);}


D D::operator+(const D& other) const {
    if (_denom >= other._denom) {
        long long int new_num = _num + (other._num << (_denom - other._denom));
        return D(new_num, _denom);
    } else {
        long long int new_num = other._num + (_num << (other._denom - _denom));
        return D(new_num, other._denom);
    }
}

D D::operator+(const long long int& other) const {return D(_num + (other << _denom), _denom);}
D D::operator-() const {return D(-_num, _denom);}
D D::operator-(const D& other) const {return *this + (-other);}
D D::operator-(const long long int& other) const {return *this + (-other);}
D D::operator*(const D& other) const {return D(_num * other._num, _denom + other._denom);}
D D::operator*(const long long int& other) const {return D(_num * other, _denom);}

D D::pow(unsigned short n) const {
    return D(static_cast<long long int>(std::pow(_num, n)), _denom * n);
}


std::string D::to_string() const {return std::to_string(_num) + "/2^" + std::to_string(_denom);}

void D::print() const {std::cout << to_string() << std::endl;}

float D::to_float() const {return static_cast<float>(_num) / (1 << _denom);}

long double D::to_long_double() const {return static_cast<long double>(_num) / (1 << _denom);}
