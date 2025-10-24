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


// Include Libraries
#include <vector>
#include <gtest/gtest.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <qdecomp/rings/cpp/Rings.hpp>


// Useful definitions
namespace mp = boost::multiprecision;
using TestingTypes = ::testing::Types<long long int, mp::cpp_int>;

template <typename T>
class Zsqrt2Tests : public ::testing::Test {};
TYPED_TEST_SUITE(Zsqrt2Tests, TestingTypes);


// Test addition and to_long_double in Zsqrt2 ring
TYPED_TEST(Zsqrt2Tests, Addition) {
    std::vector<Zsqrt2<TypeParam>> test_cases = {
        Zsqrt2<TypeParam>(0, 0),
        Zsqrt2<TypeParam>(1, 0),
        Zsqrt2<TypeParam>(0, 1),
        Zsqrt2<TypeParam>(2, 1),
        Zsqrt2<TypeParam>(-3, 2),
        Zsqrt2<TypeParam>(-101, 7)
    };

    for (Zsqrt2<TypeParam> a : test_cases) {
    for (Zsqrt2<TypeParam> b : test_cases) {
        Zsqrt2<TypeParam> result = a + b;
        EXPECT_NEAR(result.to_long_double(), a.to_long_double() + b.to_long_double(), 1e-9);
    }
    }
}

// Test multiplication and to_long_double in Zsqrt2 ring
TYPED_TEST(Zsqrt2Tests, Multiplication) {
    std::vector<Zsqrt2<TypeParam>> test_cases = {
        Zsqrt2<TypeParam>(0, 0),
        Zsqrt2<TypeParam>(1, 0),
        Zsqrt2<TypeParam>(0, 1),
        Zsqrt2<TypeParam>(2, 1),
        Zsqrt2<TypeParam>(-3, 2),
        Zsqrt2<TypeParam>(-101, 7)
    };

    for (Zsqrt2<TypeParam> a : test_cases) {
    for (Zsqrt2<TypeParam> b : test_cases) {
        Zsqrt2<TypeParam> result = a * b;
        EXPECT_NEAR(result.to_long_double(), a.to_long_double() * b.to_long_double(), 1e-9);
    }
    }
}

// Test exponentiation and to_long_double in Zsqrt2 ring
TYPED_TEST(Zsqrt2Tests, Exponentiation) {
    std::vector<Zsqrt2<TypeParam>> test_cases = {
        Zsqrt2<TypeParam>(0, 0),
        Zsqrt2<TypeParam>(1, 0),
        Zsqrt2<TypeParam>(0, 1),
        Zsqrt2<TypeParam>(2, 1),
        Zsqrt2<TypeParam>(-3, 2),
        Zsqrt2<TypeParam>(-101, 7)
    };

    for (Zsqrt2<TypeParam> a : test_cases) {
    for (unsigned int exp : {0, 1, 2, 3, 4}) {
        Zsqrt2<TypeParam> result = a.pow(exp);
        EXPECT_NEAR(result.to_long_double(), std::pow(a.to_long_double(), exp), 1e-9);
    }
    }
}

// Test sqrt2_conjugate and equality in Zsqrt2 ring
TYPED_TEST(Zsqrt2Tests, Sqrt2Conjugate) {
    std::vector<Zsqrt2<TypeParam>> test_cases = {
        Zsqrt2<TypeParam>(0, 0),
        Zsqrt2<TypeParam>(1, 0),
        Zsqrt2<TypeParam>(0, 1),
        Zsqrt2<TypeParam>(2, 1),
        Zsqrt2<TypeParam>(-3, 2),
        Zsqrt2<TypeParam>(-101, 7)
    };

    for (Zsqrt2<TypeParam> a : test_cases) {
        Zsqrt2<TypeParam> conj = a.sqrt2_conjugate();
        Zsqrt2<TypeParam> sum = a + conj;
        EXPECT_EQ(conj.sqrt2_conjugate(), a);
        EXPECT_TRUE(sum.is_int());
    }
}

// Test unit_reduce, euclidean_div and to_int in Zsqrt2 ring
TYPED_TEST(Zsqrt2Tests, UnitReduce) {
    std::vector<Zsqrt2<TypeParam>> test_cases = {
        Zsqrt2<TypeParam>(1, 0),
        Zsqrt2<TypeParam>(2, 1),
        Zsqrt2<TypeParam>(-3, 2),
        Zsqrt2<TypeParam>(-101, 7),
        Zsqrt2<TypeParam>(1, 2),
        Zsqrt2<TypeParam>(13, 5),
        Zsqrt2<TypeParam>(-17, 2),
        Zsqrt2<TypeParam>(-10001, 2),
        Zsqrt2<TypeParam>(-17, 200013007),
    };

    for (Zsqrt2<TypeParam> a : test_cases) {
        Zsqrt2<TypeParam> copy = a;
        a.unit_reduce();
        Zsqrt2<TypeParam> reminder = std::get<1>(euclidean_div(copy, a));
        EXPECT_EQ(reminder.to_int(), 0);
    }
}

// Test euclidean_div and to_int in Zsqrt2 ring
TYPED_TEST(Zsqrt2Tests, EuclideanDiv) {
    std::vector<Zsqrt2<TypeParam>> test_cases = {
        Zsqrt2<TypeParam>(1, 0),
        Zsqrt2<TypeParam>(2, 1),
        Zsqrt2<TypeParam>(-3, 2),
        Zsqrt2<TypeParam>(-101, 7),
        Zsqrt2<TypeParam>(1, 2),
        Zsqrt2<TypeParam>(13, 5),
        Zsqrt2<TypeParam>(-17, 2),
    };

    for (Zsqrt2<TypeParam> num : test_cases) {
    for (Zsqrt2<TypeParam> div : test_cases) {
        Zsqrt2<TypeParam> q(0, 0), r(0, 0);
        std::tie(q, r) = euclidean_div(num, div);
        EXPECT_EQ(num, q * div + r);

        if (num == div) {
            EXPECT_EQ(q.to_int(), 1);
            EXPECT_EQ(r.to_int(), 0);
        }
    }
    }
}
