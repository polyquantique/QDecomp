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
class Dsqrt2Tests : public ::testing::Test {};
TYPED_TEST_SUITE(Dsqrt2Tests, TestingTypes);


// Test addition and to_long_double in Dsqrt2 ring
TYPED_TEST(Dsqrt2Tests, Addition) {
    std::vector<Dsqrt2<TypeParam>> test_cases = {
        Dsqrt2<TypeParam>(0, 0, 0, 0),
        Dsqrt2<TypeParam>(1, 0, 0, 0),
        Dsqrt2<TypeParam>(0, 1, 0, 0),
        Dsqrt2<TypeParam>(2, 1, 11, 1),
        Dsqrt2<TypeParam>(-3, 2, -20, 2),
        Dsqrt2<TypeParam>(-100, 5, 50, 3)
    };

    for (Dsqrt2<TypeParam> a : test_cases) {
    for (Dsqrt2<TypeParam> b : test_cases) {
        Dsqrt2<TypeParam> result = a + b;
        EXPECT_NEAR(result.to_long_double(), a.to_long_double() + b.to_long_double(), 1e-9);
    }
    }
}

// Test multiplication and to_long_double in Dsqrt2 ring
TYPED_TEST(Dsqrt2Tests, Multiplication) {
    std::vector<Dsqrt2<TypeParam>> test_cases = {
        Dsqrt2<TypeParam>(0, 0, 0, 0),
        Dsqrt2<TypeParam>(1, 0, 0, 0),
        Dsqrt2<TypeParam>(0, 1, 0, 0),
        Dsqrt2<TypeParam>(2, 1, 11, 1),
        Dsqrt2<TypeParam>(-3, 2, -20, 2),
        Dsqrt2<TypeParam>(-100, 5, 50, 3)
    };

    for (Dsqrt2<TypeParam> a : test_cases) {
    for (Dsqrt2<TypeParam> b : test_cases) {
        Dsqrt2<TypeParam> result = a * b;
        EXPECT_NEAR(result.to_long_double(), a.to_long_double() * b.to_long_double(), 1e-9);
    }
    }
}

// Test exponentiation and to_long_double in Dsqrt2 ring
TYPED_TEST(Dsqrt2Tests, Exponentiation) {
    std::vector<Dsqrt2<TypeParam>> test_cases = {
        Dsqrt2<TypeParam>(0, 0, 0, 0),
        Dsqrt2<TypeParam>(1, 0, 0, 0),
        Dsqrt2<TypeParam>(0, 1, 0, 0),
        Dsqrt2<TypeParam>(2, 1, 11, 1),
        Dsqrt2<TypeParam>(-3, 2, -20, 2),
        Dsqrt2<TypeParam>(-100, 5, 50, 3)
    };

    for (Dsqrt2<TypeParam> a : test_cases) {
    for (unsigned int exp : {0, 1, 2, 3, 4}) {
        Dsqrt2<TypeParam> result = a.pow(exp);
        EXPECT_NEAR(result.to_long_double(), std::pow(a.to_long_double(), exp), 1e-9);
    }
    }
}

// Test sqrt2_multiply and equality in Dsqrt2 ring
TYPED_TEST(Dsqrt2Tests, Sqrt2Multiply) {
    std::vector<Dsqrt2<TypeParam>> test_cases = {
        Dsqrt2<TypeParam>(0, 0, 0, 0),
        Dsqrt2<TypeParam>(1, 0, 0, 0),
        Dsqrt2<TypeParam>(0, 1, 0, 0),
        Dsqrt2<TypeParam>(2, 1, 11, 1),
        Dsqrt2<TypeParam>(-3, 2, -20, 2),
        Dsqrt2<TypeParam>(-100, 5, 50, 3)
    };
    Dsqrt2<TypeParam> sqrt2(0, 0, 1, 0);

    for (Dsqrt2<TypeParam> a : test_cases) {
    for (unsigned int exp : {0, 1, 2, 3, 4}) {
        Dsqrt2<TypeParam> result = a.sqrt2_multiply(exp);
        EXPECT_EQ(result, a * sqrt2.pow(exp));
    }
    }
}

// Test sqrt2_conjugate and equality in Dsqrt2 ring
TYPED_TEST(Dsqrt2Tests, Sqrt2Conjugate) {
    std::vector<Dsqrt2<TypeParam>> test_cases = {
        Dsqrt2<TypeParam>(0, 0, 0, 0),
        Dsqrt2<TypeParam>(1, 0, 0, 0),
        Dsqrt2<TypeParam>(0, 1, 0, 0),
        Dsqrt2<TypeParam>(2, 1, 11, 1),
        Dsqrt2<TypeParam>(-3, 2, -20, 2),
        Dsqrt2<TypeParam>(-100, 5, 50, 3)
    };

    for (Dsqrt2<TypeParam> a : test_cases) {
        Dsqrt2<TypeParam> conj = a.sqrt2_conjugate();
        Dsqrt2<TypeParam> sum = a + conj;
        EXPECT_EQ(conj.sqrt2_conjugate(), a);
        EXPECT_TRUE(sum.is_D());
    }
}


// Test cast_Dsqrt2 in Dsqrt2 ring
TEST(Dsqrt2Tests, CastDsqrt2) {
    std::vector<std::pair<Dsqrt2<long long int>, Dsqrt2<mp::cpp_int>>> test_cases = {
        {Dsqrt2<long long int>(0, 0, 0, 0), Dsqrt2<mp::cpp_int>(0, 0, 0, 0)},
        {Dsqrt2<long long int>(0, 0, 1, 1), Dsqrt2<mp::cpp_int>(0, 0, 1, 1)},
        {Dsqrt2<long long int>(1, 0, 0, 0), Dsqrt2<mp::cpp_int>(1, 0, 0, 0)},
        {Dsqrt2<long long int>(2, 0, 11, 3), Dsqrt2<mp::cpp_int>(2, 0, 11, 3)},
        {Dsqrt2<long long int>(2, 1, -20, 0), Dsqrt2<mp::cpp_int>(2, 1, -20, 0)},
        {Dsqrt2<long long int>(-3127, 3, 1, 2), Dsqrt2<mp::cpp_int>(-3127, 3, 1, 2)}
    };

    for (auto& [type1, type2] : test_cases) {
        Dsqrt2<mp::cpp_int> cast_to_2 = cast_Dsqrt2<long long int, mp::cpp_int>(type1);
        Dsqrt2<long long int> cast_to_1 = cast_Dsqrt2<mp::cpp_int, long long int>(type2);

        EXPECT_EQ(cast_to_2, type2);
        EXPECT_EQ(cast_to_1, type1);
    }
}
