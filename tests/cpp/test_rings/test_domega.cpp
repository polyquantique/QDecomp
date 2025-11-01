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
class DomegaTests : public ::testing::Test {};
TYPED_TEST_SUITE(DomegaTests, TestingTypes);


// Test addition, real and imag in Domega ring
TYPED_TEST(DomegaTests, Addition) {
    std::vector<Domega<TypeParam>> test_cases = {
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 1, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 1),
        Domega<TypeParam>(2, 1, 11, 2, -3, 0, 20, 1),
        Domega<TypeParam>(-3, 2, -20, 1, 2, 1, -11, 2)
    };

    for (Domega<TypeParam> a : test_cases) {
    for (Domega<TypeParam> b : test_cases) {
        Domega<TypeParam> result = a + b;
        EXPECT_EQ(result.real(), a.real() + b.real());
        EXPECT_EQ(result.imag(), a.imag() + b.imag());
    }
    }
}

// Test multiplication, real and imag in Domega ring
TYPED_TEST(DomegaTests, Multiplication) {
    std::vector<Domega<TypeParam>> test_cases = {
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 1, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 1),
        Domega<TypeParam>(2, 1, 11, 2, -3, 0, 20, 1),
        Domega<TypeParam>(-3, 2, -20, 1, 2, 1, -11, 2)
    };

    for (Domega<TypeParam> a : test_cases) {
    for (Domega<TypeParam> b : test_cases) {
        Dsqrt2<TypeParam> real = a.real() * b.real() - a.imag() * b.imag();
        Dsqrt2<TypeParam> imag = a.imag() * b.real() + a.real() * b.imag();
        Domega<TypeParam> result = a * b;
        EXPECT_EQ(result.real(), real);
        EXPECT_EQ(result.imag(), imag);
    }
    }
}

// Test exponentiation in Domega ring
TYPED_TEST(DomegaTests, Exponentiation) {
    std::vector<Domega<TypeParam>> test_cases = {
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 1, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 1),
        Domega<TypeParam>(2, 1, 11, 2, -3, 0, 20, 1),
        Domega<TypeParam>(-3, 2, -20, 1, 2, 1, -11, 2)
    };

    for (Domega<TypeParam> a : test_cases) {
    for (unsigned int exp : {0, 1, 2, 3, 4}) {
        Domega<TypeParam> result = a.pow(exp);
        Domega<TypeParam> brute_force(0, 0, 0, 0, 0, 0, 1, 0);
        
        for (unsigned int i = 0; i < exp; i++) {
            brute_force = brute_force * a;
        }

        EXPECT_EQ(result, brute_force);
    }
    }
}

// Test complex_conjugate in Domega ring
TYPED_TEST(DomegaTests, ComplexConjugate) {
    std::vector<Domega<TypeParam>> test_cases = {
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 1, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 1),
        Domega<TypeParam>(2, 1, 11, 2, -3, 0, 20, 1),
        Domega<TypeParam>(-3, 2, -20, 1, 2, 1, -11, 2)
    };

    for (Domega<TypeParam> a : test_cases) {
        Domega<TypeParam> conj = a.complex_conjugate();
        Domega<TypeParam> sum = a + conj;
        EXPECT_EQ(conj.complex_conjugate(), a);
        EXPECT_EQ(sum.imag().to_int(), 0);
    }
}

// Test sqrt2_conjugate in Domega ring
TYPED_TEST(DomegaTests, Sqrt2Conjugate) {
    std::vector<Domega<TypeParam>> test_cases = {
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 1, 0),
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 1),
        Domega<TypeParam>(2, 1, 11, 2, -3, 0, 20, 1),
        Domega<TypeParam>(-3, 2, -20, 1, 2, 1, -11, 2)
    };

    for (Domega<TypeParam> a : test_cases) {
        Domega<TypeParam> conj = a.sqrt2_conjugate();
        Domega<TypeParam> sum = a + conj;
        EXPECT_EQ(conj.sqrt2_conjugate(), a);
        EXPECT_EQ(sum.a(), 0);
        EXPECT_EQ(sum.c(), 0);
    }
}

// Test sde, exponentiation, complex_conjugate and to_Dsqrt2 in Domega ring
TYPED_TEST(DomegaTests, Sde) {
    std::vector<Domega<TypeParam>> test_cases = {
        Domega<TypeParam>(0, 0, 0, 0, 0, 0, 1, 0),
        Domega<TypeParam>(2, 1, 11, 2, -3, 0, 20, 1),
        Domega<TypeParam>(-3, 2, -20, 1, 2, 1, -11, 2),
        Domega<TypeParam>(2, 0, 4, 0, -8, 0, -20, 0),
        Domega<TypeParam>(1, 0, 3, 0, -8, 0, -20, 0)
    };
    Domega<TypeParam> sqrt2(-1, 0, 0, 0, 1, 0, 0, 0);
    Domega<TypeParam> sqrt2_inv(-1, 1, 0, 0, 1, 1, 0, 0);

    for (Domega<TypeParam> a : test_cases) {
        int sde = a.sde();

        if (sde >= 0) { EXPECT_TRUE((a * sqrt2.pow(sde)).is_Zomega()); }
        else { EXPECT_TRUE((a * sqrt2_inv.pow(-sde)).is_Zomega()); }

        if (sde > 0) { EXPECT_FALSE((a * sqrt2.pow(sde-1)).is_Zomega()); }
        else { EXPECT_FALSE((a * sqrt2_inv.pow(-sde+1)).is_Zomega()); }
    }
}


// Test cast_Domega in Domega ring
TEST(DomegaTests, CastDomega) {
    std::vector<std::pair<Domega<long long int>, Domega<mp::cpp_int>>> test_cases = {
        {Domega<long long int>(0, 0, 0, 0, 0, 0, 0, 0), Domega<mp::cpp_int>(0, 0, 0, 0, 0, 0, 0, 0)},
        {Domega<long long int>(0, 0, 0, 0, 0, 0, 1, 0), Domega<mp::cpp_int>(0, 0, 0, 0, 0, 0, 1, 0)},
        {Domega<long long int>(0, 0, 0, 0, 1, 0, 0, 0), Domega<mp::cpp_int>(0, 0, 0, 0, 1, 0, 0, 0)},
        {Domega<long long int>(0, 0, 1, 0, 0, 0, 0, 0), Domega<mp::cpp_int>(0, 0, 1, 0, 0, 0, 0, 0)},
        {Domega<long long int>(2, 1, 1, 3, 11, 0, 2, 0), Domega<mp::cpp_int>(2, 1, 1, 3, 11, 0, 2, 0)},
        {Domega<long long int>(-3, 101, 2, 0, -20, 0, 1, 21), Domega<mp::cpp_int>(-3, 101, 2, 0, -20, 0, 1, 21)},
        {Domega<long long int>(-3127, 0, 2, 1, -20, 2, 1, 3), Domega<mp::cpp_int>(-3127, 0, 2, 1, -20, 2, 1, 3)}
    };

    for (auto& [type1, type2] : test_cases) {
        Domega<mp::cpp_int> cast_to_2 = cast_Domega<long long int, mp::cpp_int>(type1);
        Domega<long long int> cast_to_1 = cast_Domega<mp::cpp_int, long long int>(type2);

        EXPECT_EQ(cast_to_2, type2);
        EXPECT_EQ(cast_to_1, type1);
    }
}
