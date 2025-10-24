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
#include <utility>
#include <gtest/gtest.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <qdecomp/rings/cpp/Rings.hpp>


// Useful definitions
namespace mp = boost::multiprecision;
using TestingTypes = ::testing::Types<long long int, mp::cpp_int>;

template <typename T>
class ZomegaTests : public ::testing::Test {};
TYPED_TEST_SUITE(ZomegaTests, TestingTypes);


// Test addition in Zomega ring
TYPED_TEST(ZomegaTests, Addition) {
    std::vector<Zomega<TypeParam>> test_cases = {
        Zomega<TypeParam>(0, 0, 0, 0),
        Zomega<TypeParam>(0, 0, 0, 1),
        Zomega<TypeParam>(0, 0, 1, 0),
        Zomega<TypeParam>(0, 1, 0, 0),
        Zomega<TypeParam>(2, 1, 11, 2),
        Zomega<TypeParam>(-3, 2, -20, 1),
        Zomega<TypeParam>(-3127, 2, -20, 1)
    };

    for (Zomega<TypeParam> a : test_cases) {
    for (Zomega<TypeParam> b : test_cases) {
        Zomega<TypeParam> result = a + b;
        EXPECT_EQ(result.a(), a.a() + b.a());
        EXPECT_EQ(result.b(), a.b() + b.b());
        EXPECT_EQ(result.c(), a.c() + b.c());
        EXPECT_EQ(result.d(), a.d() + b.d());
    }
    }
}

// Test multiplication and to_Domega in Zomega ring
TYPED_TEST(ZomegaTests, Multiplication) {
    std::vector<Zomega<TypeParam>> test_cases = {
        Zomega<TypeParam>(0, 0, 0, 0),
        Zomega<TypeParam>(0, 0, 0, 1),
        Zomega<TypeParam>(0, 0, 1, 0),
        Zomega<TypeParam>(0, 1, 0, 0),
        Zomega<TypeParam>(2, 1, 11, 2),
        Zomega<TypeParam>(-3, 2, -20, 1),
        Zomega<TypeParam>(-3127, 2, -20, 1)
    };

    for (Zomega<TypeParam> a : test_cases) {
    for (Zomega<TypeParam> b : test_cases) {
        Zomega<TypeParam> result = a * b;

        Domega<TypeParam> copy_a = a.to_Domega();
        Domega<TypeParam> copy_b = b.to_Domega();

        // The Domega multiplication is properly tested.
        EXPECT_EQ(result, (copy_a * copy_b).to_Zomega());
    }
    }
}

// Test exponentiation and multiplication in Zomega ring
TYPED_TEST(ZomegaTests, Exponentiation) {
    std::vector<Zomega<TypeParam>> test_cases = {
        Zomega<TypeParam>(0, 0, 0, 0),
        Zomega<TypeParam>(0, 0, 0, 1),
        Zomega<TypeParam>(0, 0, 1, 0),
        Zomega<TypeParam>(0, 1, 0, 0),
        Zomega<TypeParam>(2, 1, 11, 2),
        Zomega<TypeParam>(-3, 2, -20, 1),
        Zomega<TypeParam>(-3127, 2, -20, 1)
    };

    for (Zomega<TypeParam> a : test_cases) {
    for (unsigned int exp : {0, 1, 2, 3, 4}) {
        Zomega<TypeParam> result = a.pow(exp);
        Zomega<TypeParam> brute_force(0, 0, 0, 1);
        
        for (unsigned int i = 0; i < exp; i++) {
            brute_force = brute_force * a;
        }

        EXPECT_EQ(result, brute_force);
    }
    }
}

// Test complex_conjugate and is_real in Zomega ring
TYPED_TEST(ZomegaTests, ComplexConjugate) {
    std::vector<Zomega<TypeParam>> test_cases = {
        Zomega<TypeParam>(0, 0, 0, 0),
        Zomega<TypeParam>(0, 0, 0, 1),
        Zomega<TypeParam>(0, 0, 1, 0),
        Zomega<TypeParam>(0, 1, 0, 0),
        Zomega<TypeParam>(2, 1, 11, 2),
        Zomega<TypeParam>(-3, 2, -20, 1),
        Zomega<TypeParam>(-3127, 2, -20, 1)
    };

    for (Zomega<TypeParam> a : test_cases) {
        Zomega<TypeParam> conj = a.complex_conjugate();
        Zomega<TypeParam> sum = a + conj;
        EXPECT_EQ(conj.complex_conjugate(), a);
        EXPECT_TRUE(sum.is_real());
    }
}

// Test sqrt2_conjugate in Zomega ring
TYPED_TEST(ZomegaTests, Sqrt2Conjugate) {
    std::vector<Zomega<TypeParam>> test_cases = {
        Zomega<TypeParam>(0, 0, 0, 0),
        Zomega<TypeParam>(0, 0, 0, 1),
        Zomega<TypeParam>(0, 0, 1, 0),
        Zomega<TypeParam>(0, 1, 0, 0),
        Zomega<TypeParam>(2, 1, 11, 2),
        Zomega<TypeParam>(-3, 2, -20, 1),
        Zomega<TypeParam>(-3127, 2, -20, 1)
    };

    for (Zomega<TypeParam> a : test_cases) {
        Zomega<TypeParam> conj = a.sqrt2_conjugate();
        Zomega<TypeParam> sum = a + conj;
        EXPECT_EQ(conj.sqrt2_conjugate(), a);
        EXPECT_EQ(sum.a(), 0);
        EXPECT_EQ(sum.c(), 0);
    }
}


// Test cast_Zomega in Zomega ring
TEST(ZomegaTests, CastZomega) {
    std::vector<std::pair<Zomega<long long int>, Zomega<mp::cpp_int>>> test_cases = {
        {Zomega<long long int>(0, 0, 0, 0), Zomega<mp::cpp_int>(0, 0, 0, 0)},
        {Zomega<long long int>(0, 0, 0, 1), Zomega<mp::cpp_int>(0, 0, 0, 1)},
        {Zomega<long long int>(0, 0, 1, 0), Zomega<mp::cpp_int>(0, 0, 1, 0)},
        {Zomega<long long int>(0, 1, 0, 0), Zomega<mp::cpp_int>(0, 1, 0, 0)},
        {Zomega<long long int>(2, 1, 11, 2), Zomega<mp::cpp_int>(2, 1, 11, 2)},
        {Zomega<long long int>(-3, 2, -20, 1), Zomega<mp::cpp_int>(-3, 2, -20, 1)},
        {Zomega<long long int>(-3127, 2, -20, 1), Zomega<mp::cpp_int>(-3127, 2, -20, 1)}
    };

    for (auto& [type1, type2] : test_cases) {
        Zomega<mp::cpp_int> cast_to_2 = cast_Zomega<long long int, mp::cpp_int>(type1);
        Zomega<long long int> cast_to_1 = cast_Zomega<mp::cpp_int, long long int>(type2);

        EXPECT_EQ(cast_to_2, type2);
        EXPECT_EQ(cast_to_1, type1);
    }
}

// Test euclidean_div in Zomega ring
TYPED_TEST(ZomegaTests, EuclideanDivision) {
    std::vector<Zomega<TypeParam>> test_cases = {
        Zomega<TypeParam>(0, 0, 0, 1),
        Zomega<TypeParam>(0, 0, 1, 0),
        Zomega<TypeParam>(0, 1, 0, 0),
        Zomega<TypeParam>(2, 1, 11, 2),
        Zomega<TypeParam>(-3, 2, -20, 1),
        Zomega<TypeParam>(-3127, 2, -20, 1)
    };

    for (Zomega<TypeParam> num : test_cases) {
    for (Zomega<TypeParam> div : test_cases) {
        Zomega<TypeParam> q(0, 0, 0, 0), r(0, 0, 0, 0);
        std::tie(q, r) = euclidean_div(num, div);
        EXPECT_EQ(num, q * div + r);

        if (num == div) {
            EXPECT_EQ(q.to_int(), 1);
            EXPECT_EQ(r.to_int(), 0);
        }
    }
    }
}

// Test gcd, euclidean_div and to_int in Zomega ring
TYPED_TEST(ZomegaTests, Gcd) {
    std::vector<Zomega<TypeParam>> test_cases = {
        Zomega<TypeParam>(0, 0, 0, 1),
        Zomega<TypeParam>(0, 0, 1, 0),
        Zomega<TypeParam>(0, 1, 0, 0),
        Zomega<TypeParam>(2, 1, 11, 2),
        Zomega<TypeParam>(-3, 2, -20, 1),
        Zomega<TypeParam>(-3127, 2, -20, 1)
    };

    for (Zomega<TypeParam> a : test_cases) {
    for (Zomega<TypeParam> b : test_cases) {
        Zomega<TypeParam> gcd_ = gcd(a, b);

        // Euclidean division of a and b by the gcd (with rest)
        Zomega<TypeParam> div_a(0, 0, 0, 0), ra(0, 0, 0, 0);
        Zomega<TypeParam> div_b(0, 0, 0, 0), rb(0, 0, 0, 0);
        std::tie(div_a, ra) = euclidean_div(a, gcd_);
        std::tie(div_b, rb) = euclidean_div(b, gcd_);

        // The rest should be zero
        EXPECT_EQ(ra.to_int(), 0);
        EXPECT_EQ(rb.to_int(), 0);

        // Multiplication of the computed quotients by the gcd
        Zomega<TypeParam> mult_a = div_a * gcd_;
        Zomega<TypeParam> mult_b = div_b * gcd_;

        // We should recover the initial numbers
        EXPECT_EQ(mult_a, a);
        EXPECT_EQ(mult_b, b);

        // The gcd of the quotients should be 1
        EXPECT_EQ(gcd(div_a, div_b).to_int(), 1);
    }
    }
}
