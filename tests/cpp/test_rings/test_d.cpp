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
#include <gtest/gtest.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <qdecomp/rings/cpp/Rings.hpp>


// Useful definitions
namespace mp = boost::multiprecision;
using TestingTypes = ::testing::Types<long long int, mp::cpp_int>;

template <typename T>
class DRingTests : public ::testing::Test {};
TYPED_TEST_SUITE(DRingTests, TestingTypes);


// Test addition in D ring
TYPED_TEST(DRingTests, Addition) {
    for (TypeParam a_num : {0, 2, -3}) {
    for (TypeParam b_num : {0, 11, -20}) {
    for (unsigned int a_denom : {0, 1, 2}) {
    for (unsigned int b_denom : {0, 1, 2}) {
        D<TypeParam> a(a_num, a_denom);
        D<TypeParam> b(b_num, b_denom);
        D<TypeParam> result = a + b;
        long double expected = a.to_long_double() + b.to_long_double();
        EXPECT_NEAR(result.to_long_double(), expected, 1e-9);
    }
    }
    }
    }
}

// Test multiplication in D ring
TYPED_TEST(DRingTests, Multiplication) {
    for (TypeParam a_num : {0, 2, -3}) {
    for (TypeParam b_num : {0, 11, -20}) {
    for (unsigned int a_denom : {0, 1, 2}) {
    for (unsigned int b_denom : {0, 1, 2}) {
        D<TypeParam> a(a_num, a_denom);
        D<TypeParam> b(b_num, b_denom);
        D<TypeParam> result = a * b;
        long double expected = a.to_long_double() * b.to_long_double();
        EXPECT_NEAR(result.to_long_double(), expected, 1e-9);
    }
    }
    }
    }
}

// Test equality and inequality in D ring
TYPED_TEST(DRingTests, Equality) {
    for (TypeParam a_num : {0, 2, -3}) {
    for (TypeParam b_num : {0, 11, -20}) {
    for (unsigned int a_denom : {0, 1, 2}) {
    for (unsigned int b_denom : {0, 1, 2}) {
        D<TypeParam> a(a_num, a_denom);
        D<TypeParam> b(b_num, b_denom);

        long double diff = (a - b).to_long_double();
        bool equal = (std::abs(diff) < 1e-9);

        if (equal) {
            EXPECT_TRUE(a == b);
            EXPECT_FALSE(a != b);
        } else {
            EXPECT_FALSE(a == b);
            EXPECT_TRUE(a != b);
        }
    }
    }
    }
    }
}

// Test exponentiation in D ring
TEST(DRingTests, Exponentiation) {
    for (long long int a_num : {0, 2, -3}) {
    for (unsigned int a_denom : {0, 1, 2}) {
    for (unsigned int exp : {0, 1, 2, 3}) {
        D<long long int> a(a_num, a_denom);
        D<long long int> result = a.pow(exp);
        long double expected = std::pow(a.to_long_double(), exp);
        EXPECT_NEAR(result.to_long_double(), expected, 1e-9);
    }
    }
    }
}

// Test the reduce method in D ring
TYPED_TEST(DRingTests, Reduce) {
    for (TypeParam a_num : {0, 2, -3, 4, -16}) {
    for (unsigned int a_denom : {0, 1, 2}) {
        D<TypeParam> a(a_num, a_denom);  // The reduce method is called in the constructor
        bool must_be_reduced = a_num % 2 == 0 && a_denom > 0 && a_num != 0;
        if (must_be_reduced) {EXPECT_NE(a.num(), a_num);}
        else {EXPECT_EQ(a.num(), a_num);}
    }
    }
}
