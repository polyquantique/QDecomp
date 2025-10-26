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
#include <qdecomp/utils/diophantine/cpp/tonelli_shanks.hpp>


// Useful definitions
namespace mp = boost::multiprecision;
using TestingTypes = ::testing::Types<long long int, mp::cpp_int>;

template <typename T>
class TonelliShanksTests : public ::testing::Test {};
TYPED_TEST_SUITE(TonelliShanksTests, TestingTypes);
#define PRIMES {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}


// Test mod_pow
TYPED_TEST(TonelliShanksTests, ModPow) {
    for (TypeParam base = -20; base <= 20; base++) {
    for (TypeParam exp = 0; exp <= 5; exp++) {
    for (TypeParam mod = 1; mod <= 10; mod++) {
        TypeParam result = mod_pow<TypeParam>(base, exp, mod);

        TypeParam expected = 1;
        for (TypeParam i = 0; i < exp; i++) {
            expected = expected * base;
        }
        expected = expected % mod;
        if (expected < 0) { expected += mod; }

        EXPECT_EQ(result, expected);
        EXPECT_GE(result, 0);
    }
    }
    }
}

// Test legendre_symbol
TYPED_TEST(TonelliShanksTests, LegendreSymbol) {
    for (TypeParam a = 0; a <= 100; a++) {
    for (TypeParam p : PRIMES) {
        int result = legendre_symbol<TypeParam>(a, p);

        EXPECT_TRUE(result == 0 || result == 1 || result == -1);
        
        if (a % p == 0) {
            EXPECT_EQ(result, 0);
        } else {
            bool is_residue = false;
            for (TypeParam x = 0; x < p; x++) {
                if ((x * x) % p == (a % p)) {
                    is_residue = true;
                    break;
                }
            }
            if (is_residue) {
                EXPECT_EQ(result, 1);
            } else {
                EXPECT_EQ(result, -1);
            }
        }
    }
    }
}

// Test tonelli_shanks_algo
TYPED_TEST(TonelliShanksTests, TonelliShanksAlgo) {
    for (TypeParam a = -3; a <= 4; a++) {
    for (TypeParam p : PRIMES) {
        int legendre = legendre_symbol<TypeParam>(a, p);

        if (legendre != 1 and p != 2) {  // No solution exists
            EXPECT_THROW(tonelli_shanks_algo<TypeParam>(a, p), std::runtime_error);
        } else {
            TypeParam r = tonelli_shanks_algo<TypeParam>(a, p);
            TypeParam mod = a % p;
            if (mod < 0) { mod += p; }  // Ensure mod is non-negative
            EXPECT_EQ((r * r) % p, mod);  // Ensure r^2 ≡ a (mod p)
        }
    }
    }
}
