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
#include <qdecomp/utils/diophantine/cpp/diophantine_equation.hpp>


// Useful definitions
namespace mp = boost::multiprecision;
using TestingTypes = ::testing::Types<long long int, mp::cpp_int>;

template <typename T>
class DiophantineTests : public ::testing::Test {};
TYPED_TEST_SUITE(DiophantineTests, TestingTypes);
#define PRIMES {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}


// Test sqrt_generic
TYPED_TEST(DiophantineTests, SqrtGeneric) {
    for (TypeParam num : {0, 1, 2, 3, 4, 9, 16, 25, 26, 100, 121, 144, 1000, 10000}) {
        TypeParam result = sqrt_generic<TypeParam>(num);
        TypeParam expected = static_cast<TypeParam>(std::sqrt(static_cast<long double>(num)));
        EXPECT_EQ(result, expected);
    }
}

// Test is_square
TYPED_TEST(DiophantineTests, IsSquare) {
    for (TypeParam num : {0, 1, 2, 3, 4, 9, 16, 25, 26, 100, 121, 144, 1000, 10000}) {
        bool result = is_square<TypeParam>(num);
        bool expected = (sqrt_generic<TypeParam>(num) * sqrt_generic<TypeParam>(num) == num);
        EXPECT_EQ(result, expected);
    }
}

// Test solve_usquare_eq_a_mod_p
TYPED_TEST(DiophantineTests, SolveUSquareEqAModP) {
    for (TypeParam p : PRIMES) {
        if ((p % 2) == 0) { continue; }  // No solution if p is even
        if ((p % 8) == 7) { continue; }  // No solution if p % 8 == 7

        TypeParam a = 0;
        if ((p % 4) == 1) { a = 1; }
        else if ((p % 8) == 3) { a = 2; }

        if (legendre_symbol(a, p) != 1) { continue; }  // No solution if a is not a quadratic residue mod p

        TypeParam u = solve_usquare_eq_a_mod_p<TypeParam>(a, p);
        EXPECT_EQ((u * u) % p, -a + p);
    }
}

// Test int_fact
TYPED_TEST(DiophantineTests, IntFact) {
    for (TypeParam n = 2; n <= 100; n++) {
        auto fact_list = int_fact<TypeParam>(n);
        TypeParam prod = 1;
        for (auto [p, m] : fact_list) {
            for (unsigned int i = 0; i < m; i++) {
                prod *= p;
            }
        }
        EXPECT_EQ(prod, n);
    }
}

// Test xi_fact
TYPED_TEST(DiophantineTests, XiFact) {
    for (TypeParam a = -1; a <= 20; a++) {
    for (TypeParam b = -5; b <= 5; b++) {
        if (a == 0) { continue; }

        Zsqrt2<TypeParam> xi(a, b);
        auto fact_list = xi_fact<TypeParam>(xi);
        
        Zsqrt2<TypeParam> prod(1, 0);
        for (auto [pi, xi_i, m_i] : fact_list) {
            prod = prod * xi_i.pow(m_i);
        }

        EXPECT_TRUE(prod || xi);
    }
    }
}

// Test pi_fact_into_xi
TYPED_TEST(DiophantineTests, PiFactIntoXi) {
    for (TypeParam pi : PRIMES) {
        Zsqrt2<TypeParam> xi = pi_fact_into_xi<TypeParam>(pi);

        if (pi == 2) {
            EXPECT_TRUE(xi == Zsqrt2<TypeParam>(0, 1) || xi == Zsqrt2<TypeParam>(0, -1));
        } else if ((pi % 8) == 1 || (pi % 8) == 7) {
            EXPECT_EQ((xi * xi.sqrt2_conjugate()).to_int(), pi);
        } else {
            EXPECT_TRUE(((pi % 8) == 3 || (pi % 8) == 5));
            EXPECT_TRUE(xi == Zsqrt2<TypeParam>(pi, 0));
        }
    }
}

// Test xi_i_fact_into_ti
TYPED_TEST(DiophantineTests, XiIFactIntoTi) {
    for (TypeParam n : PRIMES) {
        Zsqrt2<TypeParam> xi_i = pi_fact_into_xi<TypeParam>(n);
        Zomega<TypeParam> xi_i_fact = xi_i_fact_into_ti<TypeParam>(n, xi_i);

        if ((n % 8) == 7) {
            EXPECT_EQ(xi_i_fact, Zomega<TypeParam>(0, 0, 0, 0));
        } else {
            Zsqrt2<TypeParam> recon_xi_i = (xi_i_fact * xi_i_fact.complex_conjugate()).to_Zsqrt2();
            EXPECT_TRUE(recon_xi_i || xi_i);
        }
    }
}

// Test solve_xi_sim_ttdag_in_z
TYPED_TEST(DiophantineTests, SolveXiSimTtDagInZ) {
    for (TypeParam a = -11; a <= 20; a++) {
    for (TypeParam b = -5; b <= 5; b++) {
        if (a == 0 and b == 0) { continue; }

        Zsqrt2<TypeParam> xi(a, b);
        Zomega<TypeParam> t = solve_xi_sim_ttdag_in_z<TypeParam>(xi);

        if (t != Zomega<TypeParam>(0, 0, 0, 0)) {
            Zsqrt2<TypeParam> recon_xi = (t * t.complex_conjugate()).to_Zsqrt2();
            EXPECT_TRUE(xi || recon_xi);  // The two values are similar / they only differ by a unit
        }
    }
    }
}

// Test solve_xi_eq_ttdag_in_d
TYPED_TEST(DiophantineTests, SolveXiEqTtDagInD) {
    for (TypeParam a = -1; a <= 10; a++) {
    for (unsigned int a_ = 0; a <= 3; a++) {
    for (TypeParam b = -5; b <= 4; b++) {
    for (unsigned int b_ = 0; b <= 3; b++) {
        Dsqrt2<TypeParam> xi(a, a_, b, b_);

        if (xi == Dsqrt2<TypeParam>(0, 0, 0, 0)) { continue; }
        
        Domega<TypeParam> t = solve_xi_eq_ttdag_in_d<TypeParam>(xi);

        if (t != Domega<TypeParam>(0, 0, 0, 0, 0, 0, 0, 0)) {
            Dsqrt2<TypeParam> recon_xi = (t * t.complex_conjugate()).to_Dsqrt2();
            
            EXPECT_EQ(recon_xi, xi);

            // xi is doubly positive
            EXPECT_TRUE(xi.to_long_double() > 0);
            EXPECT_TRUE(xi.sqrt2_conjugate().to_long_double() > 0);
        }
    }
    }
    }
    }
}
