# Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import itertools

import numpy as np
import pytest
from sympy import factorint, primerange

from qdecomp.rings import *
from qdecomp.utils.diophantine.diophantine_equation import *
from qdecomp.utils.diophantine.tonelli_shanks import tonelli_shanks_algo


@pytest.mark.parametrize("n", range(2, 20))
def test_integer_fact(n):
    """
    Test the integer_fact() function.
    """
    factors = integer_fact(n)

    assert np.prod([fact**exp for fact, exp in factors]) == n
    assert factorint(n) == {fact: exp for fact, exp in factors}


def test_integer_fact_errors():
    """
    Test the integer_fact() function.
    """
    with pytest.raises(ValueError, match="The number must be greater than 1."):
        integer_fact(1)

    with pytest.raises(ValueError, match="The number must be an integer. Got "):
        integer_fact(2.3)


@pytest.mark.parametrize("a, b", itertools.product(range(-1, 20), range(-5, 5)))
def test_xi_fact(a, b):
    """
    Test the xi_fact() function.
    """
    xi = Zsqrt2(a, b)
    fact = xi_fact(xi)

    xi_calculated = np.prod([xi_i**mi for xi_i, mi in fact])

    if xi == 0:  # xi is nul
        assert xi_calculated == 0

    else:
        assert are_sim_Zsqrt2(xi, xi_calculated)


@pytest.mark.parametrize("pi", primerange(0, 100))
def test_pi_fact_into_xi(pi):
    """
    Test the pi_fact_into_xi() function.
    """
    if pi == 2:
        xi = pi_fact_into_xi(pi)
        assert xi == Zsqrt2(0, 1) or xi == Zsqrt2(0, -1)

    elif pi % 8 == 1 or pi % 8 == 7:
        xi = pi_fact_into_xi(pi)
        assert xi * xi.sqrt2_conjugate() == pi

    else:
        # pi is its own factorization in Z[sqrt(2)]
        xi = pi_fact_into_xi(pi)
        assert pi % 8 == 3 or pi % 8 == 5
        assert xi is None


@pytest.mark.parametrize("n", primerange(0, 100))
def test_xi_i_fact_into_ti(n):
    """
    Test the xi_i_fact_into_ti() function.
    """
    xi_i = pi_fact_into_xi(n)  # A prime in Z[sqrt(2)]
    if xi_i is None:
        xi_i = Zsqrt2(n, 0)  # p is its own factorization in Z[sqrt(2)]

    xi_i_fact = xi_i_fact_into_ti(xi_i, True)

    if n % 8 == 7:
        # There is no solution to the equation xi_i ~ t_i * t_i†
        assert xi_i_fact is None

    else:
        xi_i_calculated = xi_i_fact * xi_i_fact.complex_conjugate()  # Instance of Zomega
        assert xi_i_calculated.a == -xi_i_calculated.c  # Assert that the imaginary part is 0

        # Convert to the Zsqrt2 class
        xi_i_calculated_to_Zsqrt2 = Zsqrt2.from_ring(xi_i_calculated)
        assert are_sim_Zsqrt2(xi_i, xi_i_calculated_to_Zsqrt2)


@pytest.mark.parametrize(
    "xi, is_prime",
    [
        (Zsqrt2(0, 1), True),
        (Zsqrt2(1, 0), False),
        (Zsqrt2(1, 1), False),
        (Zsqrt2(2, 0), False),
        (Zsqrt2(0, 2), False),
        (Zsqrt2(3, 0), True),
        (Zsqrt2(0, 3), False),
        (Zsqrt2(0, 15), False),
        (Zsqrt2(15, 0), False),
    ],
)
def test_xi_fact_into_ti_error(xi, is_prime):
    """
    Test the error raised by the xi_fact_into_ti() function.
    """
    if not is_prime:
        with pytest.raises(
            ValueError, match=r"The input argument must be a prime in Z\[sqrt\(2\)\]."
        ):
            xi_i_fact_into_ti(xi, check_prime=True)

    else:
        xi_i_fact_into_ti(xi, check_prime=True)
        assert True  # The code has run without error


@pytest.mark.parametrize("n", range(-100, 1000))
def test_is_square(n):
    """Test the is_square() function."""
    if n == 0:
        assert is_square(n)
        return

    assert is_square(n**2)
    assert not is_square(n**2 + 1)
    if n < 0:
        assert not is_square(n)


@pytest.mark.parametrize("p", primerange(0, 100))
def test_solve_usquare_eq_a_mod_p(p):
    """
    Test the solve_usquare_eq_a_mod_p() function.
    """
    if p % 2 == 0:  # There is no solution in that case
        return

    if p % 8 == 7:  # There is no solution in that case
        return

    if p % 4 == 1:
        a = 1

    elif p % 8 == 3:
        a = 2

    u = solve_usquare_eq_a_mod_p(a, p)

    assert (u**2) % p == -a + p


@pytest.mark.parametrize("a, p", itertools.product(range(-3, 4), primerange(3, 100)))
def test_tonelli_shanks_algo(a, p):
    """
    Test the tonelli_shanks_algo() function.
    """
    try:
        r = tonelli_shanks_algo(a, p)
    except ValueError:
        return

    # If a solution is found, check that it is correct
    assert (r**2) % p == a % p


@pytest.mark.parametrize(
    "a, p",
    [
        (-1, 1),
        (-2, 23),
    ],
)
def test_tonelli_shanks_algo_error(a, p):
    """
    Test the error raised by the tonelli_shanks_algo() function.
    """
    with pytest.raises(ValueError, match=f"a = {a} is not a quadratic residue modulo p = {p}."):
        tonelli_shanks_algo(a, p)


gcd_test_list = [
    Zomega(0, 0, 1, 0),  # 1
    Zomega(0, 1, 0, 0),  # i
    Zomega(0, 0, 1, 1),  # delta = 1 + omega
    Zomega(1, 2, 3, 4),
    Zomega(-1, -2, -3, -4),
    Zomega(-5, 2, 3, -4),
]


@pytest.mark.parametrize("a, b", itertools.product(gcd_test_list, gcd_test_list))
def test_gcd(a, b):
    """
    Test the gcd_Zomega() function.
    """
    gcd = gcd_Zomega(a, b)

    # Euclidean division of a and b by the gcd (with rest)
    div_a, ra = euclidean_div_Zomega(a, gcd)
    div_b, rb = euclidean_div_Zomega(b, gcd)

    assert (ra == 0) and (rb == 0)  # The rest should be 0

    # Multiplication of the computed quotients by the gcd
    mult_a = div_a * gcd
    mult_b = div_b * gcd

    assert (mult_a == a) and (mult_b == b)  # We should find the initial numbers

    assert gcd_Zomega(div_a, div_b) == 1  # The gcd of the quotients should be 1


omega_div_test_list = [
    Zomega(0, 0, 1, 0),  # 1
    Zomega(0, 1, 0, 0),  # i
    Zomega(0, 0, 1, 1),  # delta = 1 + omega
    Zomega(1, 2, 3, 4),
    Zomega(-1, -2, -3, -4),
    Zomega(-5, 2, 3, -4),
]


@pytest.mark.parametrize("num, div", itertools.product(omega_div_test_list, omega_div_test_list))
def test_euclidean_div_Zomega(num, div):
    """
    Test the euclidean division function in Z[omega].
    """
    q, r = euclidean_div_Zomega(num, div)
    assert (q * div + r) == num


sqrt2_div_test_list = [
    Zsqrt2(0, 1),  # 1
    Zsqrt2(1, 0),  # sqrt(2)
    Zsqrt2(1, 1),  # lambda = 1 + sqrt(2) (unit)
    Zsqrt2(1, 2),
    Zsqrt2(-1, 2),
    Zsqrt2(5, -1),
]


@pytest.mark.parametrize("num, div", itertools.product(sqrt2_div_test_list, sqrt2_div_test_list))
def test_euclidean_div_Zsqrt2(num, div):
    """
    Test the euclidean division function oin Z[sqrt2].
    """
    q, r = euclidean_div_Zsqrt2(num, div)
    assert (q * div + r) == num


sim_test_list = [
    Zsqrt2(1, 0),  # 1
    Zsqrt2(0, 1),  # sqrt(2)
    Zsqrt2(1, 1),  # lambda = 1 + sqrt(2) (unit)
    Zsqrt2(1, 2),
    Zsqrt2(-1, 2),
    Zsqrt2(5, -1),
]


@pytest.mark.parametrize("xi, m, n", itertools.product(sim_test_list, [0, 1], range(0, 4)))
def test_are_sim_Zsqrt2(xi, m, n):
    """
    Test the are_sim_Zsqrt2() function.

    xi: element of Z[sqrt(2)]
    m, n: define a unit u = (-1)^m * lambda^n
    """
    lambd = Zsqrt2(1, 1)  # Lambda = 1 + sqrt(2)
    lambd_inv = Zsqrt2(-1, 1)  # Lambda**-1 = -1 + sqrt(2)

    u = (-1) ** m * lambd**n  # Generate a unit
    u_ = (-1) ** m * lambd_inv**n  # Generate another unit

    xi_prime = u * xi  # Generate a similar number
    xi_prime_ = u_ * xi  # Generate another similar number

    assert are_sim_Zsqrt2(xi, xi_prime)
    assert not are_sim_Zsqrt2(xi + 1, xi_prime)
    assert not are_sim_Zsqrt2(xi, 2 * xi_prime)

    assert are_sim_Zsqrt2(xi, xi_prime_)
    assert not are_sim_Zsqrt2(xi + 1, xi_prime_)
    assert not are_sim_Zsqrt2(xi, 2 * xi_prime_)


@pytest.mark.parametrize("m, n", itertools.product([0, 1], range(0, 4)))
def test_is_unit_Zsqrt2(m, n):
    """
    Test the is_unit_Zsqrt2() function.
    """
    lambd = Zsqrt2(1, 1)  # Lambda = 1 + sqrt(2)
    lambd_inv = Zsqrt2(-1, 1)  # Lambda**-1 = -1 + sqrt(2)

    u = (-1) ** m * lambd**n  # Generate a unit
    u_ = (-1) ** m * lambd_inv**n  # Generate another unit

    assert is_unit_Zsqrt2(u)
    assert not is_unit_Zsqrt2(u + 1)
    assert not is_unit_Zsqrt2(u * 2)

    assert is_unit_Zsqrt2(u_)
    assert not is_unit_Zsqrt2(u_ + 1)
    assert not is_unit_Zsqrt2(u_ * 2)


@pytest.mark.parametrize("a, b", itertools.product(range(-1, 20), range(-5, 5)))
def test_solve_xi_sim_ttdag_in_z(a, b):
    """
    Test the solve_xi_sim_ttdag_in_z() function.
    """
    xi = Zsqrt2(a, b)
    t = solve_xi_sim_ttdag_in_z(xi)

    if t is not None:
        recombination = t * t.complex_conjugate()
        recombination_Zsqrt2 = Zsqrt2.from_ring(recombination)

        assert are_sim_Zsqrt2(xi, recombination_Zsqrt2)


@pytest.mark.parametrize(
    "a, a_, b, b_", itertools.product(range(-1, 10), range(0, 3), range(-5, 4), range(0, 3))
)
def test_solve_xi_eq_ttdag_in_d(a, a_, b, b_):
    """
    Test the solve_xi_eq_ttdag_in_d() function.
    """
    xi = Dsqrt2(D(a, a_), D(b, b_))

    t = solve_xi_eq_ttdag_in_d(xi)

    if t is not None:
        recombination = t * t.complex_conjugate()

        assert recombination == Domega.from_ring(xi)
        assert float(xi) >= 0 and float(xi.sqrt2_conjugate()) >= 0  # xi is doubly positive
