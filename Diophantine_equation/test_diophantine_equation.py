import numpy as np
import pytest
from sympy import factorint, primerange, randprime

from diophantine_equation import *


def test_integer_fact():
    """
    Test the integer_fact() function.
    """
    for _ in range(10):
        num = np.random.randint(2, 1000)
        factors = integer_fact(num)

        assert np.prod([fact**exp for fact, exp in factors]) == num
        assert factorint(num) == {fact: exp for fact, exp in factors}


def test_xi_fact():
    """
    Test the xi_fact() function.
    """
    for _ in range(15):
        xi = Zsqrt2(*np.random.randint(-100, 100, 2))
        fact = xi_fact(xi)

        xi_calculated = np.prod([xi_i**mi for xi_i, mi in fact])
        assert are_sim_Zsqrt2(xi, xi_calculated)


@pytest.mark.parametrize("pi", primerange(3, 100))
def test_pi_fact_into_xi(pi):
    """
    Test the pi_fact_into_xi() function.
    """
    if pi % 8 == 1 or pi % 8 == 7:
        xi = pi_fact_into_xi(pi)
        assert xi * xi.conjugate() == pi

    else:
        # pi is its own factorization in Z[sqrt(2)]
        xi = pi_fact_into_xi(pi)
        assert pi % 8 == 3 or pi % 8 == 5
        assert xi is None

def test_xi_i_fact_into_ti():
    """
    Test the xi_i_fact_into_ti() function.
    """
    for i in range(15):
        if i == 0:
            p = 2
        else:
            p = randprime(3, 1000)  # A random prime number


        xi_i = pi_fact_into_xi(p)  # A random prime in Z[sqrt(2)]
        if xi_i is None:
            xi_i = Zsqrt2(p, 0)  # p is its own factorization in Z[sqrt(2)]

        xi_i_fact = xi_i_fact_into_ti(xi_i)

        if p % 8 == 7:
            # There is no solution to the equation xi_i ~ t_i * t_i†
            assert xi_i_fact is None

        else:
            xi_i_calculated = xi_i_fact * xi_i_fact.complex_conjugate()  # Instance of Domega
            assert xi_i_calculated.a == -xi_i_calculated.c  # Assert that the imaginary part is 0
            # Convert to the Zsqrt2 class
            xi_i_calculated_to_Zsqrt2 = Zsqrt2(xi_i_calculated.d.num, xi_i_calculated.c.num)
            assert are_sim_Zsqrt2(xi_i, xi_i_calculated_to_Zsqrt2)

def test_solve_usquare_eq_a_mod_p():
    """
    Test the solve_usquare_eq_a_mod_p() function.
    """
    for _ in range(15):
        p = 0
        while p %2 == 0 or p % 8 == 7:
            # There is no solution when p is even or p % 8 == 7
            p = randprime(1, 1000)
        
        a = 0

        if p % 4 == 1:
            a = 1
        
        elif p % 8 == 3:
            a = 2

        assert a != 0

        u = solve_usquare_eq_a_mod_p(a, p)

        assert (u**2) % p == -a + p


def test_gcd():
    """
    Test the gcd_Zomega() function.
    """
    for _ in range(10):
        a = Zomega(*np.random.randint(-10, 10, 4))  # First number
        b = Zomega(*np.random.randint(-10, 10, 4))  # Second number

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


def test_euclidean_div_Zomega():
    """
    Test the euclidean division function in Z[omega].
    """
    for _ in range(10):
        num = Zomega(*np.random.randint(-100, 100, 4))
        div = Zomega(*np.random.randint(-100, 100, 4))

        q, r = euclidean_div_Zomega(num, div)

        assert (q * div + r) == num


def test_euclidean_div_Zsqrt2():
    """
    Test the euclidean division function oin Z[sqrt2].
    """
    for _ in range(10):
        num = Zsqrt2(*np.random.randint(-100, 100, 2))
        div = Zsqrt2(*np.random.randint(-100, 100, 2))

        q, r = euclidean_div_Zsqrt2(num, div)

        assert (q * div + r) == num


def test_are_sim_Zsqrt2():
    """
    Test the are_sim_Zsqrt2() function.
    """
    lambd = Zsqrt2(1, 1)  # Lambda = 1 + sqrt(2)
    for _ in range(15):
        m = np.random.randint(0, 1)
        n = np.random.randint(0, 10)
        u = (-1) ** m * lambd**n  # Generate a random unit

        xi = Zsqrt2(*np.random.randint(-100, 100, 2))  # Random number in Z[sqrt(2)]
        xi_prime = u * xi  # Generate a similar number

        assert are_sim_Zsqrt2(xi, xi_prime)
        assert not are_sim_Zsqrt2(xi + 1, xi_prime)
        assert not are_sim_Zsqrt2(xi, 2 * xi_prime)


def test_is_unit_Zsqrt2():
    """
    Test the is_unit_Zsqrt2() function.
    """
    lambd = Zsqrt2(1, 1)  # Lambda = 1 + sqrt(2)
    for _ in range(15):
        m = np.random.randint(0, 1)
        n = np.random.randint(0, 10)
        u = (-1) ** m * lambd**n  # Generate a random unit

        assert is_unit_Zsqrt2(u)
        assert not is_unit_Zsqrt2(u + 1)
