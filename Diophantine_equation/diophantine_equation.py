# Copyright 2022-2023 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
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

"""
This module solves the Diophantine equation xi = t† t for t where xi is given.

Input: xi in D[sqrt(2)]
Output: t in D[omega] such that xi = t† t


Example:

xi = Dsqrt2(D(13, 1), D(4, 1))  # Input: xi = 13/2 + 2 sqrt(2)
t = solve_xi_eq_ttdag_in_d(xi)  # Solution
tt = t * t.complex_conjugate()  # Product tt = t * t†
tt = tt.convert(Dsqrt2)         # Convert tt from D[omega] to D[sqrt(2)]

print(f"{xi = }")  # xi = 13/2+2√2
print(f"{t = }")   # t = - 2ω3 + 1/2ω2 + 3/2
print(f"{tt = }")  # tt = 13/2+2√2

"""

import sys

sys.path.append("../CliffordPlusT")

import numpy as np
from sympy import diophantine, symbols

from grid_algorithm_1D.Rings import *


def integer_fact(p):
    """
    Find the factorization of an integer p. This function returns a list of tuples (p_i, m_i) where
    p_i is a prime factor of p and m_i is its power.

    :param n: An integer
    :return: A list of tuples (p_i, m_i) containing the prime factors of n and their powers
    """
    if p < 2:
        raise ValueError("The number must be greater than 1.")

    if int(p) != p:
        raise ValueError(f"The number must be an integer. Got *.")

    n = p
    factors = []  # List of tuples (p_i, m_i)

    counter = 0
    while n % 2 == 0:
        counter += 1
        n = n // 2

    if counter > 0:
        factors.append((2, counter))

    # n must be odd at this point, so a skip of 2 ( i = i + 2) can be used
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        counter = 0

        # while i divides n, append i and divide n
        while n % i == 0:
            counter += 1
            n = n // i

        if counter > 0:
            factors.append((i, counter))

        if i > np.sqrt(n):
            break

    # If n != 1 at this point, n is a prime
    if n != 1:
        factors.append((n, 1))

    return factors


def xi_fact(xi):
    """
    Finds the factorization of xi (up to a prime) in the ring Z[sqrt(2)] where xi is an element of
    Z[sqrt(2)]. This function returns a list of tuples (xi_i, mi), where xi_i is a prime factor of
    xi in Z[sqrt(2)] and mi is its power.

    :param xi: An element of Z[sqrt(2)]
    :return: A list of tuples (xi_i, mi) containing the prime factors of xi and their powers
    """
    if xi == 0:  # 0 cannot be factorized
        return [
            (Zsqrt2(0, 0), 1),
        ]

    xi_fact_list = []
    p = (xi * xi.sqrt2_conjugate()).p

    if p == 1 or p == -1:  # xi is a unit
        return [
            (xi, 1),
        ]

    if p < 0:
        p = -p
        xi_fact_list.append((Zsqrt2(-1, 0), 1))

    pi_list = integer_fact(p)

    for pi, mi in pi_list:
        # If pi = 2, xi_i = sqrt(2)
        if pi == 2:
            xi_fact_list.append((Zsqrt2(0, 1), mi))

        # If pi % 8 == 1 or 7, we can factorize pi into xi_i where pi = xi_i * xi_i⋅
        elif pi % 8 == 1 or pi % 8 == 7:
            xi_i = pi_fact_into_xi(pi)

            # Determine wether we need to add xi_i or its conjugate to the factorization and how
            # many times
            xi_temp = xi
            for i in range(mi + 1):
                xi_temp, r = euclidean_div_Zsqrt2(xi_temp, xi_i)

                if r != 0:
                    break

            if i != 0:
                xi_fact_list.append((xi_i, i))
            if i != mi:
                xi_fact_list.append((xi_i.sqrt2_conjugate(), mi - i))

        # If pi % 8 == 3 or 5, pi is its own factorization in Z[sqrt(2)]
        # We need to append pi mi/2 times to the factorization of xi since pi = xi * xi
        else:
            xi_fact_list.append((Zsqrt2(pi, 0), mi // 2))

    return xi_fact_list


def pi_fact_into_xi(pi):
    """
    Solve the equation pi = xi_i * xi_i⋅ = a**2 - 2 * b**2 where ⋅ denotes the sqrt(2) conjugate.
    pi is a prime integer and xi_i = a + b * sqrt(2) is an element of Z[sqrt(2)]. pi has a
    factorization only if pi % 8 = 1 or 7 or pi = 2. In any other case, the function returns None.

    :param pi: A prime integer
    :return: An element of Z[sqrt(2)] for which pi = xi_i * xi_i⋅, or None if pi % 8 != 1 or 7
    """
    if pi == 2:
        return Zsqrt2(0, 1)

    if not (pi % 8 == 1 or pi % 8 == 7):
        return None

    a, b, t = symbols("a b t", integer=True)
    equation = a**2 - 2 * b**2 - pi
    solutions = diophantine(equation, t)

    a0, b0 = solutions.pop()  # Extract the first solution

    sub = {t: 0}
    xi = Zsqrt2(int(a0.subs(sub)), int(b0.subs(sub)))

    return xi


def xi_i_fact_into_ti(xi_i):
    """
    Solve the equation xi_i = t_i * t_i† where † denotes the complex conjugate. xi_i is a prime
    element in Z[sqrt(2)] and t_i is an element of Z[omega]. xi_i has a factorization only if
    pi % 8 = 1, 3 or 5, where pi = xi_i * xi_i⋅ or if pi = 2.

    Note: this function assumes xi_i is a prime element in Z[sqrt(2)]. No check is performed to
    verify this assumption.

    :param xi_i: A prime element in Z[sqrt(2)]
    :return: An element of Z[omega] for which xi_i = t_i * t_i†, or None if xi_i % 8 = 7
    """
    if xi_i == Zsqrt2(0, 1):  # xi_i = sqrt(2)
        delta = Zomega(0, 0, 1, 1)  # delta = 1 + omega
        return delta

    if xi_i.q == 0:  # xi_i is already a prime integer
        pi = xi_i.p
    else:
        pi = (xi_i * xi_i.sqrt2_conjugate()).p

    if pi % 4 == 1:
        u = solve_usquare_eq_a_mod_p(1, pi)
        xi_i_converted = Zomega(-xi_i.q, 0, xi_i.q, xi_i.p)
        ti = gcd_Zomega(xi_i_converted, Zomega(0, 1, 0, u))  # Second term: u + i
        return ti

    if pi % 8 == 3:  # xi_i = pi which is an integer in that case
        u = solve_usquare_eq_a_mod_p(2, pi)
        xi_i_converted = Zomega(0, 0, 0, xi_i.p)
        ti = gcd_Zomega(xi_i_converted, Zomega(1, 0, 1, u))  # Second term: u + i sqrt(2)
        return ti

    if pi % 8 == 7:
        return None


def solve_usquare_eq_a_mod_p(a, p):
    """
    Solve the diophantine equation u**2 = -a (mod p) where a, p and u are integers. This function
    returns the first integer solution of the equation. p is a prime.

    :param a: An integer
    :param p: A prime integer
    :return: The first integer solution of the equation u**2 = -a (mod p)
    """
    # The equation to solve is u**2 = q * p - a where q is an integer
    q, u, t = symbols("q u t", integer=True)
    equation = u**2 - q * p + a
    solutions = diophantine(equation, t)

    _, u0 = solutions.pop()  # Extract the first solution
    sol = int(u0.subs({t: 0}))

    return sol


def gcd_Zomega(x, y):
    """
    Find the greatest common divider (gcd) of x and y in the ring Z[omega]. x and y are elements of
    the ring Z[omega]. The algorithm implemented is the Euler method extended to the ring Z[omega].

    :param x: First number in Z[omega]
    :param y: Second number in Z[omega]
    :return: The greatest common divider of x and y in Z[omega]
    """
    a, b = x, y
    while b != 0:
        _, r = euclidean_div_Zomega(a, b)
        a, b = b, r

    return a


def euclidean_div_Zomega(num, div):
    """
    Compute the euclidean division of num by div where num and div are elements of Z[omega]. This
    function return q and r such that num = q * div + r.

    :param num: Number to be divided in Z[omega]
    :param div: Divider in Z[omega]
    :return: A tuple (q, r) where q is the result of the division and r is the rest
    """
    # div_sc = div.sqrt2_conjugate()  # Complex conjugate of the divider
    div_cc = div.complex_conjugate()  # Sqrt(2) conjugate of the divider

    div_div_cc = div * div_cc  # Product of the divider by its complex conjugate

    # Convert the denominator into an integer
    denom = (div_div_cc * div_div_cc.sqrt2_conjugate()).d.num
    # Apply the same multiplication on the numerator
    numer = num * div_cc * div_div_cc.sqrt2_conjugate()

    n = numer
    a, b, c, d = n.a.num, n.b.num, n.c.num, n.d.num  # Extract the coefficients of numer
    # Divide the coefficients by the integer denominator and round them
    a_, b_, c_, d_ = (
        round(a / denom),
        round(b / denom),
        round(c / denom),
        round(d / denom),
    )

    q = Zomega(a_, b_, c_, d_)  # Construction of the divider with the new coefficients
    r = num - q * div  # Calculation of the rest of the division

    return q, r


def euclidean_div_Zsqrt2(num, div):
    """
    Perform the euclidean division of num in Z[sqrt(2)]. This function returns q and r such that
    num = q * div + r.

    :param num: Number to be divided in Z[sqrt(2)]
    :param div: Divider in Z[sqrt(2)]
    :return: A tuple (q, r) where q is the result of the division and r is the rest
    """
    num_ = num * div.sqrt2_conjugate()
    den_ = (div * div.sqrt2_conjugate()).p

    a_, b_ = num_.p, num_.q
    a, b = round(a_ / den_), round(b_ / den_)

    q = Zsqrt2(a, b)
    r = num - q * div

    return q, r


def are_sim_Zsqrt2(x, y):
    """
    Determine if x ~ y. Equivalently, x ~ y if there exists a unit u such that x = u * y. x, y and u
    are elements of Z[sqrt(2)].

    :param x: First number in Z[sqrt(2)]
    :param y: Second number in Z[sqrt(2)]
    :return: True if x ~ y, False otherwise
    """
    # Test if y is a divider of x and y is a divider of x
    _, r1 = euclidean_div_Zsqrt2(x, y)
    _, r2 = euclidean_div_Zsqrt2(y, x)
    return (r1 == 0) and (r2 == 0)


def is_unit_Zsqrt2(x):
    """
    Determine if x is a unit in the ring Z[sqrt(2)].

    :param x: A number in Z[sqrt(2)]
    :return: True if x is a unit, False otherwise
    """
    integer = x * x.sqrt2_conjugate()
    return (integer == 1) or (integer == -1)


def solve_xi_sim_ttdag_in_z(xi):
    """
    Solve the equation xi ~ t * t† for t where † denotes the complex conjugate. xi is an element of
    Z[sqrt(2)] and t is an element of Z[omega]. This function returns the first solution of the
    equation. If no solution exists, the function returns None.

    :param xi: An element of Z[sqrt(2)]
    :return: An element of Z[omega] for which xi = t * t†, or None if no solution exists
    """
    xi_fact_list = xi_fact(xi)

    t = Zomega(0, 0, 0, 1)
    for xi_i, mi in xi_fact_list:
        if xi_i == -1:
            continue

        if mi % 2 == 0:  # For even exponents, xi_i ** mi = xi_i ** (mi // 2) * xi_i ** (mi // 2)
            factor = xi_i ** (mi // 2)
            d = factor.p
            c = factor.q
            a = -c
            t *= Zomega(a, 0, c, d)

        else:
            ti_i = xi_i_fact_into_ti(xi_i)
            if ti_i is None:
                return None

            t *= ti_i**mi

    return t


def solve_xi_eq_ttdag_in_d(xi):
    """
    Solve the equation xi = t * t† or t where † denotes the complex conjugate. xi is an element of
    D[sqrt(2)] and t is an element of D[omega]. This function returns the first solution of the
    equation. If no solution exists, the function returns None.

    :param xi: An element of D[sqrt(2)]
    :return: An element of D[omega] for which xi = t * t†, or None if no solution exists
    """
    # The equation only has a solution if xi is doubly positive, i.e. xi >= 0 and xi⋅ >= 0.
    if float(xi) < 0 or float(xi.sqrt2_conjugate()) < 0:
        return None

    l = (xi * xi.sqrt2_conjugate()).p.denom  # Greatest denominator power of 2
    xi_prime_temp = Dsqrt2(D(0, 0), D(1, 0)) ** l * xi  # xi_prime is in Z[sqrt(2)]
    xi_prime = Zsqrt2(xi_prime_temp.p.num, xi_prime_temp.q.num)  # Convert xi_prime to Z[sqrt(2)]

    s = solve_xi_sim_ttdag_in_z(xi_prime)  # Solve the equation xi' ~ s * s†
    if s is None:  # If there is no solution to the equation xi' ~ s * s†
        return None

    delta = Zomega(0, 0, 1, 1)  # delta = 1 + omega
    # delta**-1 = delta * lambda**-1 * omega**-1 / sqrt(2)
    delta_inv = (
        delta * Domega((-1, 0), (0, 0), (1, 0), (-1, 0)) * Domega((0, 0), (-1, 1), (0, 0), (1, 1))
    )

    delta_inv_l = delta_inv**l  # delta_l = delta ** l

    t = delta_inv_l * s  # t = delta**-l * s

    tt = t * t.complex_conjugate()  # tt = t * t†

    tt = tt.convert(Dsqrt2)

    # Find u such that xi = u * t * t†
    denom = (tt * tt.sqrt2_conjugate()).d  # Element of ring D
    u_temp = xi * tt.sqrt2_conjugate() * int(2**denom.denom)
    u = Zsqrt2(u_temp.p.num // denom.num, u_temp.q.num // denom.num)

    # u is of the form u = lambda**2n => n = ln(u) / 2 ln(lambda)
    n = round(np.log(float(u)) / (2 * np.log(float(Zsqrt2(1, 1)))))

    # v**2 = u => v = lambda**n
    if n > 0:
        v = Domega((-1, 0), (0, 0), (1, 0), (1, 0)) ** n  # lambda**n
    elif n == 0:
        v = Domega((0, 0), (0, 0), (0, 0), (1, 0))  # 1
    else:
        v = Domega((-1, 0), (0, 0), (1, 0), (-1, 0)) ** -n  # (lambda**-1)**n

    return t * v
