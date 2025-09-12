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

r"""
This module solves the Diophantine equation :math:`\xi = t \cdot t^\dagger` for :math:`t \in \mathbb{D}[\omega]` where :math:`\xi
\in \mathbb{D}[\sqrt{2}]` is given. The solution :math:`t` is returned if it exists, or `None` otherwise. This
module is an implementation of the algorithm presented in Section 6 and Appendix C of :cite:`diophantine_ross`.

| **Input:** :math:`\xi \in \mathbb{D}[\sqrt{2}]`
| **Output:** :math:`t \in \mathbb{D}[\omega]`, the solution to the equation :math:`\xi = t \cdot t^\dagger`, or `None` if no solution exists for the specified :math:`\xi`

**Example:**

.. code-block:: python

    >>> from qdecomp.rings import *
    >>> from qdecomp.utils.diophantine import solve_xi_eq_ttdag_in_d

    # Solve a Diophantine equation that has a solution
    >>> xi = Dsqrt2(D(13, 1), D(4, 1))  # Input
    >>> t = solve_xi_eq_ttdag_in_d(xi)  # Compute the solution

    >>> print(f"{xi = }")
    xi = 13/2^1+2/2^0√2
    >>> print(f"{t = }")
    t = -2/2^0ω3 + 1/2^1ω2 + 0/2^0ω + 3/2^1

    # Check the solution
    >>> xi_calculated_in_Domega = t * t.complex_conjugate()        # Calculate (t * t†)
    >>> xi_calculated = Dsqrt2.from_ring(xi_calculated_in_Domega)  # Convert the result from D[omega] to D[sqrt(2)]

    >>> print(f"{xi_calculated = }")
    xi_calculated = 13/2^1+2/2^0√2
    >>> print(f"{xi == xi_calculated = }")
    xi == xi_calculated = True

    # Solve a Diophantine equation that doesn't have any solution
    >>> xi = Dsqrt2(D(9, 1), D(3, 1))   # Input
    >>> t = solve_xi_eq_ttdag_in_d(xi)  # Compute the solution

    >>> print(f"{xi = }")
    xi = 9/2^1+3/2^1√2
    >>> print(f"{t = }")
    t = None
"""

from math import log, sqrt
from typing import Union

from qdecomp.rings import *
from qdecomp.utils.diophantine.tonelli_shanks import tonelli_shanks_algo


# ----------------------------- #
# Functions for Rings algebra   #
# ----------------------------- #
def gcd_Zomega(x: Zomega, y: Zomega) -> Zomega:
    r"""
    Find the greatest common divider (or :math:`gcd`) of :math:`x` and :math:`y` in the ring :math:`\mathbb{Z}[\omega]`. :math:`x` and :math:`y` are elements of
    the ring :math:`\mathbb{Z}[\omega]`. The algorithm implemented is the Euler method extended to the ring
    :math:`\mathbb{Z}[\omega]`.

    Args:
        x (Zomega): First number
        y (Zomega): Second number

    Returns:
        Zomega: The greatest common divider of :math:`x` and :math:`y`
    """
    a, b = x, y
    while b != 0:
        _, r = euclidean_div_Zomega(a, b)
        a, b = b, r

    return a


def euclidean_div_Zomega(num: Zomega, div: Zomega) -> tuple[Zomega, Zomega]:
    r"""
    Compute the euclidean division of :math:`num` by :math:`div`, where :math:`num` and :math:`div` are elements of :math:`\mathbb{Z}[\omega]`. This
    function return :math:`q` and :math:`r` such that :math:`num = q \cdot div + r`.

    Args:
        num (Zomega): Number to be divided
        div (Zomega): Divider

    Returns:
        tuple: :math:`(q, r)` where :math:`q` is the result of the division and :math:`r` is the rest
    """
    div_cc = div.complex_conjugate()  # √2 conjugate of the divider
    div_div_cc = div * div_cc  # Product of the divider by its complex conjugate

    # Convert the denominator into an integer
    denom_D = div_div_cc * div_div_cc.sqrt2_conjugate()  # Element of the ring D
    denom = denom_D.d  # Convert to an integer
    # Apply the same multiplication on the numerator
    numer = num * div_cc * div_div_cc.sqrt2_conjugate()

    n = numer
    a, b, c, d = n.a, n.b, n.c, n.d  # Extract the coefficients of numer
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


def euclidean_div_Zsqrt2(num: Zsqrt2, div: Zsqrt2) -> tuple[Zsqrt2, Zsqrt2]:
    r"""
    Perform the euclidean division of num in :math:`\mathbb{Z}[\sqrt{2}]`. This function returns :math:`q` and :math:`r` such that
    :math:`num = q \cdot div + r`.

    Args:
        num (Zsqrt2): Number to be divided
        div (Zsqrt2): Divider

    Returns:
        tuple: :math:`(q, r)` where :math:`q` is the result of the division and :math:`r` is the rest
    """
    num_ = num * div.sqrt2_conjugate()
    den_ = (div * div.sqrt2_conjugate()).a

    a_, b_ = num_.a, num_.b
    a, b = round(a_ / den_), round(b_ / den_)

    q = Zsqrt2(a, b)
    r = num - q * div

    return q, r


def are_sim_Zsqrt2(x: Zsqrt2, y: Zsqrt2) -> bool:
    r"""
    Determine if :math:`x \sim y`. Equivalently, :math:`x \sim y` if there exists a unit :math:`u` such that :math:`x = u \cdot y`.
    :math:`x`, :math:`y` and :math:`u` are elements of :math:`\mathbb{Z}[\sqrt{2}]`.

    Args:
        x (Zsqrt2): First number
        y (Zsqrt2): Second number

    Returns:
        bool: `True` if :math:`x \sim y`, `False` otherwise
    """
    # Test if y is a divider of x and y is a divider of x
    _, r1 = euclidean_div_Zsqrt2(x, y)
    _, r2 = euclidean_div_Zsqrt2(y, x)
    return (r1 == 0) and (r2 == 0)


def is_unit_Zsqrt2(x: Zsqrt2) -> bool:
    r"""
    Determine if :math:`x` is a unit in the ring :math:`\mathbb{Z}[\sqrt{2}]`.

    Args:
        x (Zsqrt2): The number to test

    Returns:
        bool: `True` if :math:`x` is a unit, `False` otherwise
    """
    integer = x * x.sqrt2_conjugate()
    return (integer == 1) or (integer == -1)


# ----------------------------- #
# Functions to solve the Diophantine equation
# ----------------------------- #
def is_square(n: int) -> bool:
    """
    Check if :math:`n` is a perfect square.

    Args:
        n (int): An integer

    Returns:
        bool: `True` if :math:`n` is a perfect square, `False` otherwise
    """
    if n < 0:
        return False

    # A square must have a modulo 16 of 0, 1, 4 or 9
    mod = n % 16
    if mod != 0 and mod != 1 and mod != 4 and mod != 9:
        return False

    # Check if n is a square
    return round(sqrt(n)) ** 2 == n


def solve_usquare_eq_a_mod_p(a: int, p: int) -> int:
    r"""
    Solve the diophantine equation :math:`u^2 = -a\ (\text{mod p})` where :math:`a`, :math:`p` and :math:`u` are integers. This function
    returns the first integer solution of the equation. :math:`p` is a prime. This problem is solved
    using the Tonelli-Shanks algorithm.

    Args:
        a (int): An integer
        p (int): A prime integer

    Returns:
        int: The first positive integer solution :math:`u` to the equation :math:`u^2 = -a\ (\text{mod p})`
    """
    if p == 1 and a == 1:  # Special case for p = 1
        return 1

    # Use the Tonelli-Shanks algorithm to find the square root of -a modulo p
    return tonelli_shanks_algo(-a, p)


def integer_fact(p: int) -> list[tuple[int, int]]:
    """
    Find the factorization of an integer :math:`p`. This function returns a list of tuples :math:`(p_i, m_i)` where
    :math:`p_i` is a prime factor of :math:`p` and :math:`m_i` is its power.

    Args:
        p (int): Number to factorize

    Returns:
        list of tuples: The prime factors of n and their powers. Each tuple is of the form
        :math:`(p_i, m_i)` where :math:`p_i` is a prime factor of :math:`p` and :math:`m_i` is its power.

    Raises:
        ValueError: If the number is less than 2.
        ValueError: If the number is not an integer.
    """
    if p < 2:
        raise ValueError("The number must be greater than 1.")

    if int(p) != p:
        raise ValueError(f"The number must be an integer. Got {p}.")

    n = p
    factors = []  # List of tuples (p_i, m_i)

    counter = 0
    while n % 2 == 0:
        counter += 1
        n = n // 2

    if counter > 0:
        factors.append((2, counter))

    # n must be odd at this point, so a skip of 2 (i = i + 2) can be used
    for i in range(3, int(sqrt(n)) + 1, 2):
        counter = 0

        # while i divides n, append i and divide n
        while n % i == 0:
            counter += 1
            n = n // i

        if counter > 0:
            factors.append((i, counter))

        if i > sqrt(n):
            break

    # If n != 1 at this point, n is a prime
    if n != 1:
        factors.append((n, 1))

    return factors


def xi_fact(xi: Zsqrt2) -> list[tuple[Zsqrt2, int]]:
    r"""
    Finds the factorization of :math:`\xi` (up to a prime) in the ring :math:`\mathbb{Z}[\sqrt{2}]` where :math:`\xi` is an
    element of :math:`\mathbb{Z}[\sqrt{2}]`. This function returns a list of tuples :math:`(\xi_i, m_i)`, where :math:`\xi_i` is
    a prime factor of :math:`\xi` in :math:`\mathbb{Z}[\sqrt{2}]` and :math:`m_i` is its power.

    Args:
        xi (Zsqrt2): An element of :math:`\mathbb{Z}[\sqrt{2}]`

    Returns:
        list of tuples: The prime factors of :math:`\xi` and their powers. Each tuple is of the form
        :math:`(\xi_i, m_i)` where :math:`\xi_i` is a prime factor of :math:`\xi` and :math:`m_i` is its power.
    """
    if xi == 0:  # 0 cannot be factorized
        return [
            (Zsqrt2(0, 0), 1),
        ]

    xi_fact_list = []
    p = (xi * xi.sqrt2_conjugate()).a

    if p == 1 or p == -1:  # ξ is a unit, so it cannot be factorized
        return [
            (xi, 1),
        ]

    if p < 0:  # If p is negative, we factorize -p > 0 instead
        p = -p
        xi_fact_list.append((Zsqrt2(-1, 0), 1))

    pi_list = integer_fact(p)

    for pi, mi in pi_list:
        # If pi = 2, ξ_i = sqrt(2)
        if pi == 2:
            xi_fact_list.append((Zsqrt2(0, 1), mi))

        # If pi % 8 == 1 or 7, we can factorize pi into ξ_i where pi = ξ_i * ξ_i⋅
        elif pi % 8 == 1 or pi % 8 == 7:
            xi_i = pi_fact_into_xi(pi)

            # Determine wether we need to add ξ_i or its conjugate to the factorization and how
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

        # If pi % 8 == 3 or 5, pi is its own factorization in Z[√2]
        # We need to append pi mi/2 times to the factorization of ξ since pi = ξ * ξ
        else:
            xi_fact_list.append((Zsqrt2(pi, 0), mi // 2))

    return xi_fact_list


def pi_fact_into_xi(pi: int) -> Union[Zsqrt2, None]:
    r"""
    Solve the equation :math:`p_i = \xi_i \cdot \xi_i^{\bullet} = a^2 - 2 \cdot b^2` where :math:`^{\bullet}` denotes
    the :math:`\sqrt{2}` conjugate. :math:`p_i` is a prime integer and :math:`\xi_i = a + b \sqrt{2}` is an element of
    :math:`\mathbb{Z}[\sqrt{2}]`. :math:`p_i` has a factorization only if :math:`p_i\ \%\ 8 = 1 \text{ or } 7` or if :math:`p_i = 2`.
    In any other case, the function returns `None`.

    Args:
        pi (int): A prime integer

    Returns:
        Zsqrt2 or None: A number :math:`\xi_i` for which :math:`p_i = \xi_i \cdot \xi_i^{\bullet}`, or `None` if :math:`p_i\ \%\ 8 \neq 1 \text{ or } 7`
    """
    if pi == 2:
        return Zsqrt2(0, 1)

    if not (pi % 8 == 1 or pi % 8 == 7):
        return None

    b = 1
    while not is_square(pi + 2 * b**2):
        b += 1

    return Zsqrt2(int(sqrt(pi + 2 * b**2)), b)


def xi_i_fact_into_ti(xi_i: Zsqrt2, check_prime: bool = False) -> Union[Zomega, None]:
    r"""
    Solve the equation :math:`\xi_i = t_i \cdot t_i^\dagger` where :math:`^\dagger` denotes the complex conjugate.
    :math:`\xi_i` is a prime element in :math:`\mathbb{Z}[\sqrt{2}]` and :math:`t_i` is an element of :math:`\mathbb{Z}[\omega]`. :math:`\xi_i` has a
    factorization only if :math:`p_i\ \%\ 8 = 1, 3 \text{ or } 5`, where :math:`p_i = \xi_i \cdot \xi_i^{\bullet}` or if :math:`p_i = 2`.

    Note: this function assumes :math:`\xi_i` is a prime element in :math:`\mathbb{Z}[\sqrt{2}]`. No check is performed to
    verify this assumption unless specified by the `check_prime` argument.

    Args:
        xi_i (Zsqrt2): A prime element in :math:`\mathbb{Z}[\sqrt{2}]`
        check_prime (bool): If set to `True`, the function will check if :math:`\xi_i` is a prime in :math:`\mathbb{Z}[\sqrt{2}]`

    Returns:
        Zomega or None: A number :math:`t_i` for which :math:`\xi_i = t_i \cdot t_i^\dagger`, or `None` if :math:`\xi_i\ \%\ 8 = 7`

    Raises:
        ValueError: If the input argument is not a prime in :math:`\mathbb{Z}[\sqrt{2}]` (only if `check_prime` is `True`,
            because this verification is computationally expensive)
    """
    # Verify if ξ_i is a prime in Z[√2]
    if check_prime:
        factors = xi_fact(xi_i)
        is_prime = True

        if len(factors) >= 3:  # The first factor might be a unit
            is_prime = False

        if len(factors) == 1:  # Check if the factor is not a unit
            if is_unit_Zsqrt2(factors[0][0]):
                is_prime = False

        if len(factors) == 2:  # Check if a least one factor is a unit
            if not (is_unit_Zsqrt2(factors[0][0]) or is_unit_Zsqrt2(factors[1][0])):
                is_prime = False

        for _, m in factors:
            if m > 1:
                is_prime = False
                break

        if not is_prime:
            raise ValueError("The input argument must be a prime in Z[sqrt(2)].")

    if xi_i == Zsqrt2(0, 1):  # xi_i = √2
        delta = Zomega(0, 0, 1, 1)  # δ = 1 + ω
        return delta

    if xi_i.b == 0:  # ξ_i is already a prime integer
        pi = xi_i.a
    else:
        pi = (xi_i * xi_i.sqrt2_conjugate()).a

    if pi % 4 == 1:
        u = solve_usquare_eq_a_mod_p(1, pi)
        xi_i_converted = Zomega.from_ring(xi_i)
        ti = gcd_Zomega(xi_i_converted, Zomega(0, 1, 0, u))  # Second term: u + i
        return ti

    if pi % 8 == 3:  # ξ_i = pi which is an integer in that case
        u = solve_usquare_eq_a_mod_p(2, pi)
        xi_i_converted = Zomega.from_ring(xi_i)
        ti = gcd_Zomega(xi_i_converted, Zomega(1, 0, 1, u))  # Second term: u + i √2
        return ti

    if pi % 8 == 7:
        return None


def solve_xi_sim_ttdag_in_z(xi: Zsqrt2) -> Union[Zomega, None]:
    r"""
    Solve the equation :math:`\xi \sim t \cdot t^\dagger` for :math:`t` where :math:`^\dagger` denotes the complex conjugate.
    :math:`\xi` is an element of :math:`\mathbb{Z}[\sqrt{2}]` and :math:`t` is an element of :math:`\mathbb{Z}[\omega]`. This function returns the
    first solution of the equation. If no solution exists, the function returns `None`.

    Args:
        xi (Zsqrt2): A number

    Returns:
        Zomega or None: A number :math:`t` for which :math:`\xi = t \cdot t^\dagger`, or `None` if no solution exists
    """
    xi_fact_list = xi_fact(xi)

    t = Zomega(0, 0, 0, 1)
    for xi_i, mi in xi_fact_list:
        if xi_i == -1:
            continue

        if mi % 2 == 0:  # For even exponents, ξ_i ** mi = ξ_i ** (mi // 2) * ξ_i ** (mi // 2)
            factor = xi_i ** (mi // 2)
            t *= Zomega.from_ring(factor)

        else:
            ti_i = xi_i_fact_into_ti(xi_i)
            if ti_i is None:
                return None

            t *= ti_i**mi

    return t


def solve_xi_eq_ttdag_in_d(xi: Dsqrt2) -> Union[Domega, None]:
    r"""
    Solve the equation :math:`\xi = t \cdot t^\dagger` for :math:`t` where :math:`^\dagger` denotes the complex conjugate. :math:`\xi`
    is an element of :math:`\mathbb{D}[\sqrt{2}]` and :math:`t` is an element of :math:`\mathbb{D}[\omega]`. This function returns the first
    solution of the equation. If no solution exists, it returns `None`.

    Args:
        xi (Dsqrt2): A number

    Returns:
        Domega or None: A number :math:`t` for which :math:`\xi = t \cdot t^\dagger`, or `None` if no solution exists
    """
    # The equation only has a solution if ξ is doubly positive, i.e. ξ >= 0 and ξ• >= 0.
    if float(xi) < 0 or float(xi.sqrt2_conjugate()) < 0:
        return None

    # If ξ = 0, the solution is 0
    if xi == 0:
        return Domega((0, 0), (0, 0), (0, 0), (0, 0))

    l = (xi * xi.sqrt2_conjugate()).a.denom  # Greatest denominator power of 2
    xi_prime_temp = Dsqrt2(D(0, 0), D(1, 0)) ** l * xi  # ξ_prime is in Z[√2]
    xi_prime = Zsqrt2.from_ring(xi_prime_temp)  # Convert ξ_prime to Z[√2]

    s = solve_xi_sim_ttdag_in_z(xi_prime)  # Solve the equation ξ' ~ s * s†
    if s is None:  # If there is no solution to the equation ξ' ~ s * s†
        return None

    delta = Zomega(0, 0, 1, 1)  # δ = 1 + ω
    # δ**-1 = δ * λ**-1 * ω**-1 / √2
    delta_inv = (
        Domega.from_ring(delta)
        * Domega((-1, 0), (0, 0), (1, 0), (-1, 0))
        * Domega((0, 0), (-1, 1), (0, 0), (1, 1))
    )
    delta_inv_l = delta_inv**l  # δ_l = δ ** l

    t = delta_inv_l * Domega.from_ring(s)  # t = δ**-l * s
    tt = Dsqrt2.from_ring(t * t.complex_conjugate())  # tt = t * t†

    # Find u such that ξ = u * t * t†
    denom = (tt * tt.sqrt2_conjugate()).a  # Element of ring D
    u_temp = xi * tt.sqrt2_conjugate() * int(2**denom.denom)
    u = Zsqrt2(u_temp.a.num // denom.num, u_temp.b.num // denom.num)

    # u is of the form u = λ**2n => n = ln(u) / 2 ln(λ)
    n = round(log(float(u)) / (2 * log(float(Zsqrt2(1, 1)))))

    # v**2 = u => v = λ**n
    if n > 0:
        v = Domega((-1, 0), (0, 0), (1, 0), (1, 0)) ** n  # λ**n
    elif n == 0:
        v = Domega((0, 0), (0, 0), (0, 0), (1, 0))  # 1
    else:
        v = Domega((-1, 0), (0, 0), (1, 0), (-1, 0)) ** -n  # (λ**-1)**n

    return t * v
