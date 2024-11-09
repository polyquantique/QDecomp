import sys
sys.path.append("../CliffordPlusT")

import numpy as np
from sympy import symbols, diophantine

from Zomega import Zomega
from grid_algorithm_1D.Zsqrt2 import Zsqrt2


def integer_fact(p):
    """
    Find the factorization of an integer p. This function returns a list of tuples (p_i, m_i) where
    p_i is a prime factor of p and m_i is its power.
    
    :param n: An integer
    :return: A list of tuples (p_i, m_i) containing the prime factors of n and their powers
    """
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

        # while i divides n, print i and divide n
        while n % i== 0:
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

def pi_fact(pi, mi):
    """
    Finds the factorization of pi**mi in the ring of integers of Z[omega] where pi is an integer
    prime, mi is an integer, and omega = exp(i pi /4). This function return a number in Z[omega] to
    the power mi for which the relation pi**mi = (ti**mi) * (ti**mi)† where † denotes the complex
    conjugate.

    :param pi: A prime integer
    :param mi: An integer
    :return: An element of Z[omega] for which pi**mi = (ti**mi) * (ti**mi)†
    """
    pass

def xi_fact(xi):
    """
    Finds the factorization of xi in the ring Z[sqrt(2)] where xi is an element of Z[sqrt(2)]. This
    function returns a list of tuples (xi_i, mi), where xi_i is a prime factor of xi in Z[sqrt(2)]
    and mi is its power.

    :param xi: An element of Z[sqrt(2)]
    :return: A list of tuples (xi_i, mi) containing the prime factors of xi and their powers
    """
    p = xi * xi.conjugate()
    pi_list = integer_fact(p)
    xi_fact_list = []

    for pi, mi in pi_list:
        if pi % 8 == 1 or pi % 8 == 7:
            xi_i = pi_fact_into_xi(pi)
        else:
            xi_i = pi_fact(pi, mi)

        xi_list = xi_fact(xi_i)

    return xi_list

def pi_fact__pi_eq_2(pi, mi):
    """
    
    """
    pass

def pi_fact__mi_even(pi, mi):
    """
    
    """
    pass


def pi_fact__pi_mod_4_eq_1(pi, mi):
    """
    
    """
    pass


def pi_fact__pi_mod_8_eq_3(pi, mi):
    """
    
    """
    pass


def pi_fact_into_xi(pi):
    """
    Solve the equation pi = xi_i * xi_i⋅ = a**2 - 2 * b**2 where ⋅ denotes the sqrt(2) conjugate.
    pi is a prime integer and xi_i = a + b * sqrt(2) is an element of Z[sqrt(2)]. pi has a 
    factorization only if pi % 8 = 1 or 7. In any other case, the function returns None.

    :param pi: A prime integer
    :return: An element of Z[sqrt(2)] for which pi = xi_i * xi_i⋅, or None if pi % 8 != 1 or 7
    """
    if not (pi % 8 == 1 or pi % 8 == 7):
        return None

    a, b, t = symbols("a b t", integer=True)
    equation = a**2 - 2 * b**2 - pi
    solutions = diophantine(equation, t)

    a0, b0 = solutions.pop()  # Extract the first solution

    sub = {t: 0}
    xi = Zsqrt2(int(a0.subs(sub)), int(b0.subs(sub)))

    return xi


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
    div_sc = div.sqrt2_conjugate()  # Complex conjugate of the divider
    div_cc = div.complex_conjugate()  # Sqrt(2) conjugate of the divider

    div_div_cc = div * div_cc  # Product of the divider by its complex conjugate

    denom = (div_div_cc * div_div_cc.sqrt2_conjugate()).d.num  # Convert the denominator into an integer
    numer = num * div_cc * div_div_cc.sqrt2_conjugate()  # Apply the same multiplication on the numerator

    n = numer
    a, b, c, d = n.a.num, n.b.num, n.c.num, n.d.num  # Extract the coefficients of numer
    # Divide the coefficients by the integer denominator and round them
    a_, b_, c_, d_ = round(a/denom), round(b/denom), round(c/denom), round(d/denom)

    q = Zomega(a_, b_, c_, d_)  # Construction of the divider with the new coefficients
    r = num - q * div  # Calculation of the rest of the division

    return q, r
