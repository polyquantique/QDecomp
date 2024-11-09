import sys
sys.path.append("../CliffordPlusT")

from Zomega import Zomega


def pi_fact(pi, mi):
    """
    Finds the factorization of pi**mi in the ring of integers of Z[omega] where pi is an integer
    prime, mi is an integer and omega = exp(i pi /4). This function return a number in Z[omega] to
    the power mi for which the relation pi**mi = (ti**mi) * (ti**mi)† where † denotes the complex
    conjugate.

    :param pi: A prime integer
    :param mi: An integer
    :return: An element of Z[omega] for which pi**mi = (ti**mi) * (ti**mi)†
    """
    pass

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
