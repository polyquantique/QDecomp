import pytest
import numpy as np

from diophantine_equation import *


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

def test_euclidean_div():
    """
    Test the euclidean division function.
    """
    num = Zomega(*np.random.randint(-10, 10, 4))
    div = Zomega(*np.random.randint(-10, 10, 4))
    
    q, r = euclidean_div_Zomega(num, div)
    
    assert (q * div + r) == num
