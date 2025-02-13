import numpy as np
import pytest

from Domega import Domega
from Zomega import Zomega


def test_Zomega():
    """
    Test that the class Zomega is equivalent to the Domega class, but with unit denominators for the
    coefficients.
    """
    for _ in range(15):
        a, b, c, d = np.random.randint(-100, 100, 4)

        nbZ = Zomega(a, b, c, d)
        nbD = Domega((a, 0), (b, 0), (c, 0), (d, 0))

        assert nbZ == nbD

        methodsZ = set(dir(nbZ))
        methodsD = set(dir(nbD))

        assert methodsZ == methodsD
