from __future__ import annotations

from CliffordPlusT.Domega import D
from grid_algorithm_1D.Zsqrt2 import Zsqrt2

import numpy as np
import math
from typing import Iterator, Any


class Dsqrt2:
    """Class to do symbolic computation with elements of the ring D[√2].

    The ring element has the form a + b√2. 
    The coefficients a, b are dyadic fractions of the form m / 2^n, where m is an integer and n is a positive integer.
    The coefficient will be automatically reduced when the class is initialized.

    The ring element can also be expressed as alpha + i*beta, where i = √-1, and alpha and beta are numbers in te ring D[√2].
    alpha = d + (c-a)/2 √2 and beta = b + (c+a)/2 √2.

    Attributes:
        a (D): Rationnal coefficient of the ring element.
        b (D): √2 coefficient of the ring element.
        is_Zsqrt2 (bool): True if the ring element is also in the ring Z[√2].
    """
    def __init__(self, a: tuple[int, int] | D, b: tuple[int, int] | D) -> None:
        """Initialize the Dsqrt2 class.
        
        Args:
            a (tuple[int, int] | D): Rationnal coefficient of the ring element.
            b (tuple[int, int] | D): √2 coefficient of the ring element.

        Raises:
            TypeError: If `a` or `b` are not instances of `D` or tuples of two integers.
        """

        # Check if `a` is of the correct type
        if not (isinstance(a, D) or (isinstance(a, tuple) and len(a) == 2 and all(isinstance(i, int) for i in a))):
            raise TypeError("Argument `a` must be an instance of `D` or a tuple of two integers (int, int).")
        
        # Check if `b` is of the correct type
        if not (isinstance(b, D) or (isinstance(b, tuple) and len(b) == 2 and all(isinstance(i, int) for i in b))):
            raise TypeError("Argument `b` must be an instance of `D` or a tuple of two integers (int, int).")
        
        self._a: D = a if isinstance(a, D) else D(a[0], a[1])
        self._b: D = b if isinstance(b, D) else D(b[0], b[1])

    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b
    
    @property
    def is_Zsqrt2(self) -> bool:
        return all([coeff.is_integer for coeff in self])
    
    def __repr__(self) -> str:
        return f"{self.a} + {self.b}√2"
    
    def __getitem__(self, i: int | slice) -> D | list[D]:
        return [self.a, self.b][i]
    
    def __iter__(self) -> Iterator[D]:
        return iter([self.a, self.b])
    
    def conjugate(self) -> Dsqrt2:
        return Dsqrt2(self.a, -self.b)
    
    def __neg__(self) -> Dsqrt2:
        return Dsqrt2(-self.a, -self.b)
    
    def __add__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        if isinstance(nb, (int, np.int32, np.int64, D)):
            return Dsqrt2(self.a + nb, self.b)
        elif isinstance(nb, Dsqrt2):
            return Dsqrt2(self.a + nb.a, self.b + nb.b)
    
    def __radd__(self, nb: int | D) -> Dsqrt2:
        return self.__add__(nb)
    
    def __sub__(self, nb: int | D | int) -> Dsqrt2:
        return self.__add__(-nb)
    
    def __rsub__(self, nb: int | D) -> Dsqrt2:
        return -self + nb
    
    def __mul__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        if isinstance(nb, (int, np.int32, np.int64)):
            return Dsqrt2(self.a * D(nb, 0), self.b * D(nb, 0))
        elif isinstance(nb, D):
            return Dsqrt2(self.a * nb, self.b * nb)
        elif isinstance(nb, Dsqrt2):
            a: D = self.a * nb.a + 2 * self.b * nb.b
            b: D = self.a * nb.b + self.b * nb.a
            return Dsqrt2(a, b)
        
    def __rmul__(self, nb: int | D | Dsqrt2) -> Dsqrt2:
        return self.__mul__(nb)
