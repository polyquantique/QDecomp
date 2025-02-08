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

"""
This module contains the definition of the Zsqrt2 class, which allows for symbolic calculation with elements of 
the ring of quadratic integers with radicand 2 \u2124[\u221A2]. The ring elements have the form a + b\u221A2, 
where a and b are integers. This class is useful among others to solve 1 dimensional grid problems, where 
solutions are found inside this ring. For more information see 
Neil J. Ross and Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations, 
https://arxiv.org/pdf/1403.2975.
"""

from __future__ import annotations

import math
import numbers as num
from decimal import Decimal, getcontext
from typing import Any, Optional


class Zsqrt2:
    """A simple class to do symbolic computation with elements of the ring \u2124[\u221A2].

    The ring element has the form a + b\u221A2, where a and b are integers.

    Attributes:
        a (int): Integer coefficient of the ring element.
        b (int): \u221A2 coefficient of the ring element.
    """

    def __init__(self, a: int, b: int) -> None:
        """Initialize the ring element.

        Args:
            a (int): Integer coefficient of the ring element.
            b (int): \u221A2 coefficient of the ring element.

        Raises:
            TypeError: If a or b are not integers.
        """
        for input in (a, b):
            if not isinstance(input, num.Integral):
                raise TypeError(
                    f"Expected inputs to be of type int, but got {type(input).__name__}."
                )
        self._a: int = a
        self._b: int = b

    @property
    def a(self) -> int:
        """Integer coefficient of the ring element."""
        return self._a

    @property
    def b(self) -> int:
        """\u221A2 coefficient of the ring element."""
        return self._b

    def conjugate(self) -> Zsqrt2:
        """Define the \u221A2-conjugation operation.

        Returns:
            Zsqrt2: \u221A2-conjugate of the ring element.
        """
        return Zsqrt2(self.a, -self.b)

    def __float__(self) -> float:
        """Define the float representation of the ring element."""
        bsqrt = self.b * math.sqrt(2)
        if math.isclose(self.a, -bsqrt, rel_tol=1e-4):
            getcontext().prec = 50
            return float(self.a + self.b * Decimal(2).sqrt())
        return self.a + bsqrt

    def __getitem__(self, i: int) -> int:
        """Access the values of a and b with their index."""
        return (self.a, self.b)[i]

    def __repr__(self) -> str:
        """Define the string representation of the ring element."""
        repr: str = ""
        repr += str(self.a)
        if self.b < 0:
            repr += str(self.b) + "\u221A2"
        else:
            repr += "+" + str(self.b) + "\u221A2"
        return repr

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of Zsqrt2 class."""
        if isinstance(nb, Zsqrt2):
            return self.a == nb.a and self.b == nb.b
        elif isinstance(nb, num.Integral):
            return self.a == nb and self.b == 0
        return False

    def __neg__(self) -> Zsqrt2:
        """Define the negation of a ring element."""
        return Zsqrt2(-self.a, -self.b)

    def __add__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define the summation operation for the Zsqrt2 class.

        Allow summation with integers or Zsqrt2 objects.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a + nb.a, self.b + nb.b)
        elif isinstance(nb, num.Integral):
            return Zsqrt2(self.a + nb, self.b)
        raise TypeError(f"Summation operation is not defined with {type(nb).__name__}.")

    def __radd__(self, nb: int) -> Zsqrt2:
        """Define the right summation of int with the Zsqrt2 class."""
        return self.__add__(nb)

    def __iadd__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define the subtraction operation for the Zsqrt2 class.

        Allow subtraction with integers and Zsqrt2 objects.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a - nb.a, self.b - nb.b)
        elif isinstance(nb, num.Integral):
            return Zsqrt2(self.a - nb, self.b)
        raise TypeError(f"Subtraction operation is not defined with {type(nb).__name__}.")

    def __rsub__(self, nb: int) -> Zsqrt2:
        """Define the right subtraction of int with the Zsqrt2 class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define in-place subtraction operation for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define the multiplication operation for the Zsqrt2 class.

        Allow multiplication with integers and Zsqrt2 objects.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a * nb.a + 2 * self.b * nb.b, self.a * nb.b + self.b * nb.a)
        elif isinstance(nb, num.Integral):
            return Zsqrt2(self.a * nb, self.b * nb)
        raise TypeError(f"Multiplication operation is not defined with {type(nb).__name__}")

    def __rmul__(self, nb: int) -> Zsqrt2:
        """Define the right multiplication of int with the Zsqrt2 class."""
        return self.__mul__(nb)

    def __imul__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define in-place multiplication operation for the class."""
        return self.__mul__(nb)

    def __pow__(self, n: int) -> Zsqrt2:
        """Define the power operation for the Zsqrt2 class. Computed using the binomial theorem.

        Exponent must be a positive integer.
        """
        if not isinstance(n, num.Integral):
            raise TypeError(f"Expected power to be of type int, but got {type(n).__name__}.")
        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n} < 0.")

        a: int = 0
        b: int = 0
        for k in range(n + 1):
            if k % 2 == 0:
                a += math.comb(n, k) * self.a ** (n - k) * self.b**k * 2 ** (k // 2)
            else:
                b += math.comb(n, k) * self.a ** (n - k) * self.b**k * 2 ** (k // 2)
        return Zsqrt2(a, b)

    def __ipow__(self, nb: int) -> Zsqrt2:
        """Define in-place power."""
        return self.__pow__(nb)

    def __round__(self, precision: Optional[int] = None) -> int | float:
        """Define the round operation.

        Args:
            precision (float, optional): Decimal precision of the rounding. Default = 0.

        Returns:
            (int | float): Ring element rounded to the given precision.
        """
        if precision is None:
            return round(float(self))
        else:
            return round(float(self), precision)

    def __floor__(self) -> int:
        """Define the floor operation."""
        return math.floor(float(self))

    def __ceil__(self) -> int:
        """Define the ceil operation."""
        return math.ceil(float(self))


# LAMBDA = 1 + \u221A2 is used to scale 1D grid problems.
LAMBDA: Zsqrt2 = Zsqrt2(1, 1)

# INVERSE_LAMBDA = -1 + \u221A2 is the inverse of LAMBDA. It is used to scale 1D grid problem.
INVERSE_LAMBDA: Zsqrt2 = -LAMBDA.conjugate()
