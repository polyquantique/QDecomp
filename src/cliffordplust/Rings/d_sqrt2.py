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


from __future__ import annotations

import math
from decimal import Decimal, getcontext
from typing import Any, Optional

import numpy as np
from cliffordplust.rings import D

__all__ = ["Dsqrt2"]

SQRT2 = math.sqrt(2)


class Dsqrt2:
    """Class to do symbolic computation with elements in the ring of quadratic dyadic fractions \u2145[\u221a2].

    The ring element has the form a + b\u221a2, where a and b are dyadic fractions of the form m / 2^n,
    where m is an integer and n is a positive integer.
    The coefficients will be automatically reduced when the class is initialized.

    Attributes:
        a (D): Rational coefficient of the ring element.
        b (D): \u221a2 coefficient of the ring element.
    """

    def __init__(self, a: tuple[int, int] | D, b: tuple[int, int] | D) -> None:
        """Initialize the Dsqrt2 class.

        Args:
            a (tuple[int, int] | D): Rational coefficient of the ring element.
            b (tuple[int, int] | D): \u221a2 coefficient of the ring element.

        Raises:
            TypeError: If the class arguments are not 2-tuples of integers (num, denom) or D objects.
            ValueError: If the denominator exponent is negative.
        """
        for input in (a, b):
            if isinstance(input, tuple):
                if len(input) != 2 or any(
                    [not isinstance(value, (int, np.integer)) for value in input]
                ):
                    raise TypeError(
                        f"Tuples must take two integer values (num, denom) but received {input}."
                    )
                elif input[1] < 0:
                    raise ValueError(f"Denominator value must be positive but got {input[1]} < 0.")
            elif not isinstance(input, D):
                raise TypeError(
                    f"Class arguments must be of type tuple[int, int] or D objects, but received {type(input).__name__}."
                )
        self._a: D = a if isinstance(a, D) else D(a[0], a[1])
        self._b: D = b if isinstance(b, D) else D(b[0], b[1])

    @property
    def a(self) -> D:
        """Rational coefficient of the ring element."""
        return self._a

    @property
    def b(self) -> D:
        """\u221A2 coefficient of the ring element."""
        return self._b

    def conjugate(self) -> Dsqrt2:
        """Define the \u221A2-conjugation operation.

        Returns:
            Dsqrt2: \u221A2-conjugate of the ring element.
        """
        return Dsqrt2(self.a, -self.b)

    def __float__(self) -> float:
        """Define the float representation of the ring element."""
        bsqrt = float(self.b) * SQRT2
        if math.isclose(float(self.a), -bsqrt, rel_tol=1e-5):
            getcontext().prec = 50
            return float(
                self.a.num / Decimal(2) ** self.a.denom
                + self.b.num * Decimal(2).sqrt() / 2**self.b.denom
            )
        return float(self.a) + float(bsqrt)

    def __getitem__(self, i: int) -> D:
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
        """Define the equality of Dsqrt2 class."""
        if isinstance(nb, Dsqrt2):
            return self.a == nb.a and self.b == nb.b
        elif isinstance(nb, (D, int, np.integer)):
            return self.a == nb and self.b == 0
        try:
            return float(self) == float(nb)
        except Exception:
            return False

    def __neg__(self) -> Dsqrt2:
        """Define the negation of a ring element."""
        return Dsqrt2(-self.a, -self.b)

    def __add__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the summation operation for the Dsqrt2 class."""
        if isinstance(nb, Dsqrt2):
            return Dsqrt2(self.a + nb.a, self.b + nb.b)
        elif isinstance(nb, (D, int, np.integer)):
            return Dsqrt2(self.a + nb, self.b)
        raise TypeError(
            f"Summation operation is not defined between Dsqrt2 and {type(nb).__name__}."
        )

    def __radd__(self, nb: int | D) -> Dsqrt2:
        """Define the right summation of integers and D objects with the Dsqrt2 class."""
        return self.__add__(nb)

    def __iadd__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the subtraction operation for the Dsqrt2 class."""
        if isinstance(nb, Dsqrt2):
            return Dsqrt2(self.a - nb.a, self.b - nb.b)
        elif isinstance(nb, (D, int, np.integer)):
            return Dsqrt2(self.a - nb, self.b)
        raise TypeError(
            f"Subtraction operation is not defined between Dsqrt2 and {type(nb).__name__}."
        )

    def __rsub__(self, nb: int) -> Dsqrt2:
        """Define the right subtraction of integers and D objects with the Dsqrt2 class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define in-place subtraction operation for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the multiplication operation for the Dsqrt2 class."""
        if isinstance(nb, Dsqrt2):
            return Dsqrt2(self.a * nb.a + 2 * self.b * nb.b, self.a * nb.b + self.b * nb.a)
        elif isinstance(nb, (D, int, np.integer)):
            return Dsqrt2(self.a * nb, self.b * nb)
        raise TypeError(
            f"Multiplication operation is not defined between Dsqrt2 and {type(nb).__name__}."
        )

    def __rmul__(self, nb: int | D) -> Dsqrt2:
        """Define the right multiplication of integers and D objects with the Dsqrt2 class."""
        return self.__mul__(nb)

    def __imul__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define in-place multiplication operation for the class."""
        return self.__mul__(nb)

    def __pow__(self, n: int) -> Dsqrt2:
        """Define the power operation for the Dsqrt2 class. Computed using the binomial theorem.

        Exponent must be a positive integer.
        """
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Expected power to be of type int, but got {type(n).__name__}.")
        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n} < 0.")

        a: D = D(0, 0)
        b: D = D(0, 0)
        for k in range(n + 1):
            if k % 2 == 0:
                a += math.comb(n, k) * self.a ** (n - k) * self.b**k * 2 ** (k // 2)
            else:
                b += math.comb(n, k) * self.a ** (n - k) * self.b**k * 2 ** (k // 2)
        return Dsqrt2(a, b)

    def __ipow__(self, nb: int) -> Dsqrt2:
        """Define in-place power."""
        return self.__pow__(nb)

    def __round__(self, precision: Optional[int] = None) -> int | float:
        """Define the round operation.

        Args:
            precision (float, optional): Decimal precision of the rounding. Default = 0.

        Returns:
            (int | float): Ring element rounded to the given precision.
        """
        return round(float(self), precision)

    def __floor__(self) -> int:
        """Define the floor operation."""
        return math.floor(float(self))

    def __ceil__(self) -> int:
        """Define the ceil operation."""
        return math.ceil(float(self))
