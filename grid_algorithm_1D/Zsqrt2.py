from __future__ import annotations

import math
from decimal import Decimal, getcontext
from typing import Any, Optional

import numpy as np


class Zsqrt2:
    """A simple class to do symbolic computation with elements of the ring Z[sqrt(2)].

    The ring element has the form a + b√2, where a and b are integers.

    Attributes:
        a (int): Integer coefficient of the ring element.
        b (int): √2 coefficient of the ring element.
    """

    def __init__(self, a: int, b: int) -> None:
        """Initialize the ring element.

        Args:
            a (int): Integer coefficient of the ring element.
            b (int): √2 coefficient of the ring element.

        Raises:
            TypeError: If a or b are not integers.
        """
        for input in (a, b):
            if not isinstance(input, (int, np.int32, np.int64)):
                raise TypeError(
                    f"Expected class inputs to be of type int, but got {type(input).__name__}."
                )
        self._a: int = a
        self._b: int = b

    @property
    def a(self) -> int:
        return self._a

    @property
    def b(self) -> int:
        return self._b

    def conjugate(self) -> Zsqrt2:
        """Define the √2-conjugation operation."""
        return Zsqrt2(self.a, -self.b)

    def __getitem__(self, i: int) -> int:
        """Access the values of a and b with their index."""
        return (self.a, self.b)[i]

    def __repr__(self) -> str:
        """Define the string representation of the ring element."""
        if self.a == 0 and self.b == 0:
            return str(0)
        elif self.b >= 0:
            return f"{self.a} + {self.b}√2"
        else:
            return f"{self.a} - {-self.b}√2"

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of Zsqrt class."""
        return float(self) == nb

    def __neg__(self) -> Zsqrt2:
        """Define the negation of a ring element."""
        return Zsqrt2(-self.a, -self.b)

    def __float__(self, precision: int = 50) -> float:
        """Define the float representation of the ring element."""
        getcontext().prec = precision
        result = Decimal(int(self.a)) + Decimal(int(self.b)) * Decimal(2).sqrt()
        return float(result)

    def __add__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define the summation operation for the Zsqrt2 class.

        Allow summation with integers or Zsqrt2 objects.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a + nb.a, self.b + nb.b)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return Zsqrt2(self.a + nb, self.b)
        else:
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
        elif isinstance(nb, (int, np.int32, np.int64)):
            return Zsqrt2(self.a - nb, self.b)
        else:
            raise TypeError(f"Subtraction operation is not defined with {type(nb).__name__}.")

    def __rsub__(self, nb: int) -> Zsqrt2:
        """Define the right subtraction of int with the Zsqrt2 class."""
        return -self + nb

    def __isub__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define in-place subtraction operation for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define the multiplication operation for the Zsqrt2 class.

        Allow multiplication with integers and Zsqrt2 objects.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a * nb.a + 2 * self.b * nb.b, self.a * nb.b + self.b * nb.a)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return Zsqrt2(self.a * nb, self.b * nb)
        else:
            raise TypeError(f"Multiplication operation is not defined with {type(nb).__name__}")

    def __rmul__(self, nb: int) -> Zsqrt2:
        """Define the right multiplication of int with the Zsqrt2 class."""
        return self * nb

    def __imul__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define in-place multiplication operation for the class."""
        return self * nb

    def __pow__(self, n: int) -> Zsqrt2:
        """Define the power operation for the Zsqrt2 class.

        Exponent must be a positive integer.
        """
        if not isinstance(n, (int, np.int32, np.int64)):
            raise TypeError(f"Expected power to be of type int, but got {type(n).__name__}.")
        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n}.")

        pow_out = Zsqrt2(1, 0)
        for _ in range(n):
            pow_out = self * pow_out
        return pow_out

    def __ipow__(self, nb: int) -> Zsqrt2:
        """Define in-place power."""
        return self**nb

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


lamb: Zsqrt2 = Zsqrt2(1, 1)
inv_lamb: Zsqrt2 = -lamb.conjugate()

if __name__ == "__main__":
    x = Zsqrt2(1, 1)
    y = Zsqrt2(2, 5)
    # print(x**-1)
    n1 = Zsqrt2(np.int64(-6), np.int64(4))
    n2 = 9
    print(2 + 3 * math.sqrt(2))
    print(float(Zsqrt2(2, 3)))
