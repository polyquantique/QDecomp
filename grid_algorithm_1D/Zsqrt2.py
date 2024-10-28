from __future__ import annotations

import math
from decimal import Decimal, getcontext
from typing import Any, Optional

import numpy as np


class Zsqrt2:
    """A simple class to do symbolic computation with elements of the ring Z[sqrt(2)].

    The ring element has the form a + b√2.

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
        self.a: int = a
        self.b: int = b
        for input in (a, b):
            if not isinstance(input, (int, np.int32, np.int64)):
                raise TypeError(
                    f"Expected inputs to be of type int, but got {type(input).__name__}."
                )

    def conjugate(self) -> Zsqrt2:
        """Define the √2-conjugation operation."""
        return Zsqrt2(self.a, -self.b)

    def __float__(self, precision: int = 50) -> float:
        """Define the float representation."""
        getcontext().prec = precision
        result = Decimal(int(self.a)) + Decimal(int(self.b)) * Decimal(2).sqrt()
        return float(result)

    def __getitem__(self, i: int) -> int:
        """Access the values of a and b with their index."""
        return (self.a, self.b)[i]

    def __repr__(self) -> str:
        """Return string representation of the ring element."""
        if self.b >= 0:
            return f"{self.a} + {self.b}√2"
        else:
            return f"{self.a} - {-self.b}√2"

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of Zsqrt class."""
        return float(self) == nb

    def __neg__(self) -> Zsqrt2:
        """Define the element negation."""
        return Zsqrt2(-self.a, -self.b)

    def __add__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Add two Z[sqrt(2)] elements.

        Args:
            nb (Zsqrt2 | int): Number to add to the ring element.

        Returns:
            Zsqrt2: Sum result in Z[sqrt(2)].

        Raises:
            TypeError: If nb is not of type Zsqrt2 or int.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a + nb.a, self.b + nb.b)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return Zsqrt2(self.a + nb, self.b)
        else:
            raise TypeError(
                f"'{type(self).__name__}' + '{type(nb).__name__}' operation is not defined"
            )

    def __radd__(self, nb: int) -> Zsqrt2:
        """Define the right summation of int with Z[sqrt2] element."""
        return self.__add__(nb)

    def __iadd__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define in-place summation."""
        return self + nb

    def __sub__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Subtracts two Z[sqrt(2)] elements.

        Args:
            nb (Zsqrt2 | int): Number to subtract to the ring element.

        Returns:
            Zsqrt2: Difference result in Z[sqrt(2)].

        Raises:
            TypeError: If nb is not of type Zsqrt2 or int.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a - nb.a, self.b - nb.b)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return Zsqrt2(self.a - nb, self.b)
        else:
            raise TypeError(
                f"'{type(self).__name__}' - '{type(nb).__name__}' operation is not defined"
            )

    def __rsub__(self, nb: int) -> Zsqrt2:
        """Define the right subtraction of int with Z[sqrt2] element."""
        return -self + nb

    def __isub__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define in-place subtraction."""
        return self - nb

    def __mul__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Product of two Z[sqrt(2)] elements.

        Args:
            nb (Zsqrt2 | int): Number to multiply to the ring element.

        Returns:
            Zsqrt2: Product result in Z[sqrt(2)].

        Raises:
            TypeError: If nb is not of type Zsqrt2 or int.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a * nb.a + 2 * self.b * nb.b, self.a * nb.b + self.b * nb.a)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return Zsqrt2(self.a * nb, self.b * nb)
        else:
            raise TypeError(
                f"'{type(self).__name__}' * '{type(nb).__name__}' operation is not defined"
            )

    def __rmul__(self, nb: int) -> Zsqrt2:
        """Define the right product of int with Z[sqrt2] element."""
        return self * nb

    def __imul__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define in-place multiplication."""
        return self * nb

    def __pow__(self, n: int) -> Zsqrt2:
        """Power of a Z[sqrt(2)] element.

        Args:
            n (int): Power of the ring element.

        Returns:
            Zsqrt2: Result in Z[sqrt(2)].

        Raises:
            TypeError: If n is not of type int.
            ValueError: If n < 0.
        """
        if not isinstance(n, (int, np.int32, np.int64)):
            raise TypeError(f"Expected power to be of type int, but got {type(n).__name__}.")
        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n}.")

        pow_out = Zsqrt2(1, 0)
        for i in range(n):
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
