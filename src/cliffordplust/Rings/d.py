from __future__ import annotations

from typing import Any

import numpy as np


class D:
    """Class to do symbolic computation with elements of the ring of dyadic fractions \u2145.

    The ring element has the form a/(2^k), where a is an integer and k is a positive integer.

    Attributes:
        num (int): Numerator of the ring element.
        denom (int): Power of 2 in the denominator of the ring element.
        is_integer (bool): True if the ring element is an integer.
    """

    def __init__(self, num: int, denom: int) -> None:
        """Initialize the ring element.

        Args:
            num (int): Numerator of the ring element
            denom (int): Power of 2 in the denominator of the ring element.

        Raises:
            TypeError: If the numerator or the denominator exponent are not integers.
            ValueError: If the denominator exponent is not positive.
        """
        for arg in (num, denom):
            if not isinstance(arg, (int, np.integer)):
                raise TypeError(
                    f"Class arguments must be of type int but received {type(arg).__name__}."
                )
        if denom < 0:
            raise ValueError(f"Denominator exponent must be positive but got {denom}.")
        self._num: int = num
        self._denom: int = denom

        # Reduce the fraction
        if not self.is_integer:
            self._reduce()

    @property
    def num(self) -> int:
        """Numerator of the dyadic fraction."""
        return self._num

    @property
    def denom(self) -> int:
        """Denominator exponent of the dyadic fraction."""
        return self._denom

    @property
    def is_integer(self) -> bool:
        """Return True if the number is an integer."""
        return self.denom == 0

    def _reduce(self) -> None:
        """Reduce the fraction if the numerator is even."""
        if self.num == 0:
            self._denom = 0
            return
        while (
            self.num & 1
        ) == 0 and self.denom > 0:  # Do while the numerator and denominator are factors of two.
            self._num >>= 1  # Divide the numerator by two.
            self._denom -= 1  # Reduce the denominator exponent by one.

    def __neg__(self) -> D:
        """Define the negation of the D class."""
        return D(-self.num, self.denom)

    def __abs__(self) -> D:
        """Define the absolute value of the D class."""
        return D(abs(self.num), self.denom)

    def __repr__(self) -> str:
        """Define the string representation of the D class."""
        return f"{self.num}/2^{self.denom}"

    def __float__(self) -> float:
        """Define the float value of the D class."""
        return self.num / 2**self.denom

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of the D class."""
        if isinstance(nb, D):
            return self.num == nb.num and self.denom == nb.denom
        elif isinstance(nb, (int, np.integer)):
            return self.denom == 0 and self.num == nb
        else:
            try:
                return float(nb) == float(self)
            except Exception:
                return False

    def __lt__(self, nb: Any) -> bool:
        """Define the < operation of the D class."""
        return float(self) < nb

    def __gt__(self, nb: Any) -> bool:
        """Define the > operation of the D class."""
        return float(self) > nb

    def __le__(self, nb: Any) -> bool:
        """Define the <= operation of the D class."""
        return self.__lt__(nb) or self.__eq__(nb)

    def __ge__(self, nb: Any) -> bool:
        """Define the >= operation of the D class."""
        return self.__gt__(nb) or self.__eq__(nb)

    def __add__(self, nb: Any) -> D:
        """Define the summation operation for the D class."""
        if isinstance(nb, D):
            if self.denom >= nb.denom:
                num: int = self.num + nb.num * 2 ** (self.denom - nb.denom)
                return D(num, self.denom)
            num = nb.num + self.num * 2 ** (nb.denom - self.denom)
            return D(num, nb.denom)
        elif isinstance(nb, (int, np.integer)):
            return D(self.num + nb * 2**self.denom, self.denom)
        raise TypeError(f"Summation operation is not defined between D and {type(nb).__name__}.")

    def __radd__(self, nb: Any) -> D:
        """Define the right summation of integers with the D class."""
        return self.__add__(nb)

    def __iadd__(self, nb: Any) -> D:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: Any) -> D:
        """Define the subtraction operation for the D class."""
        if isinstance(nb, (D, int, np.integer)):
            return self.__add__(-nb)
        raise TypeError(f"Subtraction operation is not defined between D and {type(nb).__name__}.")

    def __rsub__(self, nb: Any) -> D:
        """Define the right subtraction of integers with the D class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: Any) -> D:
        """Define the in-place subtraction for the d class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Any) -> D:
        """Define the product operation for the D class."""
        if isinstance(nb, D):
            return D(self.num * nb.num, self.denom + nb.denom)
        elif isinstance(nb, (int, np.integer)):
            return D(self.num * nb, self.denom)
        raise TypeError(f"Product operation is not defined between D and {type(nb).__name__}.")

    def __rmul__(self, nb: Any) -> D:
        """Define the right multiplication of integers with the D class."""
        return self.__mul__(nb)

    def __imul__(self, nb: Any) -> D:
        """Define the inplace-multiplication for the D class."""
        return self.__mul__(nb)

    def __pow__(self, n: int) -> D:
        """Define the power operation for the D class.

        Power must be a positive integer.
        """
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Expected power to be of type int, but got {type(n).__name__}.")
        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n}.")
        return D(self.num**n, n * self.denom)
    
    def __ipow__(self, n: int) -> D:
        """Define the inplace-power operation for the D class."""
        return self.__pow__(n)

