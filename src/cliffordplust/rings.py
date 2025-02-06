from __future__ import annotations

import math
from decimal import Decimal, getcontext
from typing import Any, Callable, Iterator, Union

import numpy as np

__all__ = ["D", "Zsqrt2", "Dsqrt2", "Zomega", "Domega", "INVERSE_LAMBDA", "LAMBDA"]

SQRT2: float = math.sqrt(2)


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
        # Do while the numerator and denominator are factors of two.
        while (self.num & 1) == 0 and self.denom > 0:
            # Divide the numerator by two.
            self._num >>= 1
            # Reduce the denominator exponent by one.
            self._denom -= 1

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

    def __add__(self, nb: int | D) -> D:
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

    def __radd__(self, nb: int | D) -> D:
        """Define the right summation of integers with the D class."""
        return self.__add__(nb)

    def __iadd__(self, nb: int | D) -> D:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: int | D) -> D:
        """Define the subtraction operation for the D class."""
        if isinstance(nb, (D, int, np.integer)):
            return self.__add__(-nb)
        raise TypeError(f"Subtraction operation is not defined between D and {type(nb).__name__}.")

    def __rsub__(self, nb: int | D) -> D:
        """Define the right subtraction of integers with the D class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: int | D) -> D:
        """Define the in-place subtraction for the d class."""
        return self.__sub__(nb)

    def __mul__(self, nb: int | D) -> D:
        """Define the product operation for the D class."""
        if isinstance(nb, D):
            return D(self.num * nb.num, self.denom + nb.denom)
        elif isinstance(nb, (int, np.integer)):
            return D(self.num * nb, self.denom)
        raise TypeError(f"Product operation is not defined between D and {type(nb).__name__}.")

    def __rmul__(self, nb: int | D) -> D:
        """Define the right multiplication of integers with the D class."""
        return self.__mul__(nb)

    def __imul__(self, nb: int | D) -> D:
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
            if not isinstance(input, (int, np.integer)):
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

    @property
    def is_integer(self) -> bool:
        """Return True if the ring element is an integer."""
        return self.b == 0

    @classmethod
    def from_ring(cls, nb: int | complex | Ring) -> Zsqrt2:
        """Convert a ring element to a Zsqrt2 object."""
        if isinstance(nb, Domega):
            if nb.is_zsqrt2:
                return Zsqrt2(nb.d.num, nb.c.num)
        elif isinstance(nb, Zomega):
            if nb.is_zsqrt2:
                return Zsqrt2(nb.d, nb.c)
        elif isinstance(nb, Zsqrt2):
            return nb
        elif isinstance(nb, Dsqrt2):
            if nb.is_zsqrt2:
                return Zsqrt2(nb.a.num, nb.b.num)
        elif isinstance(nb, D):
            if nb.is_integer:
                return Zsqrt2(nb.num, 0)
        elif isinstance(nb, (int, np.integer)):
            return Zsqrt2(nb, 0)
        raise ValueError(f"Cannot convert {type(nb).__name__} to Zsqrt2.")

    def sqrt2_conjugate(self) -> Zsqrt2:
        """Define the \u221A2-conjugation operation.

        Returns:
            Zsqrt2: \u221A2-conjugate of the ring element.
        """
        return Zsqrt2(self.a, -self.b)

    def __float__(self) -> float:
        """Define the float representation of the ring element."""
        bsqrt = self.b * math.sqrt(2)
        if math.isclose(self.a, -bsqrt, rel_tol=1e-5):
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
        elif isinstance(nb, (int, np.integer)):
            return self.a == nb and self.b == 0
        try:
            return float(self) == float(nb)
        except Exception:
            return False

    def __lt__(self, nb: Any) -> bool:
        """Define the < operation for the class."""
        return float(self) < nb

    def __le__(self, nb: Any) -> bool:
        """Define the <= operation for the class."""
        return self.__lt__(nb) or self.__eq__(nb)

    def __gt__(self, nb: Any) -> bool:
        """Define the > operation for the class."""
        return float(self) > nb

    def __ge__(self, nb: Any) -> bool:
        """Define the >= operation for the class."""
        return self.__gt__(nb) or self.__eq__(nb)

    def __neg__(self) -> Zsqrt2:
        """Define the negation of a ring element."""
        return Zsqrt2(-self.a, -self.b)

    def __add__(self, nb: Zsqrt2 | int) -> Zsqrt2:
        """Define the summation operation for the Zsqrt2 class.

        Allow summation with integers or Zsqrt2 objects.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a + nb.a, self.b + nb.b)
        elif isinstance(nb, (int, np.integer)):
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
        elif isinstance(nb, (int, np.integer)):
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
        elif isinstance(nb, (int, np.integer)):
            return Zsqrt2(self.a * nb, self.b * nb)
        raise TypeError(f"Multiplication operation is not defined with {type(nb).__name__}.")

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
        if not isinstance(n, (int, np.integer)):
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

    @property
    def is_zomega(self) -> bool:
        """Return True if the ring element is in the ring \u2124[\u03C9]."""
        return self.a.is_integer and self.b.is_integer

    @property
    def is_zsqrt2(self) -> bool:
        """Return True if the ring element is in the ring \u2124[\u221a2]."""
        return self.a.is_integer and self.b.is_integer

    @property
    def is_d(self) -> bool:
        """Return True if the ring element is in the ring \u2145."""
        return self.b == 0

    @property
    def is_integer(self) -> bool:
        """Return True if the ring element is an integer."""
        return self.b == 0 and self.a.is_integer

    @classmethod
    def from_ring(cls, nb: int | complex | Ring) -> Dsqrt2:
        """Convert a ring element to a Dsqrt2 object."""
        if isinstance(nb, Domega):
            if nb.is_dsqrt2:
                return Dsqrt2(a=nb.d, b=nb.c)
        elif isinstance(nb, Zomega):
            if nb.is_dsqrt2:
                return Dsqrt2(a=(nb.d, 0), b=(nb.c, 0))
        elif isinstance(nb, Dsqrt2):
            return nb
        elif isinstance(nb, Zsqrt2):
            return Dsqrt2(D(nb.a, 0), D(nb.b, 0))
        elif isinstance(nb, D):
            return Dsqrt2(nb, (0, 0))
        elif isinstance(nb, int):
            return Dsqrt2((nb, 0), (0, 0))
        raise ValueError(f"Cannot convert {type(nb).__name__} to Dsqrt2.")

    def sqrt2_conjugate(self) -> Dsqrt2:
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

    def __lt__(self, nb: Any) -> bool:
        """Define the < operation for the class."""
        return float(self) < nb

    def __le__(self, nb: Any) -> bool:
        """Define the <= operation for the class."""
        return self.__lt__(nb) or self.__eq__(nb)

    def __gt__(self, nb: Any) -> bool:
        """Define the > operation for the class."""
        return float(self) > nb

    def __ge__(self, nb: Any) -> bool:
        """Define the >= operation for the class."""
        return self.__gt__(nb) or self.__eq__(nb)

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


class Zomega:
    """Class to do symbolic computation with elements of the ring of cyclotomic integers of degree 8 \u2124[\u03C9].

    The ring element has the form a\u03C9^3 + b\u03C9^2 + c\u03C9 + d, where \u03C9 = (1 + i)/\u221a2 is the eight root of unity.
    The coefficients a, b, c, d are integers.

    The ring element can also be expressed as \u03B1 + i*\u03B2, where i = \u221a-1, and \u03B1 and \u03B2 are numbers in the ring \u2145[\u221a2]
    and are equal to \u03B1 = d + (c-a)/2 \u221a2 and \u03B2 = b + (c+a)/2 \u221a2.

    Attributes:
        a (int): \u03C9^3 coefficient of the ring element.
        b (int): \u03C9^2 coefficient of the ring element.
        c (int): \u03C9^1 coefficient of the ring element.
        d (int): \u03C9^0 coefficient of the ring element.
    """

    def __init__(self, a: int, b: int, c: int, d: int) -> None:
        """Initialize the Zomega class.

        Args:
            a (int): \u03C9^3 coefficient of the ring element.
            b (int): \u03C9^2 coefficient of the ring element.
            c (int): \u03C9^1 coefficient of the ring element.
            d (int): \u03C9^0 coefficient of the ring element.

        Raises:
            TypeError: If the class arguments are not integers.
        """
        for input in (a, b, c, d):
            if not isinstance(input, (int, np.integer)):
                raise TypeError(
                    f"Class arguments must be of type int but received {type(input).__name__}."
                )
        self._a: int = a
        self._b: int = b
        self._c: int = c
        self._d: int = d

    @property
    def a(self) -> int:
        """\u03C9^3 coefficient of the ring element."""
        return self._a

    @property
    def b(self) -> int:
        """\u03C9^2 coefficient of the ring element."""
        return self._b

    @property
    def c(self) -> int:
        """\u03C9^1 coefficient of the ring element."""
        return self._c

    @property
    def d(self) -> int:
        """\u03C9^0 coefficient of the ring element."""
        return self._d

    @property
    def is_dsqrt2(self) -> bool:
        """True if the ring element is element of \u2145[\u221a2]."""
        return self.b == 0 and self.c + self.a == 0

    @property
    def is_zsqrt2(self) -> bool:
        """True if the ring element is element of \u2124[\u221a2]."""
        return self.b == 0 and self.c + self.a == 0 and (self.c - self.a) % 2 == 0

    @property
    def is_d(self) -> bool:
        """True if the ring element is element of \u2145."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @property
    def is_integer(self) -> bool:
        """True if the ring element is an integer."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @classmethod
    def from_ring(cls, nb: int | complex | Ring) -> Zomega:
        """Convert a ring element to a Zomega object."""
        if isinstance(nb, Domega):
            if nb.is_zomega:
                return cls(a=nb.a.num, b=nb.b.num, c=nb.c.num, d=nb.d.num)
        elif isinstance(nb, Zomega):
            return nb
        elif isinstance(nb, Dsqrt2):
            if nb.is_zomega:
                return cls(a=-nb.b.num, b=0, c=nb.b.num, d=nb.a.num)
        elif isinstance(nb, Zsqrt2):
            return cls(a=-nb.b, b=0, c=nb.b, d=nb.a)
        elif isinstance(nb, D):
            if nb.is_integer:
                return cls(a=0, b=0, c=0, d=nb.num)
        elif isinstance(nb, (int, np.integer)):
            return cls(a=0, b=0, c=0, d=nb)
        elif isinstance(nb, complex):
            if nb.real.is_integer() and nb.imag.is_integer():
                return cls(a=0, b=int(nb.imag), c=0, d=int(nb.real))
        raise ValueError(f"Cannot convert {type(nb).__name__} to Zomega.")

    def real(self) -> float:
        """Return the real part of the ring element.

        Returns:
            float: Real part of the ring element in float representation.
        """
        sqrt_value = (self.c - self.a) / SQRT2
        # Maintain high precision if the two values are close to each other.
        if math.isclose(self.d, -sqrt_value, rel_tol=1e-5):
            getcontext().prec = 50
            return float(self.d + (self.c - self.a) / Decimal(2).sqrt())
        return self.d + sqrt_value

    def imag(self) -> float:
        """Return the imaginary part of the ring element.

        Returns:
            float : Imaginary part of the ring element in float representation.
        """
        sqrt_value = (self.c + self.a) / SQRT2
        # Maintain high precision if the two values are close to each other.
        if math.isclose(self.b, -sqrt_value, rel_tol=1e-5):
            getcontext().prec = 50
            return float(self.b + (self.c + self.a) / Decimal(2).sqrt())
        return self.b + sqrt_value

    def complex_conjugate(self) -> Zomega:
        """Compute complex conjugate of the ring element.

        Returns:
            Zomega: Complex conjugate of the ring element.
        """
        return Zomega(a=-self.c, b=-self.b, c=-self.a, d=self.d)

    def sqrt2_conjugate(self) -> Zomega:
        """Compute the \u221a2 conjugate of the ring element.

        Returns:
            Zomega: \u221a2 conjugate of the ring element.
        """
        return Zomega(a=-self.a, b=self.b, c=-self.c, d=self.d)

    def __repr__(self) -> str:
        """Define the string representation of the class."""
        sign: Callable[[int], str] = lambda coeff: " + " if coeff >= 0 else " - "
        value: Callable[[int], str] = lambda coeff: str(coeff) if coeff >= 0 else str(-coeff)
        return (
            str(self.a)
            + "\u03C93"
            + sign(self.b)
            + value(self.b)
            + "\u03C92"
            + sign(self.c)
            + value(self.c)
            + "\u03C91"
            + sign(self.d)
            + value(self.d)
        )

    def __getitem__(self, i: int | slice) -> int | list[int]:
        """Return the coefficients of the ring element from their index."""
        return [self.a, self.b, self.c, self.d][i]

    def __iter__(self) -> Iterator[int]:
        """Allow iteration in the class coefficients."""
        return iter([self.a, self.b, self.c, self.d])

    def __eq__(self, nb: Any) -> bool:
        """Define the == operation of the class."""
        if isinstance(nb, Zomega):
            return self.a == nb.a and self.b == nb.b and self.c == nb.c and self.d == nb.d
        elif isinstance(nb, (int, np.integer)):
            return self.is_integer and self.d == nb
        else:
            try:
                return self.real() == complex(nb).real and self.imag() == complex(nb).imag
            except Exception:
                return False

    def __neg__(self) -> Zomega:
        """Define the negation of the ring element."""
        return Zomega(-self.a, -self.b, -self.c, -self.d)

    def __complex__(self) -> complex:
        """Define the complex representation of the class."""
        return self.real() + 1.0j * self.imag()

    def __add__(self, nb: int | Zomega) -> Zomega:
        """Define the summation operation for the ring elements."""
        if isinstance(nb, Zomega):
            return Zomega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d)
        elif isinstance(nb, (int, np.integer)):
            return Zomega(self.a, self.b, self.c, self.d + nb)
        raise TypeError(
            f"Summation operation is not defined between Zomega and {type(nb).__name__}."
        )

    def __radd__(self, nb: int | Zomega) -> Zomega:
        """Define the right summation of integers with the Zomega class."""
        return self.__add__(nb)

    def __iadd__(self, nb: int | Zomega) -> Zomega:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: int | Zomega) -> Zomega:
        """Define the subtraction operation for the ring element."""
        if isinstance(nb, (Zomega, int, np.integer)):
            return self.__add__(-nb)
        raise TypeError(
            f"Subtraction operation is not defined between Zomega and {type(nb).__name__}."
        )

    def __rsub__(self, nb: int | Zomega) -> Zomega:
        """Define the right subtraction of integers with the Zomega class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: int | Zomega) -> Zomega:
        """Define the in-place subtraction for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: int | Zomega) -> Zomega:
        """Define the multiplication operation for the Zomega class."""
        if isinstance(nb, Zomega):
            a: int = (self.a * nb.d) + (self.b * nb.c) + (self.c * nb.b) + (self.d * nb.a)
            b: int = -(self.a * nb.a) + (self.b * nb.d) + (self.c * nb.c) + (self.d * nb.b)
            c: int = -(self.a * nb.b) + -(self.b * nb.a) + (self.c * nb.d) + (self.d * nb.c)
            d: int = -(self.a * nb.c) + -(self.b * nb.b) + -(self.c * nb.a) + (self.d * nb.d)
            return Zomega(a, b, c, d)

        elif isinstance(nb, (int, np.integer)):
            return Zomega(self.a * nb, self.b * nb, self.c * nb, self.d * nb)

        raise TypeError(f"Product operation is not defined between Zomega and {type(nb).__name__}.")

    def __rmul__(self, nb: int | Zomega) -> Zomega:
        """Define the right multiplication of integers with the Zomega class."""
        return self.__mul__(nb)

    def __imul__(self, nb: int | Zomega) -> Zomega:
        """Define the in-place multiplication for the class."""
        return self.__mul__(nb)

    def __pow__(self, power: int) -> Zomega:
        """Define the power operation for the Zomega class.

        Exponent must be positive integers. Uses the multinomial theorem.
        """
        if not isinstance(power, (int, np.integer)):
            raise TypeError(f"Exponent must be an integer, but received {type(power).__name__}.")
        if power < 0:
            raise ValueError(f"Expected exponent to be a positive integer, but got {power}.")

        coeff: list[int] = [0, 0, 0, 0]
        for k1 in range(power + 1):
            for k2 in range(power + 1 - k1):
                for k3 in range(power + 1 - k1 - k2):
                    k4: int = power - (k1 + k2 + k3)
                    exponent: int = 3 * k1 + 2 * k2 + k3
                    multinomial_coefficient: int = math.factorial(power) // math.prod(
                        map(math.factorial, (k1, k2, k3, k4))
                    )
                    coeff[3 - exponent % 4] += (
                        (-1) ** (exponent // 4)
                        * self.a**k1
                        * self.b**k2
                        * self.c**k3
                        * self.d**k4
                        * multinomial_coefficient
                    )
        return Zomega(coeff[0], coeff[1], coeff[2], coeff[3])

    def __ipow__(self, nb: int) -> Zomega:
        """Define the in-place power operation of the ring element."""
        return self.__pow__(nb)


class Domega:
    """Class to do symbolic computation with elements of the ring \u2145[\u03C9].

    The ring element has the form a\u03C9^3 + b\u03C9^2 + c\u03C9 + d, where \u03C9 = (1 + i)/\u221a2 is the eight root of unity.
    The coefficients a, b, c, d are dyadic fractions of the form m / 2^n, where m is an integer and n is a positive integer.
    The coefficients will be automatically reduced when the class is initialized.

    The ring element can also be expressed as \u03B1 + i*\u03B2, where i = \u221a-1, and \u03B1 and \u03B2 are numbers in the ring \u2145[\u221a2]
    and are equal to \u03B1 = d + (c-a)/2 \u221a2 and \u03B2 = b + (c+a)/2 \u221a2.

    Attributes:
        a (D): \u03C9^3 coefficient of the ring element.
        b (D): \u03C9^2 coefficient of the ring element.
        c (D): \u03C9^1 coefficient of the ring element.
        d (D): \u03C9^0 coefficient of the ring element.
    """

    def __init__(
        self,
        a: tuple[int, int] | D,
        b: tuple[int, int] | D,
        c: tuple[int, int] | D,
        d: tuple[int, int] | D,
    ) -> None:
        """Initialize the Domega class.

        Args:
            a (tuple[int, int] | D): \u03C9^3 coefficient of the ring element.
            b (tuple[int, int] | D): \u03C9^2 coefficient of the ring element.
            c (tuple[int, int] | D): \u03C9^1 coefficient of the ring element.
            d (tuple[int, int] | D): \u03C9^0 coefficient of the ring element.

        Raises:
            TypeError: If the class arguments are not 2-tuples of integers (num, denom) or D objects.
            ValueError: If the denominator exponent is negative.
        """
        for input in (a, b, c, d):
            if isinstance(input, tuple):
                if len(input) != 2 or any(
                    [not isinstance(value, (int, np.integer)) for value in input]
                ):
                    raise TypeError(
                        f"Tuples must take two integer values (num, denom) but received {input}."
                    )
                elif input[1] < 0:
                    raise ValueError(
                        f"Denominator exponent must be positive but got {input[1]} < 0."
                    )
            elif not isinstance(input, D):
                raise TypeError(
                    f"Class arguments must be of type tuple[int, int] or D objects but received {type(input).__name__}."
                )

        self._a: D = a if isinstance(a, D) else D(a[0], a[1])
        self._b: D = b if isinstance(b, D) else D(b[0], b[1])
        self._c: D = c if isinstance(c, D) else D(c[0], c[1])
        self._d: D = d if isinstance(d, D) else D(d[0], d[1])

    @property
    def a(self) -> D:
        """\u03C9^3 coefficient of the ring element."""
        return self._a

    @property
    def b(self) -> D:
        """\u03C9^2 coefficient of the ring element."""
        return self._b

    @property
    def c(self) -> D:
        """\u03C9^1 coefficient of the ring element."""
        return self._c

    @property
    def d(self) -> D:
        """\u03C9^0 coefficient of the ring element."""
        return self._d

    @property
    def is_zomega(self) -> bool:
        """True if the ring element is element of \u2124[\u03C9]."""
        return all([coeff.is_integer for coeff in self])

    @property
    def is_dsqrt2(self) -> bool:
        """True if the ring element is element of \u2145[\u221a2]."""
        return self.b == 0 and self.c + self.a == 0

    @property
    def is_zsqrt2(self) -> bool:
        """True if the ring element is element of \u2124[\u221a2]."""
        return (
            self.b == 0
            and self.c + self.a == 0
            and self.d.is_integer
            and ((self.c - self.a) * D(1, 1)).is_integer
        )

    @property
    def is_d(self) -> bool:
        """True if the ring element is element of \u2145."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @property
    def is_integer(self) -> bool:
        """True if the ring element is an integer."""
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d.is_integer

    @classmethod
    def from_ring(cls, nb: int | complex | Ring) -> Domega:
        """Convert a ring element to a Domega object."""
        if isinstance(nb, Domega):
            return nb
        elif isinstance(nb, Zomega):
            return cls((nb.a, 0), (nb.b, 0), (nb.c, 0), (nb.d, 0))
        elif isinstance(nb, Dsqrt2):
            return cls(a=-nb.b, b=(0, 0), c=nb.b, d=nb.a)
        elif isinstance(nb, Zsqrt2):
            return cls(a=(-nb.b, 0), b=(0, 0), c=(nb.b, 0), d=(nb.a, 0))
        elif isinstance(nb, D):
            return cls(a=(0, 0), b=(0, 0), c=(0, 0), d=nb)
        elif isinstance(nb, (int, np.integer)):
            return cls(a=(0, 0), b=(0, 0), c=(0, 0), d=(nb, 0))
        elif isinstance(nb, complex):
            if nb.real.is_integer() and nb.imag.is_integer():
                return cls(a=(0, 0), b=(int(nb.imag), 0), c=(0, 0), d=(int(nb.real), 0))
        raise ValueError(f"Cannot convert {type(nb).__name__} to Domega.")

    def real(self) -> float:
        """Return the real part of the ring element.

        Returns:
            float: Real part of the ring element in float representation.
        """
        sqrt_value = float(self.c - self.a) / SQRT2
        # Maintain high precision if the two values are close to each other.
        if math.isclose(float(self.d), -sqrt_value, rel_tol=1e-5):
            getcontext().prec = 50
            return float(
                self.d.num / Decimal(2) ** self.d.denom
                + (self.c - self.a).num / Decimal(2) ** (self.c - self.a).denom / Decimal(2).sqrt()
            )
        return float(self.d) + sqrt_value

    def imag(self) -> float:
        """Return the imaginary part of the ring element.

        Returns:
            float : Imaginary part of the ring element in float representation.
        """
        sqrt_value = float(self.c + self.a) / SQRT2
        # Maintain high precision if the two values are close to each other.
        if math.isclose(float(self.b), -sqrt_value, rel_tol=1e-5):
            getcontext().prec = 50
            return float(
                self.b.num / Decimal(2) ** self.b.denom
                + (self.c + self.a).num / Decimal(2) ** (self.c + self.a).denom / Decimal(2).sqrt()
            )
        return float(self.b) + sqrt_value

    def sde(self) -> int | float:
        """Return the smallest denominator exponent (sde) of base \u221a2 of the ring element.

        The sde of the ring element d \u2208 \u2145[\u03C9] is the smallest integer value k such that d * (\u221a2)^k \u2208 \u2124[\u03C9].
        """
        sde: int = 0

        # If at least one of the coefficient is not an integer, multiply all coefficients by 2^k_max to make all of them integers.
        if any([coeff.denom != 0 for coeff in self]):
            k_max: int = max([coeff.denom for coeff in self])
            coeffs: list[int] = [coeff.num * 2 ** (k_max - coeff.denom) for coeff in self]
            sde += 2 * k_max
        else:
            # If all coefficients are integers and are even, we can divide by 2 and remain in Z[ω].
            coeffs = [coeff.num for coeff in self]

            # If all coefficients are zero, return negative infinity.
            if not any(coeffs):
                return -math.inf
            while all([(coeff & 1) == 0 for coeff in coeffs]):
                coeffs = [coeff >> 1 for coeff in coeffs]
                sde -= 2

        # If a anb c have the same parity and b and d have the same parity, we can divide by √2 and remain in Z[ω] if we redefine the coefficients.
        while coeffs[0] & 1 == coeffs[2] & 1 and coeffs[1] & 1 == coeffs[3] & 1:
            alpha: int = (coeffs[1] - coeffs[3]) >> 1
            beta: int = (coeffs[2] + coeffs[0]) >> 1
            gamma: int = (coeffs[1] + coeffs[3]) >> 1
            delta: int = (coeffs[2] - coeffs[0]) >> 1
            coeffs = [alpha, beta, gamma, delta]
            sde -= 1
        return sde

    def complex_conjugate(self) -> Domega:
        """Compute complex conjugate of the ring element.

        Returns:
            Domega: Complex conjugate of the ring element.
        """
        return Domega(a=-self.c, b=-self.b, c=-self.a, d=self.d)

    def sqrt2_conjugate(self) -> Domega:
        """Compute the \u221a2 conjugate of the ring element.

        Returns:
            Domega: \u221a2 conjugate of the ring element.
        """
        return Domega(a=-self.a, b=self.b, c=-self.c, d=self.d)

    def __repr__(self) -> str:
        """Define the string representation of the class."""
        sign: Callable[[D], str] = lambda coeff: " + " if coeff >= 0 else " - "
        value: Callable[[D], str] = lambda coeff: str(coeff) if coeff >= 0 else str(-coeff)
        return (
            str(self.a)
            + "\u03C93"
            + sign(self.b)
            + value(self.b)
            + "\u03C92"
            + sign(self.c)
            + value(self.c)
            + "\u03C9"
            + sign(self.d)
            + value(self.d)
        )

    def __getitem__(self, i: int | slice) -> D | list[D]:
        """Return the coefficients of the ring element from their index."""
        return [self.a, self.b, self.c, self.d][i]

    def __iter__(self) -> Iterator[D]:
        """Allow iteration in the class coefficients."""
        return iter([self.a, self.b, self.c, self.d])

    def __eq__(self, nb: Any) -> bool:
        """Define the == operation of the class."""
        if isinstance(nb, Domega):
            return self.a == nb.a and self.b == nb.b and self.c == nb.c and self.d == nb.d
        elif isinstance(nb, (D, int, np.integer)):
            return self.is_d and self.d == nb
        else:
            try:
                return self.real() == complex(nb).real and self.imag() == complex(nb).imag
            except Exception:
                return False

    def __neg__(self) -> Domega:
        """Define the negation of the ring element."""
        return Domega(-self.a, -self.b, -self.c, -self.d)

    def __complex__(self) -> complex:
        """Define the complex representation of the class."""
        return self.real() + 1.0j * self.imag()

    def __add__(self, nb: int | D | Domega) -> Domega:
        """Define the summation operation for the ring elements."""
        if isinstance(nb, (Domega)):
            return Domega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d)
        elif isinstance(nb, (D, int, np.integer)):
            return Domega(self.a, self.b, self.c, self.d + nb)
        raise TypeError(
            f"Summation operation is not defined between Domega and {type(nb).__name__}."
        )

    def __radd__(self, nb: int | D | Domega) -> Domega:
        """Define the right summation of integers and D objects with the Domega class."""
        return self.__add__(nb)

    def __iadd__(self, nb: int | D | Domega) -> Domega:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: int | D | Domega) -> Domega:
        """Define the subtraction operation for the ring element."""
        if isinstance(nb, (Domega, D, int, np.integer)):
            return self.__add__(-nb)
        raise TypeError(
            f"Subtraction operation is not defined between Domega and {type(nb).__name__}."
        )

    def __rsub__(self, nb: int | D | Domega) -> Domega:
        """Define the right subtraction of integers and D objects with the Domega class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: int | D | Domega) -> Domega:
        """Define the in-place subtraction for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: int | D | Domega) -> Domega:
        """Define the multiplication operation for the Domega class."""
        if isinstance(nb, Domega):
            a: D = (self.a * nb.d) + (self.b * nb.c) + (self.c * nb.b) + (self.d * nb.a)
            b: D = -(self.a * nb.a) + (self.b * nb.d) + (self.c * nb.c) + (self.d * nb.b)
            c: D = -(self.a * nb.b) + -(self.b * nb.a) + (self.c * nb.d) + (self.d * nb.c)
            d: D = -(self.a * nb.c) + -(self.b * nb.b) + -(self.c * nb.a) + (self.d * nb.d)
            return Domega(a, b, c, d)

        elif isinstance(nb, (D, int, np.integer)):
            return Domega(self.a * nb, self.b * nb, self.c * nb, self.d * nb)

        raise TypeError(f"Product operation is not defined between Domega and {type(nb).__name__}.")

    def __rmul__(self, nb: int | D | Domega) -> Domega:
        """Define the right multiplication of integers and D objects with the Domega class."""
        return self.__mul__(nb)

    def __imul__(self, nb: int | D | Domega) -> Domega:
        """Define the in-place multiplication for the class."""
        return self.__mul__(nb)

    def __pow__(self, power: int) -> Domega:
        """Define the power operation for the Domega class.

        Exponent must be positive integers. Uses the multinomial theorem.
        """
        if not isinstance(power, (int, np.integer)):
            raise TypeError(f"Exponent must be an integer, but received {type(power).__name__}.")
        if power < 0:
            raise ValueError(f"Expected exponent to be a positive integer, but got {power}.")

        coeff: list[D] = [D(0, 0), D(0, 0), D(0, 0), D(0, 0)]
        for k1 in range(power + 1):
            for k2 in range(power + 1 - k1):
                for k3 in range(power + 1 - k1 - k2):
                    k4: int = power - (k1 + k2 + k3)
                    exponent: int = 3 * k1 + 2 * k2 + k3
                    multinomial_coefficient: int = math.factorial(power) // math.prod(
                        map(math.factorial, (k1, k2, k3, k4))
                    )
                    coeff[3 - exponent % 4] += (
                        (-1) ** (exponent // 4)
                        * self.a**k1
                        * self.b**k2
                        * self.c**k3
                        * self.d**k4
                        * multinomial_coefficient
                    )
        return Domega(coeff[0], coeff[1], coeff[2], coeff[3])

    def __ipow__(self, nb: int) -> Domega:
        """Define the in-place power operation of the ring element."""
        return self.__pow__(nb)


Ring = Union[D, Zsqrt2, Dsqrt2, Zomega, Domega]

# LAMBDA = 1 + \u221A2 is used to scale 1D grid problems.
LAMBDA: Zsqrt2 = Zsqrt2(1, 1)

# INVERSE_LAMBDA = -1 + \u221A2 is the inverse of LAMBDA. It is used to scale 1D grid problem.
INVERSE_LAMBDA: Zsqrt2 = -LAMBDA.sqrt2_conjugate()
