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
Symbolic computation with ring elements.

The :mod:`rings` module provides tools for symbolic computations with elements of various mathematical rings.
Rings are used in many algorithms for the approximation of z-rotation gates into Clifford+T unitaries.

The module includes the following classes:
    - :class:`D`: Ring of dyadic fractions :math:`\\mathbb{D}`.
    - :class:`Zsqrt2`: Ring of quadratic integers with radicand 2 :math:`\\mathbb{Z}[\\sqrt{2}]`.
    - :class:`Dsqrt2`: Ring of quadratic dyadic fractions with radicand 2 :math:`\\mathbb{D}[\\sqrt{2}]`.
    - :class:`Zomega`: Ring of cyclotomic integers of degree 8 :math:`\\mathbb{Z}[\\omega]`.
    - :class:`Domega`: Ring of cyclotomic dyadic fractions of degree 8 :math:`\\mathbb{D}[\\omega]`.

For more information, see :cite:`rings_ross`.
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext
from typing import Any, Callable, Iterator, Union

import mpmath as mp
import numpy as np

__all__ = ["D", "Zsqrt2", "Dsqrt2", "Zomega", "Domega", "INVERSE_LAMBDA", "LAMBDA"]

SQRT2: float = math.sqrt(2)


class D:
    """
    Class to do symbolic computation with elements of the ring of dyadic fractions :math:`\\mathbb{D}`.

    The ring element has the form :math:`a/(2^k)`, where a is an integer and k is a positive integer.

    Parameters:
        num (int): Numerator of the ring element.
        denom (int): Power of 2 in the denominator of the ring element.
        is_integer (bool): True if the ring element is an integer.
    """

    def __init__(self, num: int, denom: int) -> None:
        """
        Initialize the ring element.

        Args:
            num (int): Numerator of the ring element
            denom (int): Power of 2 in the denominator of the ring element. Must be positive.

        Raises:
            TypeError: If the numerator or the denominator exponent are not integers.
            ValueError: If the denominator exponent is negative.
        """
        for arg in (num, denom):
            if not isinstance(arg, (int, np.integer)):
                raise TypeError(
                    f"Class arguments must be of type int, but received {type(arg).__name__}."
                )

        if denom < 0:
            raise ValueError(f"Denominator exponent must be positive, but got {denom}.")

        self._num: int = num
        self._denom: int = denom

        # Reduce the fraction if the numerator is even.
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

    def mpfloat(self) -> float:
        """Define the mpfloat value of the D class."""
        return self.num / 2 ** mp.mpf(self.denom)

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of the D class."""
        if isinstance(nb, D):
            return self.num == nb.num and self.denom == nb.denom

        elif isinstance(nb, (int, np.integer)):
            return self.denom == 0 and self.num == nb

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

        raise TypeError(f"Summation is not defined between D and {type(nb).__name__}.")

    def __radd__(self, nb: int | D) -> D:
        """Define right summation with the D class."""
        return self.__add__(nb)

    def __iadd__(self, nb: int | D) -> D:
        """Define the in-place summation operation for the D class."""
        return self.__add__(nb)

    def __sub__(self, nb: int | D) -> D:
        """Define the subtraction operation for the D class."""
        if isinstance(nb, (D, int, np.integer)):
            return self.__add__(-nb)

        raise TypeError(f"Subtraction is not defined between D and {type(nb).__name__}.")

    def __rsub__(self, nb: int | D) -> D:
        """Define the right subtraction with the D class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: int | D) -> D:
        """Define the in-place subtraction for the D class."""
        return self.__sub__(nb)

    def __mul__(self, nb: int | D) -> D:
        """Define the product operation for the D class."""
        if isinstance(nb, D):
            return D(self.num * nb.num, self.denom + nb.denom)

        elif isinstance(nb, (int, np.integer)):
            return D(self.num * nb, self.denom)

        raise TypeError(f"Product is not defined between D and {type(nb).__name__}.")

    def __rmul__(self, nb: int | D) -> D:
        """Define the right multiplication with the D class."""
        return self.__mul__(nb)

    def __imul__(self, nb: int | D) -> D:
        """Define the inplace-multiplication for the D class."""
        return self.__mul__(nb)

    def __pow__(self, n: int) -> D:
        """Define the power operation for the D class.

        Power must be a positive integer.
        """
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Expected power to be an integer, but got {type(n).__name__}.")

        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n}.")

        return D(self.num**n, n * self.denom)

    def __ipow__(self, n: int) -> D:
        """Define the inplace-power for the D class."""
        return self.__pow__(n)


class Zsqrt2:
    """
    Class to do symbolic computation with elements of the ring of quadratic integers with radicand 2 :math:`\\mathbb{Z}[\\sqrt{2}]`.

    The ring element has the form :math:`a + b\\sqrt{2}`, where a and b are integers.

    Parameters:
        a (int): Integer coefficient of the ring element.
        b (int): :math:`\\sqrt{2}` coefficient of the ring element.
    """

    def __init__(self, a: int, b: int) -> None:
        """
        Initialize the ring element.

        Args:
            a (int): Integer coefficient of the ring element.
            b (int): :math:`\\sqrt{2}` coefficient of the ring element.

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
        """:math:`\\sqrt{2}` coefficient of the ring element."""
        return self._b

    @property
    def is_integer(self) -> bool:
        """Return True if the ring element is an integer."""
        return self.b == 0

    @classmethod
    def from_ring(cls, nb: int | Ring) -> Zsqrt2:
        """
        Convert a ring element to a Zsqrt2 object. The conversion must be possible.

        Args:
            nb (int | Ring): Ring element or integer to convert to Zsqrt2.

        Returns:
            Zsqrt2: Zsqrt2 object converted from the ring element.

        Raises:
            ValueError: If the conversion is not possible.
        """
        if isinstance(nb, Domega):
            if nb.is_zsqrt2:
                return cls(nb.d.num, nb.c.num)

        elif isinstance(nb, Zomega):
            if nb.is_zsqrt2:
                return cls(nb.d, nb.c)

        elif isinstance(nb, Dsqrt2):
            if nb.is_zsqrt2:
                return cls(nb.a.num, nb.b.num)

        elif isinstance(nb, Zsqrt2):
            return nb

        elif isinstance(nb, D):
            if nb.is_integer:
                return cls(nb.num, 0)

        elif isinstance(nb, (int, np.integer)):
            return cls(nb, 0)

        raise ValueError(f"Cannot convert {type(nb).__name__} to Zsqrt2.")

    def sqrt2_conjugate(self) -> Zsqrt2:
        """Define the :math:`\\sqrt{2}`-conjugation operation.

        Returns:
            Zsqrt2: :math:`\\sqrt{2}`-conjugate of the ring element.
        """
        return Zsqrt2(self.a, -self.b)

    def __float__(self) -> float:
        """Define the float value of the ring element."""
        bsqrt = self.b * SQRT2

        if math.isclose(self.a, -bsqrt, rel_tol=1e-5):
            # Maintain high precision if the two values are close to each other.
            getcontext().prec = 50
            return float(self.a + self.b * Decimal(2).sqrt())

        return self.a + bsqrt

    def mpfloat(self) -> float:
        """Define the `mpfloat` value of the Zsqrt2 class."""
        return self.a + self.b * mp.sqrt(2)

    def __getitem__(self, i: int) -> int:
        """Access the values of a and b from their index."""
        return (self.a, self.b)[i]

    def __repr__(self) -> str:
        """Define the string representation of the ring element."""
        if self.b < 0:
            return str(self.a) + str(self.b) + "\u221a2"

        return str(self.a) + "+" + str(self.b) + "\u221a2"

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of Zsqrt2 classes."""
        if isinstance(nb, Zsqrt2):
            return self.a == nb.a and self.b == nb.b

        elif isinstance(nb, (int, np.integer)):
            return self.a == nb and self.b == 0

        return False

    def __lt__(self, nb: Any) -> bool:
        """Define the < operation for the Zsqrt2 class."""
        return float(self) < nb

    def __le__(self, nb: Any) -> bool:
        """Define the <= operation for the Zsqrt2 class."""
        return self.__lt__(nb) or self.__eq__(nb)

    def __gt__(self, nb: Any) -> bool:
        """Define the > operation for the Zsqrt2 class."""
        return float(self) > nb

    def __ge__(self, nb: Any) -> bool:
        """Define the >= operation for the Zsqrt2 class."""
        return self.__gt__(nb) or self.__eq__(nb)

    def __neg__(self) -> Zsqrt2:
        """Define the negation of the Zsqrt2 class."""
        return Zsqrt2(-self.a, -self.b)

    def __add__(self, nb: int | Zsqrt2) -> Zsqrt2:
        """Define the summation operation for the Zsqrt2 class.

        Allow summation with integers or Zsqrt2 objects.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a + nb.a, self.b + nb.b)

        elif isinstance(nb, (int, np.integer)):
            return Zsqrt2(self.a + nb, self.b)

        raise TypeError(f"Summation is not defined between Zsqrt2 and {type(nb).__name__}.")

    def __radd__(self, nb: int | Zsqrt2) -> Zsqrt2:
        """Define the right summation with the Zsqrt2 class."""
        return self.__add__(nb)

    def __iadd__(self, nb: int | Zsqrt2) -> Zsqrt2:
        """Define the in-place summation for the Zsqrt2 class."""
        return self.__add__(nb)

    def __sub__(self, nb: int | Zsqrt2) -> Zsqrt2:
        """Define the subtraction operation for the Zsqrt2 class.

        Allow subtraction with integers and Zsqrt2 objects.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a - nb.a, self.b - nb.b)

        elif isinstance(nb, (int, np.integer)):
            return Zsqrt2(self.a - nb, self.b)

        raise TypeError(f"Subtraction is not defined between Zsqrt2 and {type(nb).__name__}.")

    def __rsub__(self, nb: int | Zsqrt2) -> Zsqrt2:
        """Define the right subtraction with the Zsqrt2 class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: int | Zsqrt2) -> Zsqrt2:
        """Define in-place subtraction for the Zsqrt2 class."""
        return self.__sub__(nb)

    def __mul__(self, nb: int | Zsqrt2) -> Zsqrt2:
        """Define the multiplication operation for the Zsqrt2 class.

        Allow multiplication with integers and Zsqrt2 objects.
        """
        if isinstance(nb, Zsqrt2):
            return Zsqrt2(self.a * nb.a + 2 * self.b * nb.b, self.a * nb.b + self.b * nb.a)

        elif isinstance(nb, (int, np.integer)):
            return Zsqrt2(self.a * nb, self.b * nb)

        raise TypeError(f"Multiplication is not defined between Zsqrt2 and {type(nb).__name__}.")

    def __rmul__(self, nb: int | Zsqrt2) -> Zsqrt2:
        """Define the right multiplication with the Zsqrt2 class."""
        return self.__mul__(nb)

    def __imul__(self, nb: int | Zsqrt2) -> Zsqrt2:
        """Define in-place multiplication for the Zsqrt2 class."""
        return self.__mul__(nb)

    def __pow__(self, n: int) -> Zsqrt2:
        """Define the power operation for the Zsqrt2 class. Exponent must be a positive integer."""
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Expected power to be an integer, but got {type(n).__name__}.")

        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n}.")

        # Compute the power
        nth_power = self
        result = Zsqrt2(1, 0)

        while n:
            if n & 1:
                result *= nth_power
            nth_power *= nth_power
            n >>= 1

        return result

    def __ipow__(self, nb: int) -> Zsqrt2:
        """Define in-place power for the Zsqrt2 class."""
        return self.__pow__(nb)


class Dsqrt2:
    """
    Class to do symbolic computation with elements of the ring of quadratic dyadic fractions :math:`\\mathbb{D}[\\sqrt{2}]`.

    The ring element has the form :math:`a + b\\sqrt{2}`, where a and b are dyadic fractions of the form :math:`m/2^n`,
    where m is an integer and n is a positive integer.
    The coefficients are automatically reduced when the class is initialized.

    Parameters:
        a (D): Rational coefficient of the ring element.
        b (D): :math:`\\sqrt{2}` coefficient of the ring element.
    """

    def __init__(self, a: tuple[int, int] | D, b: tuple[int, int] | D) -> None:
        """
        Initialize the Dsqrt2 class.

        Args:
            a (tuple[int, int] | D): Rational coefficient of the ring element.
            b (tuple[int, int] | D): :math:`\\sqrt{2}` coefficient of the ring element.

        Raises:
            TypeError: If the class arguments are not 2-tuples of integers or D objects.
            ValueError: If the denominator exponent is negative.
        """
        for input in (a, b):
            if isinstance(input, tuple):
                if len(input) != 2 or any(
                    [not isinstance(value, (int, np.integer)) for value in input]
                ):
                    raise TypeError(
                        f"Tuples must take two integer values (num, denom), but received {input}."
                    )

                elif input[1] < 0:
                    raise ValueError(f"Denominator exponent must be positive, but got {input[1]}.")

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
        """:math:`\\sqrt{2}` coefficient of the ring element."""
        return self._b

    @property
    def is_zomega(self) -> bool:
        """Return True if the ring element is in the ring :math:`\\mathbb{Z}[\\omega]`."""
        return self.a.is_integer and self.b.is_integer

    @property
    def is_zsqrt2(self) -> bool:
        """Return True if the ring element is in the ring :math:`\\mathbb{Z}[\\sqrt{2}]`."""
        return self.a.is_integer and self.b.is_integer

    @property
    def is_d(self) -> bool:
        """Return True if the ring element is in the ring :math:`\\mathbb{D}`."""
        return self.b == 0

    @property
    def is_integer(self) -> bool:
        """Return True if the ring element is an integer."""
        return self.b == 0 and self.a.is_integer

    @classmethod
    def from_ring(cls, nb: int | Ring) -> Dsqrt2:
        """
        Convert a ring element to a Dsqrt2 object. The conversion must be possible.

        Args:
            nb (int | Ring): Ring element or integer to convert to Dsqrt2.

        Returns:
            Dsqrt2: Dsqrt2 object converted from the ring element.

        Raises:
            ValueError: If the conversion is not possible.
        """
        if isinstance(nb, Domega):
            if nb.is_dsqrt2:
                return cls(nb.d, nb.c)

        elif isinstance(nb, Zomega):
            if nb.is_dsqrt2:
                return cls((nb.d, 0), (nb.c, 0))

        elif isinstance(nb, Dsqrt2):
            return nb

        elif isinstance(nb, Zsqrt2):
            return cls((nb.a, 0), (nb.b, 0))

        elif isinstance(nb, D):
            return cls(nb, (0, 0))

        elif isinstance(nb, int):
            return cls((nb, 0), (0, 0))

        raise ValueError(f"Cannot convert {type(nb).__name__} to Dsqrt2.")

    def sqrt2_conjugate(self) -> Dsqrt2:
        """Define the :math:`\\sqrt{2}`-conjugation operation.

        Returns:
            Dsqrt2: :math:`\\sqrt{2}`-conjugate of the ring element.
        """
        return Dsqrt2(self.a, -self.b)

    def __float__(self) -> float:
        """Define the float value of the ring element."""
        bsqrt = float(self.b) * SQRT2
        a = float(self.a)

        if math.isclose(a, -bsqrt, rel_tol=1e-5):
            # Maintain high precision if the two values are close to each other.
            getcontext().prec = 50
            return float(
                self.a.num / (Decimal(2) ** self.a.denom)
                + self.b.num * Decimal(2).sqrt() / (2**self.b.denom)
            )

        return a + bsqrt

    def mpfloat(self) -> float:
        """Define the `mpfloat` value of the Dsqrt2 class."""
        return self.a.mpfloat() + self.b.mpfloat() * mp.sqrt(2)

    def __getitem__(self, i: int) -> D:
        """Access the values of a and b from their index."""
        return (self.a, self.b)[i]

    def __repr__(self) -> str:
        """Define the string representation of the ring element."""
        if self.b < 0:
            return str(self.a) + str(self.b) + "\u221a2"

        return str(self.a) + "+" + str(self.b) + "\u221a2"

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of Dsqrt2 classes."""
        if isinstance(nb, Dsqrt2):
            return self.a == nb.a and self.b == nb.b

        elif isinstance(nb, (D, int, np.integer)):
            return self.a == nb and self.b == 0

        return False

    def __lt__(self, nb: Any) -> bool:
        """Define the < operation for the Dsqrt2 class."""
        return float(self) < nb

    def __le__(self, nb: Any) -> bool:
        """Define the <= operation for the Dsqrt2 class."""
        return self.__lt__(nb) or self.__eq__(nb)

    def __gt__(self, nb: Any) -> bool:
        """Define the > operation for the Dsqrt2 class."""
        return float(self) > nb

    def __ge__(self, nb: Any) -> bool:
        """Define the >= operation for the Dsqrt2 class."""
        return self.__gt__(nb) or self.__eq__(nb)

    def __neg__(self) -> Dsqrt2:
        """Define the negation of the Dsqrt2 class."""
        return Dsqrt2(-self.a, -self.b)

    def __add__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the summation operation for the Dsqrt2 class."""
        if isinstance(nb, Dsqrt2):
            return Dsqrt2(self.a + nb.a, self.b + nb.b)

        elif isinstance(nb, (D, int, np.integer)):
            return Dsqrt2(self.a + nb, self.b)

        raise TypeError(f"Summation is not defined between Dsqrt2 and {type(nb).__name__}.")

    def __radd__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the right summation with the Dsqrt2 class."""
        return self.__add__(nb)

    def __iadd__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the in-place summation for the Dsqrt2 class."""
        return self.__add__(nb)

    def __sub__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the subtraction operation for the Dsqrt2 class."""
        if isinstance(nb, Dsqrt2):
            return Dsqrt2(self.a - nb.a, self.b - nb.b)

        elif isinstance(nb, (D, int, np.integer)):
            return Dsqrt2(self.a - nb, self.b)

        raise TypeError(f"Subtraction is not defined between Dsqrt2 and {type(nb).__name__}.")

    def __rsub__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the right subtraction with the Dsqrt2 class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define in-place subtraction operation for the Dsqrt2 class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the multiplication operation for the Dsqrt2 class."""
        if isinstance(nb, Dsqrt2):
            return Dsqrt2(self.a * nb.a + self.b * nb.b * 2, self.a * nb.b + self.b * nb.a)

        elif isinstance(nb, (D, int, np.integer)):
            return Dsqrt2(self.a * nb, self.b * nb)

        raise TypeError(f"Multiplication is not defined between Dsqrt2 and {type(nb).__name__}.")

    def __rmul__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define the right multiplication with the Dsqrt2 class."""
        return self.__mul__(nb)

    def __imul__(self, nb: Dsqrt2 | D | int) -> Dsqrt2:
        """Define in-place multiplication operation for the Dsqrt2 class."""
        return self.__mul__(nb)

    def __pow__(self, n: int) -> Dsqrt2:
        """Define the power operation for the sqrt2 class. Exponent must be a positive integer."""
        # Check the input
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Expected power to be an integer, but got {type(n).__name__}.")

        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n}.")

        # Compute the power
        nth_power = self
        result = Dsqrt2((1, 0), (0, 0))

        while n:
            if n & 1:
                result *= nth_power
            nth_power *= nth_power
            n >>= 1

        return result

    def __ipow__(self, nb: int) -> Dsqrt2:
        """Define in-place power operation for the Dsqrt2 class."""
        return self.__pow__(nb)


class Zomega:
    """
    Class to do symbolic computation with elements of the ring of cyclotomic integers of degree 8 :math:`\\mathbb{Z}[\\omega]`.

    The ring element has the form :math:`a\\omega^3 + b\\omega^2 + c\\omega + d`, where :math:`\\omega = (1 + i)/\\sqrt{2}`.
    The coefficients a, b, c, d are integers.

    The ring element can also be expressed as :math:`\\alpha + i\\beta`, where :math:`i = \\sqrt{-1}`, and :math:`\\alpha` and :math:`\\beta` are numbers in the ring :math:`\\mathbb{D}[\\sqrt{2}]`.
    These numbers are related to the coefficient a, b, c and d through the expressions: :math:`\\alpha = d + (c-a)/2 \\sqrt{2}` and :math:`\\beta = b + (c+a)/2 \\sqrt{2}`.

    Parameters:
        a (int): :math:`\\omega^3` coefficient of the ring element.
        b (int): :math:`\\omega^2` coefficient of the ring element.
        c (int): :math:`\\omega^1` coefficient of the ring element.
        d (int): :math:`\\omega^0` coefficient of the ring element.
    """

    def __init__(self, a: int, b: int, c: int, d: int) -> None:
        """
        Initialize the Zomega class.

        Args:
            a (int): :math:`\\omega^3` coefficient of the ring element.
            b (int): :math:`\\omega^2` coefficient of the ring element.
            c (int): :math:`\\omega^1` coefficient of the ring element.
            d (int): :math:`\\omega^0` coefficient of the ring element.

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
        """:math:`\\omega^3` coefficient of the ring element."""
        return self._a

    @property
    def b(self) -> int:
        """:math:`\\omega^2` coefficient of the ring element."""
        return self._b

    @property
    def c(self) -> int:
        """:math:`\\omega^1` coefficient of the ring element."""
        return self._c

    @property
    def d(self) -> int:
        """:math:`\\omega^0` coefficient of the ring element."""
        return self._d

    @property
    def is_dsqrt2(self) -> bool:
        """True if the ring element is element of :math:`\\mathbb{D}[\\sqrt{2}]`."""
        return self.b == 0 and self.c + self.a == 0

    @property
    def is_zsqrt2(self) -> bool:
        """True if the ring element is element of :math:`\\mathbb{Z}[\\sqrt{2}]`."""
        return self.b == 0 and self.c + self.a == 0

    @property
    def is_d(self) -> bool:
        """True if the ring element is element of :math:`\\mathbb{D}`."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @property
    def is_integer(self) -> bool:
        """True if the ring element is an integer."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @classmethod
    def from_ring(cls, nb: int | complex | Ring) -> Zomega:
        """
        Convert a ring element to a Zomega object. The conversion must be possible.

        Args:
            nb (int | complex | Ring): Ring element to convert to Zomega.

        Returns:
            Zomega: Zomega object converted from the ring element.

        Raises:
            ValueError: If the conversion is not possible.
        """
        if isinstance(nb, Domega):
            if nb.is_zomega:
                return cls(nb.a.num, nb.b.num, nb.c.num, nb.d.num)

        elif isinstance(nb, Zomega):
            return nb

        elif isinstance(nb, Dsqrt2):
            if nb.is_zomega:
                return cls(-nb.b.num, 0, nb.b.num, nb.a.num)

        elif isinstance(nb, Zsqrt2):
            return cls(-nb.b, 0, nb.b, nb.a)

        elif isinstance(nb, D):
            if nb.is_integer:
                return cls(0, 0, 0, nb.num)

        elif isinstance(nb, (int, np.integer)):
            return cls(0, 0, 0, nb)

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

        if math.isclose(self.d, -sqrt_value, rel_tol=1e-5):
            # Maintain high precision if the two values are close to each other.
            getcontext().prec = 50
            return float(self.d + (self.c - self.a) / Decimal(2).sqrt())

        return self.d + sqrt_value

    def mp_real(self) -> float:
        """Return the real part of the ring element in `mpfloat` representation."""
        return self.d + (self.c - self.a) / mp.sqrt(2)

    def imag(self) -> float:
        """Return the imaginary part of the ring element.

        Returns:
            float : Imaginary part of the ring element in float representation.
        """
        sqrt_value = (self.c + self.a) / SQRT2

        if math.isclose(self.b, -sqrt_value, rel_tol=1e-5):
            # Maintain high precision if the two values are close to each other.
            getcontext().prec = 50
            return float(self.b + (self.c + self.a) / Decimal(2).sqrt())

        return self.b + sqrt_value

    def mp_imag(self) -> float:
        """Return the imaginary part of the ring element in `mpfloat` representation."""
        return self.b + (self.c + self.a) / mp.sqrt(2)

    def __complex__(self) -> complex:
        """Define the complex value of the ring element."""
        return self.real() + 1j * self.imag()

    def mpcomplex(self) -> complex:
        """Define the `mpcomplex` value of the Zomega class."""
        return mp.mpc(self.mp_real(), self.mp_imag())

    def complex_conjugate(self) -> Zomega:
        """Compute the complex conjugate of the ring element.

        Returns:
            Zomega: Complex conjugate of the ring element.
        """
        return Zomega(a=-self.c, b=-self.b, c=-self.a, d=self.d)

    def sqrt2_conjugate(self) -> Zomega:
        """Compute the :math:`\\sqrt{2}`-conjugate of the ring element.

        Returns:
            Zomega: :math:`\\sqrt{2}`-conjugate of the ring element.
        """
        return Zomega(a=-self.a, b=self.b, c=-self.c, d=self.d)

    def __repr__(self) -> str:
        """Define the string representation of the ring element."""
        sign: Callable[[int], str] = lambda coeff: " + " if coeff >= 0 else " - "
        value: Callable[[int], str] = lambda coeff: str(coeff) if coeff >= 0 else str(-coeff)
        return (
            str(self.a)
            + "\u03c93"
            + sign(self.b)
            + value(self.b)
            + "\u03c92"
            + sign(self.c)
            + value(self.c)
            + "\u03c91"
            + sign(self.d)
            + value(self.d)
        )

    def __getitem__(self, i: int | slice) -> int | list[int]:
        """Return the coefficients of the ring element from their index."""
        return [self.a, self.b, self.c, self.d][i]

    def __iter__(self) -> Iterator[int]:
        """Allow iteration through the class coefficients."""
        return iter([self.a, self.b, self.c, self.d])

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of Zomega classes."""
        if isinstance(nb, Zomega):
            return self.a == nb.a and self.b == nb.b and self.c == nb.c and self.d == nb.d

        elif isinstance(nb, (int, np.integer)):
            return self.is_integer and self.d == nb

        return False

    def __neg__(self) -> Zomega:
        """Define the negation of the ring element."""
        return Zomega(-self.a, -self.b, -self.c, -self.d)

    def __add__(self, nb: int | Zomega) -> Zomega:
        """Define the summation operation for the Zomega class."""
        if isinstance(nb, Zomega):
            return Zomega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d)

        elif isinstance(nb, (int, np.integer)):
            return Zomega(self.a, self.b, self.c, self.d + nb)

        raise TypeError(f"Summation is not defined between Zomega and {type(nb).__name__}.")

    def __radd__(self, nb: int | Zomega) -> Zomega:
        """Define the right summation with the Zomega class."""
        return self.__add__(nb)

    def __iadd__(self, nb: int | Zomega) -> Zomega:
        """Define the in-place summation for the Zomega class."""
        return self.__add__(nb)

    def __sub__(self, nb: int | Zomega) -> Zomega:
        """Define the subtraction operation for the Zomega class."""
        if isinstance(nb, (Zomega, int, np.integer)):
            return self.__add__(-nb)

        raise TypeError(f"Subtraction is not defined between Zomega and {type(nb).__name__}.")

    def __rsub__(self, nb: int | Zomega) -> Zomega:
        """Define the right subtraction with the Zomega class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: int | Zomega) -> Zomega:
        """Define the in-place subtraction for the Zomega class."""
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

        raise TypeError(f"Product is not defined between Zomega and {type(nb).__name__}.")

    def __rmul__(self, nb: int | Zomega) -> Zomega:
        """Define the right multiplication with the Zomega class."""
        return self.__mul__(nb)

    def __imul__(self, nb: int | Zomega) -> Zomega:
        """Define the in-place multiplication for the Zomega class."""
        return self.__mul__(nb)

    def __pow__(self, power: int) -> Zomega:
        """Define the power operation for the Zomega class. Exponent must be a positive integer."""
        # Check the input
        if not isinstance(power, (int, np.integer)):
            raise TypeError(f"Exponent must be an integer, but received {type(power).__name__}.")

        elif power < 0:
            raise ValueError(f"Exponent must be a positive integer, but got {power}.")

        # Compute the power
        nth_power = self
        result = Zomega(0, 0, 0, 1)

        while power:
            if power & 1:
                result *= nth_power
            nth_power *= nth_power
            power >>= 1

        return result

    def __ipow__(self, nb: int) -> Zomega:
        """Define the in-place power operation of the Zomega class."""
        return self.__pow__(nb)


class Domega:
    """
    Class to do symbolic computation with elements of the ring :math:`\\mathbb{D}[\\omega]`.

    The ring element has the form :math:`a\\omega^3 + b\\omega^2 + c\\omega + d`, where :math:`\\omega = (1 + i)/\\sqrt{2}`.
    The coefficients a, b, c, d are dyadic fractions of the form :math:`m / 2^n`, where m is an integer and n is a positive integer.
    The coefficients are automatically reduced when the class is initialized.

    The ring element can also be expressed as :math:`\\alpha + i\\beta`, where :math:`i = \\sqrt{-1}`, and :math:`\\alpha` and :math:`\\beta` are numbers in the ring :math:`\\mathbb{D}[\\sqrt{2}]`.
    These numbers are related to the coefficient a, b, c and d through the expressions: :math:`\\alpha = d + (c-a)/2 \\sqrt{2}` and :math:`\\beta = b + (c+a)/2 \\sqrt{2}`.


    Parameters:
        a (D): :math:`\\omega^3` coefficient of the ring element.
        b (D): :math:`\\omega^2` coefficient of the ring element.
        c (D): :math:`\\omega^1` coefficient of the ring element.
        d (D): :math:`\\omega^0` coefficient of the ring element.
    """

    def __init__(
        self,
        a: tuple[int, int] | D,
        b: tuple[int, int] | D,
        c: tuple[int, int] | D,
        d: tuple[int, int] | D,
    ) -> None:
        """
        Initialize the Domega class.

        Args:
            a (tuple[int, int] | D): :math:`\\omega^3` coefficient of the ring element.
            b (tuple[int, int] | D): :math:`\\omega^2` coefficient of the ring element.
            c (tuple[int, int] | D): :math:`\\omega^1` coefficient of the ring element.
            d (tuple[int, int] | D): :math:`\\omega^0` coefficient of the ring element.

        Raises:
            TypeError: If the class arguments are not 2-tuples of integers or D objects.
            ValueError: If the denominator exponent is negative.
        """
        for input in (a, b, c, d):
            if isinstance(input, tuple):
                if len(input) != 2 or any(
                    [not isinstance(value, (int, np.integer)) for value in input]
                ):
                    raise TypeError(
                        f"Tuples must take two integer values (num, denom), but received {input}."
                    )

                elif input[1] < 0:
                    raise ValueError(f"Denominator exponent must be positive but got {input[1]}.")

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
        """:math:`\\omega^3` coefficient of the ring element."""
        return self._a

    @property
    def b(self) -> D:
        """:math:`\\omega^2` coefficient of the ring element."""
        return self._b

    @property
    def c(self) -> D:
        """:math:`\\omega^1` coefficient of the ring element."""
        return self._c

    @property
    def d(self) -> D:
        """:math:`\\omega^0` coefficient of the ring element."""
        return self._d

    @property
    def is_zomega(self) -> bool:
        """True if the ring element is element of :math:`\\mathbb{Z}[\\omega]`."""
        return self.a.is_integer and self.b.is_integer and self.c.is_integer and self.d.is_integer

    @property
    def is_dsqrt2(self) -> bool:
        """True if the ring element is element of :math:`\\mathbb{D}[\\sqrt{2}]`."""
        return self.b == 0 and self.c + self.a == 0

    @property
    def is_zsqrt2(self) -> bool:
        """True if the ring element is element of :math:`\\mathbb{Z}[\\sqrt{2}]`."""
        return self.b == 0 and self.c + self.a == 0 and self.d.is_integer and self.c.is_integer

    @property
    def is_d(self) -> bool:
        """True if the ring element is element of :math:`\\mathbb{D}`."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @property
    def is_integer(self) -> bool:
        """True if the ring element is an integer."""
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d.is_integer

    @classmethod
    def from_ring(cls, nb: int | complex | Ring) -> Domega:
        """
        Convert a ring element to a Domega object. The conversion must be possible.

        Args:
            nb (int | complex | Ring): Ring element to convert to Domega.

        Returns:
            Domega: Domega object converted from the ring element.

        Raises:
            ValueError: If the conversion is not possible.
        """
        if isinstance(nb, Domega):
            return nb

        elif isinstance(nb, Zomega):
            return cls((nb.a, 0), (nb.b, 0), (nb.c, 0), (nb.d, 0))

        elif isinstance(nb, Dsqrt2):
            return cls(-nb.b, (0, 0), nb.b, nb.a)

        elif isinstance(nb, Zsqrt2):
            return cls((-nb.b, 0), (0, 0), (nb.b, 0), (nb.a, 0))

        elif isinstance(nb, D):
            return cls((0, 0), (0, 0), (0, 0), nb)

        elif isinstance(nb, (int, np.integer)):
            return cls((0, 0), (0, 0), (0, 0), (nb, 0))

        elif isinstance(nb, complex):
            if nb.real.is_integer() and nb.imag.is_integer():
                return cls((0, 0), (int(nb.imag), 0), (0, 0), (int(nb.real), 0))

        raise ValueError(f"Cannot convert {type(nb).__name__} to Domega.")

    def real(self) -> float:
        """Return the real part of the ring element.

        Returns:
            float: Real part of the ring element in float representation.
        """
        sqrt_value = float(self.c - self.a) / SQRT2
        d = float(self.d)

        if math.isclose(d, -sqrt_value, rel_tol=1e-5):
            # Maintain high precision if the two values are close to each other.
            getcontext().prec = 50
            return float(
                self.d.num / (Decimal(2) ** self.d.denom)
                + (self.c - self.a).num
                / (Decimal(2) ** (self.c - self.a).denom)
                / Decimal(2).sqrt()
            )

        return d + sqrt_value

    def mp_real(self) -> float:
        """Return the real part of the ring element in `mpfloat` representation."""
        return self.d.mpfloat() + (self.c - self.a).mpfloat() / mp.sqrt(2)

    def imag(self) -> float:
        """Return the imaginary part of the ring element.

        Returns:
            float : Imaginary part of the ring element in float representation.
        """
        sqrt_value = float(self.c + self.a) / SQRT2
        b = float(self.b)

        if math.isclose(b, -sqrt_value, rel_tol=1e-5):
            # Maintain high precision if the two values are close to each other.
            getcontext().prec = 50
            return float(
                self.b.num / (Decimal(2) ** self.b.denom)
                + (self.c + self.a).num
                / (Decimal(2) ** (self.c + self.a).denom)
                / Decimal(2).sqrt()
            )

        return b + sqrt_value

    def mp_imag(self) -> float:
        """Return the imaginary part of the ring element in `mpfloat` representation."""
        return self.b.mpfloat() + (self.c + self.a).mpfloat() / mp.sqrt(2)

    def __complex__(self) -> complex:
        """Define the complex value of the class."""
        return self.real() + 1j * self.imag()

    def mpcomplex(self) -> complex:
        """Define the `mpcomplex` value of the Domega class."""
        return mp.mpc(self.mp_real(), self.mp_imag())

    def sde(self) -> int | float:
        """
        Return the smallest denominator exponent (sde) of base :math:`\\sqrt{2}` of the ring element.

        The sde of the ring element :math:`d \\in \\mathbb{D}[\\omega]` is the smallest integer value k such that :math:`d * (\\sqrt{2})^k \\in \\mathbb{Z}[\\omega]`.
        """
        sde: int = 0

        # If at least one of the coefficient is not an integer, multiply all coefficients by 2^k_max to make all of them integers.
        if not self.is_zomega:
            k_max: int = max([coeff.denom for coeff in self])
            coeffs: list[int] = [coeff.num * 2 ** (k_max - coeff.denom) for coeff in self]
            sde += 2 * k_max
        else:
            # If all coefficients are integers.
            coeffs = [coeff.num for coeff in self]

            # If all coefficients are zero, return negative infinity.
            if not any(coeffs):
                return -math.inf

            # If all coefficients are even, we can divide by 2 and remain in Z[ω].
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
        """Compute the complex conjugate of the ring element.

        Returns:
            Domega: Complex conjugate of the ring element.
        """
        return Domega(a=-self.c, b=-self.b, c=-self.a, d=self.d)

    def sqrt2_conjugate(self) -> Domega:
        """Compute the :math:`\\sqrt{2}`-conjugate of the ring element.

        Returns:
            Domega: :math:`\\sqrt{2}`-conjugate of the ring element.
        """
        return Domega(a=-self.a, b=self.b, c=-self.c, d=self.d)

    def __repr__(self) -> str:
        """Define the string representation of the class."""
        sign: Callable[[D], str] = lambda coeff: " + " if coeff.num >= 0 else " - "
        value: Callable[[D], str] = lambda coeff: str(coeff) if coeff.num >= 0 else str(-coeff)

        return (
            str(self.a)
            + "\u03c93"
            + sign(self.b)
            + value(self.b)
            + "\u03c92"
            + sign(self.c)
            + value(self.c)
            + "\u03c9"
            + sign(self.d)
            + value(self.d)
        )

    def __getitem__(self, i: int | slice) -> D | list[D]:
        """Return the coefficients of the ring element from their index."""
        return [self.a, self.b, self.c, self.d][i]

    def __iter__(self) -> Iterator[D]:
        """Allow iteration through the class coefficients."""
        return iter([self.a, self.b, self.c, self.d])

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of Domega classes."""
        if isinstance(nb, Domega):
            return self.a == nb.a and self.b == nb.b and self.c == nb.c and self.d == nb.d

        elif isinstance(nb, (D, int, np.integer)):
            return self.is_d and self.d == nb

        return False

    def __neg__(self) -> Domega:
        """Define the negation of the ring element."""
        return Domega(-self.a, -self.b, -self.c, -self.d)

    def __add__(self, nb: int | D | Domega) -> Domega:
        """Define the summation operation for the Domega class."""
        if isinstance(nb, Domega):
            return Domega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d)

        elif isinstance(nb, (D, int, np.integer)):
            return Domega(self.a, self.b, self.c, self.d + nb)

        raise TypeError(f"Summation is not defined between Domega and {type(nb).__name__}.")

    def __radd__(self, nb: int | D | Domega) -> Domega:
        """Define the right summation with the Domega class."""
        return self.__add__(nb)

    def __iadd__(self, nb: int | D | Domega) -> Domega:
        """Define the in-place summation operation for the Domega class."""
        return self.__add__(nb)

    def __sub__(self, nb: int | D | Domega) -> Domega:
        """Define the subtraction operation for the Domega class."""
        if isinstance(nb, (Domega, D, int, np.integer)):
            return self.__add__(-nb)

        raise TypeError(f"Subtraction is not defined between Domega and {type(nb).__name__}.")

    def __rsub__(self, nb: int | D | Domega) -> Domega:
        """Define the right subtraction with the Domega class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: int | D | Domega) -> Domega:
        """Define the in-place subtraction for the Domega class."""
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

        raise TypeError(f"Product is not defined between Domega and {type(nb).__name__}.")

    def __rmul__(self, nb: int | D | Domega) -> Domega:
        """Define the right multiplication with the Domega class."""
        return self.__mul__(nb)

    def __imul__(self, nb: int | D | Domega) -> Domega:
        """Define the in-place multiplication for the Domega class."""
        return self.__mul__(nb)

    def __pow__(self, power: int) -> Domega:
        """Define the power operation for the Domega class. Exponent must be a positive integer."""
        # Check the input
        if not isinstance(power, (int, np.integer)):
            raise TypeError(f"Exponent must be an integer, but received {type(power).__name__}.")

        if power < 0:
            raise ValueError(f"Expected exponent to be a positive integer, but got {power}.")

        # Compute the power
        nth_power = self
        result = Domega((0, 0), (0, 0), (0, 0), (1, 0))

        while power:
            if power & 1:
                result *= nth_power
            nth_power *= nth_power
            power >>= 1

        return result

    def __ipow__(self, nb: int) -> Domega:
        """Define the in-place power operation of the Domega class."""
        return self.__pow__(nb)


# General type for all ring instances.
Ring = Union[D, Zsqrt2, Dsqrt2, Zomega, Domega]

# LAMBDA = 1 + √2 is used to scale 1D grid problems.
LAMBDA: Zsqrt2 = Zsqrt2(1, 1)

# INVERSE_LAMBDA = -1 + √2 is the inverse of LAMBDA. It is used to scale 1D grid problem.
INVERSE_LAMBDA: Zsqrt2 = -LAMBDA.sqrt2_conjugate()
