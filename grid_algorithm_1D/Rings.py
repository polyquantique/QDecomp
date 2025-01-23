# Copyright 2022-2023 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
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
Symbolic Computation with Ring Elements.

The Ring module provides tools for symbolic computations with elements of various mathematical rings.
Rings are used in many algorithms for the approximation of z-rotation gates into Clifford+T unitaries.
For more information see 
Neil J. Ross and Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations, 
https://arxiv.org/pdf/1403.2975.

Classes:
    - D: Ring of dyadic fractions \u2145.
    - Domega: Ring of cyclotomic dyadic fractions of degree 8 \u2145[\u03C9].
    - Zomega: Ring of cyclotomic integers of degree 8 \u2124[\u03C9].
    - Dsqrt2: Ring of quadratic dyadic fractions with radicand 2 \u2145[\u221a2].
    - Zsqrt2: Ring of quadratic integers with radicand 2 \u2124[\u221a2].

Function:
    - output_type(): Determine the output type when doing operations with elements from different rings. 
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext
from numbers import Complex, Integral, Real
from typing import Any, Callable, Iterator, Union


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
            ValueError: If the denominator is not positive.
        """
        if not isinstance(num, Integral) or not isinstance(denom, Integral):
            raise TypeError(
                f"Class arguments must be of type int but received {type(num).__name__ if not isinstance(num, Integral) else type(denom).__name__}."
            )
        elif denom < 0:
            raise ValueError(f"Denominator exponent must be positive but got {denom}.")
        self.__num: int = num
        self.__denom: int = denom

        # Reduce the fraction
        if not self.is_integer:
            self.__reduce()

    @property
    def num(self) -> int:
        """Numerator of the dyadic fraction."""
        return self.__num

    @property
    def denom(self) -> int:
        """Denominator exponent of the dyadic fraction."""
        return self.__denom

    @property
    def is_integer(self) -> bool:
        """Return True if the number is an integer."""
        return self.denom == 0

    def __reduce(self) -> None:
        """Reduce the fraction if the numerator is even."""
        if self.num == 0:
            self.__denom = 0
            return
        while (self.num & 1) == 0 and self.denom > 0:
            self.__num >>= 1
            self.__denom -= 1

    def __neg__(self) -> D:
        """Define the negation of the D class."""
        return D(-self.num, self.denom)

    def __abs__(self) -> D:
        """Define the absolute value of the D class."""
        return D(abs(self.num), self.denom)

    def __repr__(self) -> str:
        """Define the string representation of the D class."""
        if self.denom == 0:
            return str(self.num)
        elif self.denom == 1:
            return f"{self.num}/2"
        return f"{self.num}/2^{self.denom}"

    def __float__(self) -> float:
        """Define the float value of the D class."""
        return self.num / 2**self.denom

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of the D class."""
        if isinstance(nb, D):
            return self.num == nb.num and self.denom == nb.denom
        elif isinstance(nb, Integral):
            return self.denom == 0 and self.num == nb
        elif isinstance(nb, Domega):
            return nb.__eq__(self)
        return math.isclose(float(self), nb)

    def __lt__(self, nb: Any) -> bool:
        """Define the < operation of the D class."""
        return float(self) < nb

    def __gt__(self, nb: Any) -> bool:
        """Define the < operation of the D class."""
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
        elif isinstance(nb, Integral):
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
        if isinstance(nb, (D, Integral)):
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
        elif isinstance(nb, Integral):
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
        if not isinstance(n, Integral):
            raise TypeError(f"Expected power to be of type int, but got {type(n).__name__}.")
        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n}.")
        return D(self.num**n, n * self.denom)

    def __round__(self) -> int:
        """Define the round operation for the D class."""
        return round(float(self))

    def __floor__(self) -> int:
        """Define the floor operation for the D class."""
        return math.floor(float(self))

    def __ceil__(self) -> int:
        """Define the floor operation for the D class."""
        return math.ceil(float(self))


class Domega:
    """Class to do symbolic computation with elements of the ring \u2145[\u03C9].

    The ring element has the form a\u03C9^3 + b\u03C9^2 + c\u03C9 + d, where \u03C9 = (1 + i)/\u221a2 is the eight root of unity.
    The coefficients a, b, c, d are dyadic fractions of the form m / 2^n, where m is an integer and n is a positive integer.
    The coefficients will be automatically reduced when the class is initialized.

    The ring element can also be expressed as \u03B1 + i*\u03B2, where i = \u221a-1, and \u03B1 and \u03B2 are numbers in te ring \u2145[\u221a2]
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
                if len(input) != 2 or any([not isinstance(value, Integral) for value in input]):
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

        self.__a: D = a if isinstance(a, D) else D(a[0], a[1])
        self.__b: D = b if isinstance(b, D) else D(b[0], b[1])
        self.__c: D = c if isinstance(c, D) else D(c[0], c[1])
        self.__d: D = d if isinstance(d, D) else D(d[0], d[1])

    @property
    def a(self) -> D:
        """\u03C9^3 coefficient of the ring element."""
        return self.__a

    @property
    def b(self) -> D:
        """\u03C9^2 coefficient of the ring element."""
        return self.__b

    @property
    def c(self) -> D:
        """\u03C9^1 coefficient of the ring element."""
        return self.__c

    @property
    def d(self) -> D:
        """\u03C9^0 coefficient of the ring element."""
        return self.__d

    @property
    def _is_Zomega(self) -> bool:
        """True if the ring element is element of \u2124[\u03C9]."""
        return all([coeff.is_integer for coeff in self])

    @property
    def _is_Dsqrt2(self) -> bool:
        """True if the ring element is element of \u2145[\u221a2]."""
        return self.b == 0 and self.c + self.a == 0

    @property
    def _is_Zsqrt2(self) -> bool:
        """True if the ring element is element of \u2124[\u221a2]."""
        return (
            self.b == 0
            and self.c + self.a == 0
            and self.d.is_integer
            and ((self.c - self.a) * D(1, 1)).is_integer
        )

    @property
    def _is_D(self) -> bool:
        """True if the ring element is element of \u2145."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @property
    def _is_integer(self) -> bool:
        """True if the ring element is an integer."""
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d.is_integer

    def convert(self, target: type | str) -> Ring:
        """Convert the ring element into another ring.

        The conversion must be possible, i.e. the ring element is part of the targeted ring.

        Args:
            target (type | str): Class of the targeted ring.

        Returns:
            Ring: Number expressed in the targeted ring.

        Raises:
            TypeError: If the target is not a type or a string.
            TypeError: If the ring element is not part of the targeted ring.
            ValueError: If the target is not valid.
        """
        if not isinstance(target, (type, str)):
            raise TypeError(
                f"Target must be a type or a string, but received {type(target).__name__}."
            )

        if target in (Domega, "Domega"):
            return Domega(self.a, self.b, self.c, self.d)
        elif target in (Zomega, "Zomega"):
            if self._is_Zomega:
                return Zomega(a=self.a.num, b=self.b.num, c=self.c.num, d=self.d.num)
            raise TypeError(
                f"Could not convert the ring element from {type(self).__name__} to Zomega.\n{str(self)} cannot be written as a\u03C9^3 + b\u03C9^2 + c\u03C9 + d, where a, b, c and d are integers."
            )
        elif target in (Dsqrt2, "Dsqrt2"):
            if self._is_Dsqrt2:
                return Dsqrt2(self.d, (self.c - self.a) * D(1, 1))
            raise TypeError(
                f"Could not convert the ring element from {type(self).__name__} to Dsqrt2.\n{str(self)} cannot be written as a + b\u221a2, where a and b are in \u2145."
            )
        elif target in (Zsqrt2, "Zsqrt2"):
            if self._is_Zsqrt2:
                return Zsqrt2(self.d.num, ((self.c - self.a) * D(1, 1)).num)
            raise TypeError(
                f"Could not convert the ring element from {type(self).__name__} to Zsqrt2.\n{str(self)} cannot be written as a + b\u221a2, where a and b are integers."
            )
        elif target in (D, "D"):
            if self._is_D:
                return self.d
            raise TypeError(f"Could not convert the ring element from {type(self).__name__} to D.")
        raise TypeError(
            f"{target.__name__ if isinstance(target, type) else target} is not a valid target."
        )

    def real(self) -> float:
        """Return the real part of the ring element.

        Returns:
            float: Real part of the ring element in float representation.
        """
        x1 = float(self.d)
        x2 = float(self.c - self.a) / math.sqrt(2)
        if math.isclose(x1, -x2, rel_tol=1e-4):
            getcontext().prec = 50
            return float(
                Decimal(self.d.num) / Decimal(2) ** Decimal(self.d.denom)
                + Decimal((self.c - self.a).num)
                / Decimal(2) ** Decimal((self.c - self.a).denom)
                / Decimal(2).sqrt()
            )
        return x1 + x2

    def imag(self) -> float:
        """Return the imaginary part of the ring element.

        Returns:
            float : Imaginary part of the ring element in float representation.
        """
        x1 = float(self.b)
        x2 = float(self.c + self.a) / math.sqrt(2)
        if math.isclose(x1, -x2, rel_tol=1e-4):
            getcontext().prec = 50
            return float(
                Decimal(self.b.num) / Decimal(2) ** Decimal(self.b.denom)
                + Decimal((self.c + self.a).num)
                / Decimal(2) ** Decimal((self.c + self.a).denom)
                / Decimal(2).sqrt()
            )
        return x1 + x2

    def sde(self) -> int | float:
        """Return the smallest denominator exponent (sde) of base \u221a2 of the ring element.

        The sde of the ring element d \u2208 \u2145[\u03C9] is the smallest integer value k such that d * (\u221a2)^k \u2208 \u2124[\u03C9].
        """
        sde: int = 0
        if any([coeff.denom != 0 for coeff in self]):
            k_max: int = max([coeff.denom for coeff in self])
            coeffs: list[int] = [coeff.num * 2 ** (k_max - coeff.denom) for coeff in self]
            sde += 2 * k_max
        else:
            coeffs = [coeff.num for coeff in self]
            if not any(coeffs):
                return -math.inf
            while all([coeff % 2 == 0 for coeff in coeffs]):
                coeffs = [coeff // 2 for coeff in coeffs]
                sde -= 2
        while coeffs[0] % 2 == coeffs[2] % 2 and coeffs[1] % 2 == coeffs[3] % 2:
            alpha: int = (coeffs[1] - coeffs[3]) // 2
            beta: int = (coeffs[2] + coeffs[0]) // 2
            gamma: int = (coeffs[1] + coeffs[3]) // 2
            delta: int = (coeffs[2] - coeffs[0]) // 2
            coeffs = [alpha, beta, gamma, delta]
            sde -= 1
        return sde

    def complex_conjugate(self) -> Ring:
        """Compute complex conjugate of the ring element.

        Returns:
            Ring: Complex conjugate of the ring element.
        """
        return Domega(a=-self.c, b=-self.b, c=-self.a, d=self.d).convert(type(self))

    def sqrt2_conjugate(self) -> Ring:
        """Compute the \u221a2 conjugate of the ring element.

        Returns:
            Ring: \u221a2 conjugate of the ring element.
        """
        return Domega(a=-self.a, b=self.b, c=-self.c, d=self.d).convert(type(self))

    def __repr__(self) -> str:
        """Define the string representation of the class."""
        sign: Callable[[D], str] = lambda coeff: "+" if coeff >= 0 else "-"
        value: Callable[[D], str] = lambda coeff: str(coeff) if coeff >= 0 else str(-coeff)
        omega: Callable[[int], str] = lambda index: "\u03C9" + str(index) if index > 0 else ""
        if self._is_D:
            return str(self.d)
        else:
            output: str = ""
            for index, coeff in enumerate(self):
                if coeff.num != 0:
                    if coeff < 0:
                        if coeff == -1 and index != 3:
                            output += "- " + omega(3 - index) + " "
                        else:
                            output += "- " + value(coeff) + omega(3 - index) + " "
                    elif coeff > 0 and len(output) == 0:
                        if coeff == 1 and index != 3:
                            output += omega(3 - index) + " "
                        else:
                            output += str(coeff) + omega(3 - index) + " "
                    else:
                        if coeff == 1 and index != 3:
                            output += sign(coeff) + " " + omega(3 - index) + " "
                        else:
                            output += sign(coeff) + " " + value(coeff) + omega(3 - index) + " "
        return output.rstrip()

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
        elif isinstance(nb, (D, Integral)):
            return self._is_D and self.d == nb
        elif isinstance(nb, (Real, Complex)):
            return math.isclose(self.real(), complex(nb).real) and math.isclose(
                self.imag(), complex(nb).imag
            )
        raise TypeError(
            f"Comparison between {self.__class__.__name__} and {type(nb).__name__} is not possible."
        )

    def __neg__(self) -> Ring:
        """Define the negation of the ring element."""
        return Domega(-self.a, -self.b, -self.c, -self.d).convert(type(self))

    def __complex__(self) -> complex:
        """Define the complex representation of the class."""
        return self.real() + 1.0j * self.imag()

    def __add__(self, nb: Any) -> Ring:
        """Define the summation operation for the ring elements."""
        if isinstance(nb, (D, Integral)):
            return Domega(self.a, self.b, self.c, self.d + nb).convert(
                _output_type(type(self), type(nb))
            )
        elif isinstance(nb, Domega):
            return Domega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d).convert(
                _output_type(type(self), type(nb))
            )
        raise TypeError(
            f"Summation operation is not defined between {self.__class__.__name__} and {type(nb).__name__}."
        )

    def __radd__(self, nb: Any) -> Ring:
        """Define the right summation of integers and D objects with the Domega class."""
        return self.__add__(nb)

    def __iadd__(self, nb: Any) -> Ring:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: Any) -> Ring:
        """Define the subtraction operation for the ring element."""
        if isinstance(nb, (Integral, D, Domega)):
            return self.__add__(-nb)
        raise TypeError(
            f"Subtraction operation is not defined between {self.__class__.__name__} and {type(nb).__name__}."
        )

    def __rsub__(self, nb: Any) -> Ring:
        """Define the right subtraction of integers and D objects with the Domega class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: Any) -> Ring:
        """Define the in-place subtraction for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Any) -> Ring:
        """Define the multiplication operation for the Domega class."""
        if isinstance(nb, (D, Integral)):
            return Domega(self.a * nb, self.b * nb, self.c * nb, self.d * nb).convert(
                _output_type(type(self), type(nb))
            )
        elif isinstance(nb, Domega):
            a: D = (self.a * nb.d) + (self.b * nb.c) + (self.c * nb.b) + (self.d * nb.a)
            b: D = -(self.a * nb.a) + (self.b * nb.d) + (self.c * nb.c) + (self.d * nb.b)
            c: D = -(self.a * nb.b) + -(self.b * nb.a) + (self.c * nb.d) + (self.d * nb.c)
            d: D = -(self.a * nb.c) + -(self.b * nb.b) + -(self.c * nb.a) + (self.d * nb.d)
            return Domega(a, b, c, d).convert(_output_type(type(self), type(nb)))
        try:
            return nb.__rmul__(self)
        except AttributeError:
            raise TypeError(
                f"Product operation is not defined between {self.__class__.__name__} and {type(nb).__name__}."
            )

    def __rmul__(self, nb: Any) -> Ring:
        """Define the right multiplication of integers and D objects with the Domega class."""
        return self.__mul__(nb)

    def __imul__(self, nb: Any) -> Ring:
        """Define the in-place multiplication for the class."""
        return self.__mul__(nb)

    def __pow__(self, power: int) -> Ring:
        """Define the power operation for the Domega class.

        Exponent must be positive integers. Uses the multinomial theorem.
        """
        if not isinstance(power, Integral):
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
        return Domega(coeff[0], coeff[1], coeff[2], coeff[3]).convert(type(self))

    def __ipow__(self, nb: int):
        """Define the in-place power operation of the ring element."""
        return self.__pow__(nb)


class Zomega(Domega):
    """Class to do symbolic computation with elements of the ring of cyclotomic integers of degree 8 \u2124[\u03C9].

    The ring element has the form a\u03C9^3 + b\u03C9^2 + c\u03C9 + d, where \u03C9 = (1 + i)/\u221a2 is the eight root of unity.
    The coefficients a, b, c, d are integers.

    The ring element can also be expressed as \u03B1 + i*\u03B2, where i = \u221a-1, and \u03B1 and \u03B2 are numbers in te ring \u2145[\u221a2]
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
        for arg in (a, b, c, d):
            if not isinstance(arg, Integral):
                raise TypeError(
                    f"Class arguments must be integers, but got {arg} of type {type(arg).__name__}"
                )
        super().__init__((a, 0), (b, 0), (c, 0), (d, 0))

    def __getitem__(self, i):
        return (self.a.num, self.b.num, self.c.num, self.d.num)[i]


class Dsqrt2(Domega):
    """Class to do symbolic computation with element in the ring of quadratic dyadic fractions \u2145[\u221a2].

    The ring element has the form p + q\u221a2, where p, q are dyadic fractions of the form m / 2^n,
    where m is an integer and n is a positive integer.
    The coefficients will be automatically reduced when the class is initialized.

    Attributes:
        p (D): Rational coefficient of the ring element.
        q (D): \u221a2 coefficient of the ring element.
    """

    def __init__(self, p: tuple[int, int] | D, q: tuple[int, int] | D) -> None:
        """Initialize the Dsqrt2 class.

        Args:
            p (tuple[int, int] | D): Rational coefficient of the ring element.
            q (tuple[int, int] | D): \u221a2 coefficient of the ring element.

        Raises:
            TypeError: If the class arguments are not 2-tuples of integers (num, denom) or D objects.
            ValueError: If the denominator exponent is negative.
        """
        for input in (p, q):
            if isinstance(input, tuple):
                if len(input) != 2 or any([not isinstance(value, Integral) for value in input]):
                    raise TypeError(
                        f"Tuples must take two integer values (num, denom) but received {input}."
                    )
                elif input[1] < 0:
                    raise ValueError(f"Denominator value must be positive but got {input[1]} < 0.")
            elif not isinstance(input, D):
                raise TypeError(
                    f"Class arguments must be of type tuple[int, int] or D objects but received {type(input).__name__}."
                )
        self.__p: D = p if isinstance(p, D) else D(p[0], p[1])
        self.__q: D = q if isinstance(q, D) else D(q[0], q[1])
        super().__init__(-self.__q, D(0, 0), self.__q, self.__p)

    @property
    def p(self) -> D:
        """Rational coefficient of the ring element."""
        return self.__p

    @property
    def q(self) -> D:
        """\u221a2 coefficient of the ring element."""
        return self.__q

    def __repr__(self) -> str:
        """Define the string representation of the ring element."""
        repr: str = ""
        if self == 0:
            return str(0)
        elif self.p != 0:
            repr += str(self.p) + " "
        if self.q != 0:
            repr += f"{"- " if self.q < 0 else "+ "}"
            repr += f"{str(abs(self.q)) if abs(self.q) != 1 else ""}"
            repr += "\u221a2"
        return repr

    def __float__(self) -> float:
        """Define the float representation of the class."""
        return self.real()

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


class Zsqrt2(Domega):
    """A simple class to do symbolic computation with elements of the ring \u2124[\u221a2].

    The ring element has the form p + q\u2124[\u221a2], where p and q are integers.

    Attributes:
        p (int): Integer coefficient of the ring element.
        q (int): \u2124[\u221a2] coefficient of the ring element.
    """

    def __init__(self, p: int, q: int) -> None:
        """Initialize the ring element.

        Args:
            p (int): Integer coefficient of the ring element.
            q (int): \u2124[\u221a2] coefficient of the ring element.

        Raises:
            TypeError: If p or q are not integers.
        """
        for input in (p, q):
            if not isinstance(input, Integral):
                raise TypeError(
                    f"Expected class inputs to be of type int, but got {type(input).__name__}."
                )
        self._p: int = p
        self._q: int = q
        super().__init__(a=(-q, 0), b=(0, 0), c=(q, 0), d=(p, 0))

    @property
    def p(self) -> int:
        """Integer coefficient of the ring element."""
        return self._p

    @property
    def q(self) -> int:
        """\u2124[\u221a2] coefficient of the ring element."""
        return self._q

    def __repr__(self) -> str:
        """Define the string representation of the ring element."""
        return Dsqrt2.__repr__(self)

    def __float__(self) -> float:
        """Define the float representation of the class."""
        return self.real()

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


def _output_type(*types: type) -> type:
    """Return the output type of class operations.

    Args:
        *types (type): List of Python types.

    Returns:
        type: Output type of the operation.
    """
    if Domega in types:
        return Domega
    elif Zomega in types:
        if Dsqrt2 in types or D in types:
            return Domega
        return Zomega
    elif Dsqrt2 in types:
        return Dsqrt2
    elif Zsqrt2 in types:
        if D in types:
            return Dsqrt2
        return Zsqrt2
    raise ValueError(
        f"Conversion between {', '.join([t.__name__ for t in types])} is not supported."
    )


Ring = Union[D, Domega, Zomega, Dsqrt2, Zsqrt2]

# lambda = 1 + \u221A2 is used to scale 1D grid problems.
lamb: Zsqrt2 = Zsqrt2(1, 1)

# inv_lambda = -1 + \u221A2 is the inverse of lambda. It is used to scale 1D grid problem.
inv_lamb: Zsqrt2 = -lamb.sqrt2_conjugate()
