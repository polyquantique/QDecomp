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
The rings are used in many algorithms for the approximation of z-rotations gates into Clifford+T unitaries.
For more information see 
Neil J. Ross and Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations, 
https://arxiv.org/pdf/1403.2975

Classes:
    - D: Ring of dyadic fractions.
    - Domega: Ring of cyclotomic dyadic fractions of degree 8 D[ω]
    - Zomega: Ring of cyclotomic integers of degree 8 Z[ω]
    - Dsqrt2: Ring of quadratic dyadic fractions with radicand 2 D[√2].
    - Zsqrt2: Ring of quadratic integers with radicand 2 Z[ω].

Function:
    - output_type(): Determine the output type when doing operations with elements from different rings. 
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext
from numbers import Integral
from typing import Any, Callable, Iterator

import numpy as np


class D:
    """Class to do symbolic computation with elements of the ring of dyadic fractions D.

    The ring element has the form a / b^k, where a is an integer and k is a positive integer.
    The fraction will automatically be reduced in the case of a even numerator.

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
            TypeError: If num or denom are not integers.
            ValueError: If denom is not positive.
        """
        if not isinstance(num, Integral) or not isinstance(denom, Integral):
            raise TypeError(
                f"Class arguments must be of type int but received {type(num).__name__ if not isinstance(num, Integral) else type(denom).__name__}."
            )
        elif denom < 0:
            raise ValueError(f"denom value must be positive but got {denom}")
        self._num: int = num
        self._denom: int = denom
        if not self.is_integer:
            self.__reduce()

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
        """Return True if the exponent in the denominator is 0, i.e. if the number is an integer."""
        return self.denom == 0

    def __reduce(self) -> None:
        """Reduce the fraction if the numerator is even."""
        if self.num == 0:
            self._denom = 0
        while self.num % 2 == 0 and self.num != 0 and self.denom > 0:
            self._num //= 2
            self._denom -= 1

    def __neg__(self) -> D:
        """Define the negation of the class D."""
        return D(-self.num, self.denom)

    def __repr__(self) -> str:
        """Define the string representation of the class D."""
        if self.denom == 0:
            return str(self.num)
        elif self.denom == 1:
            return f"{self.num}/2"
        else:
            return f"{self.num}/2^{self.denom}"

    def __float__(self) -> float:
        """Define the float value of the class."""
        return self.num / 2**self.denom

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of the D class."""
        return math.isclose(float(self), nb)

    def __neq__(self, nb: Any) -> bool:
        """Define the != operation of the D class."""
        return not math.isclose(float(self), nb)

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

    def __add__(self, nb: Any) -> D | Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the summation operation for the D class."""
        if isinstance(nb, D):
            if self._denom >= nb._denom:
                num: int = self.num + nb.num * 2 ** (self.denom - nb.denom)
                return D(num, self.denom)
            else:
                num = nb.num + self.num * 2 ** (nb.denom - self.denom)
                return D(num, nb.denom)
        elif isinstance(nb, Integral):
            return D(self.num + nb * 2**self.denom, self.denom)
        elif issubclass(type(nb), Domega):
            return nb.__add__(self)
        raise TypeError(f"Summation operation is not defined between D and {type(nb).__name__}")

    def __radd__(self, nb: Any) -> D | Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the right summation of integers with the D class."""
        return self.__add__(nb)

    def __iadd__(self, nb: Any) -> D | Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: Any) -> D | Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the subtraction operation for the D class."""
        if not isinstance(nb, (D, Integral)) and not issubclass(type(nb), Domega):
            raise TypeError(
                f"Subtraction operation is not defined between D and {type(nb).__name__}."
            )
        return self.__add__(-nb)

    def __rsub__(self, nb: Any) -> D | Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the right subtraction of integers with the D class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: Any) -> D | Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the in-place subtraction for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Any) -> D | Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the product operation for the D class."""
        if isinstance(nb, D):
            return D(self.num * nb.num, self.denom + nb.denom)
        elif isinstance(nb, Integral):
            return D(self.num * nb, self.denom)
        elif issubclass(type(nb), Domega):
            return nb.__mul__(self)
        raise TypeError(f"Product operation is not defined between D and {type(nb).__name__}.")

    def __rmul__(self, nb: Any) -> D | Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the right multiplication of integers with the D class."""
        return self.__mul__(nb)

    def __imul__(self, nb: Any) -> D | Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the inplace-multiplication for the class."""
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
        """Define the round operation on the D class."""
        return round(float(self))

    def __floor__(self) -> int:
        """Define the floor operation on the D class."""
        return math.floor(float(self))

    def __ceil__(self) -> int:
        """Define the floor operation on the D class."""
        return math.ceil(float(self))


class Domega:
    """Class to do symbolic computation with elements of the ring D[ω].

    The ring element has the form aω^3 + bω^2 + cω + d, where ω = (1 + i)/√2 is the eight root of unity.
    The coefficients a, b, c, d are dyadic fractions of the form m / 2^n, where m is an integer and n is a positive integer.
    The coefficients will be automatically reduced when the class is initialized.

    The ring element can also be expressed as alpha + i*beta, where i = √-1, and alpha and beta are numbers in te ring D[√2].
    alpha = d + (c-a)/2 √2 and beta = b + (c+a)/2 √2.

    Attributes:
        a (D): ω^3 coefficient of the ring element.
        b (D): ω^2 coefficient of the ring element.
        c (D): ω^1 coefficient of the ring element.
        d (D): ω^0 coefficient of the ring element.
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
            a (tuple[int, int] | D): ω^3 coefficient of the ring element.
            b (tuple[int, int] | D): ω^2 coefficient of the ring element.
            c (tuple[int, int] | D): ω^1 coefficient of the ring element.
            d (tuple[int, int] | D): ω^0 coefficient of the ring element.

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
                    raise ValueError(f"Denominator value must be positive but got {input[1]} < 0.")
            elif not isinstance(input, D):
                raise TypeError(
                    f"Class arguments must be of type tuple[int, int] or D objects but received {type(input).__name__}."
                )

        self._a: D = a if isinstance(a, D) else D(a[0], a[1])
        self._b: D = b if isinstance(b, D) else D(b[0], b[1])
        self._c: D = c if isinstance(c, D) else D(c[0], c[1])
        self._d: D = d if isinstance(d, D) else D(d[0], d[1])

    @property
    def a(self):
        """ω^3 coefficient of the ring element."""
        return self._a

    @property
    def b(self):
        """ω^2 coefficient of the ring element."""
        return self._b

    @property
    def c(self):
        """ω^1 coefficient of the ring element."""
        return self._c

    @property
    def d(self):
        """ω^0 coefficient of the ring element."""
        return self._d

    @property
    def _is_Zomega(self) -> bool:
        """True if the ring element is also in Z[ω]."""
        return all([coeff.is_integer for coeff in self])

    @property
    def _is_Dsqrt2(self) -> bool:
        """True if the ring element is also in D[√2]."""
        return self.b == 0 and self.c + self.a == 0

    @property
    def _is_Zsqrt2(self) -> bool:
        """True if the ring element is also in Z[√2]."""
        return (
            self.b == 0
            and self.c + self.a == 0
            and self.d.is_integer
            and ((self.c - self.a) * D(1, 1)).is_integer
        )

    @property
    def _is_D(self) -> bool:
        """True if the ring element is also in D."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @property
    def _is_integer(self) -> bool:
        """True if the ring element is an integer."""
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d.is_integer

    def convert(self, target: type) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Convert the ring element into another ring.

        The conversion must be possible, i.e. the ring element is part of the targeted ring.

        Args:
            target (type): Class of the targeted ring element.

        Returns:
            Zsqrt2 | Dsqrt2 | Zomega | Domega: Number expressed as the targeted ring element.

        Raises:
            TypeError: If the ring element is not part of the targeted ring.
        """
        if target is Domega:
            return Domega(self.a, self.b, self.c, self.d)
        elif target is Zomega:
            if self._is_Zomega:
                return Zomega(a=self.a.num, b=self.b.num, c=self.c.num, d=self.d.num)
            else:
                raise TypeError(
                    f"Could not convert the ring element from {type(self)} to Zomega.\n{str(self)} cannot be written as aω^3 + bω^2 + cω + d, where a, b, c and d are integers."
                )
        elif target is Dsqrt2:
            if self._is_Dsqrt2:
                return Dsqrt2(self.d, (self.c - self.a) * D(1, 1))
            else:
                raise TypeError(
                    f"Could not convert the ring element from {type(self)} to Dsqrt2.\n{str(self)} cannot be written as a + b√2, where a and b are in D."
                )
        elif target is Zsqrt2:
            if self._is_Zsqrt2:
                return Zsqrt2(self.d.num, ((self.c - self.a) * D(1, 1)).num)
            else:
                raise TypeError(
                    f"Could not convert the ring element from {type(self)} to Zsqrt2.\n{str(self)} cannot be written as a + b√2, where a and b are integers."
                )
        elif target is D:
            if self._is_D:
                return self.d
            else:
                raise TypeError(
                    f"Could not convert the ring element from {type(self)} to D.\n{str(self)} cannot be written as a in D."
                )
        elif target is int:
            if self._is_integer:
                return self.d.num
            else:
                raise TypeError(
                    f"Could not convert the ring element from {type(self)} to Z.\n{str(self)} cannot be written as an integer."
                )
        else:
            raise TypeError(f"{target.__name__} is not a valid target.")

    def real(self, exact: bool = False) -> float | Dsqrt2:
        """Return the real part of the ring element.

        Args:
            exact (bool): If set to true, return the real part as a Dsqrt2 object. Default == False.

        Returns:
            float | Dsqrt2: Real part of the ring element.
        """
        if exact:
            return Dsqrt2(self.d, (self.c - self.a) * D(1, 1))
        else:
            x1 = float(self.d)
            x2 = float(self.c - self.a) / math.sqrt(2)
            if math.isclose(x1, -x2, rel_tol=1e-4):
                getcontext().prec = 50
                return float(
                    Decimal(self.d.num) / Decimal(2) ** self.d.denom
                    + Decimal((self.c - self.a).num)
                    / Decimal(2) ** (self.c - self.a).denom
                    / Decimal(2).sqrt()
                )
            return x1 + x2

    def imag(self, exact: bool = False) -> float | Dsqrt2:
        """Return the imaginary part of the ring element.

        Args:
            exact (bool): If set to true, return the imaginary part as a Dsqrt2 object. Default == False.

        Returns:
            float | Dsqrt2: Imaginary part of the ring element.
        """
        if exact:
            return Dsqrt2(self.b, (self.c + self.a) * D(1, 1))
        else:
            x1 = float(self.b)
            x2 = float(self.c - self.a) / math.sqrt(2)
            if math.isclose(x1, -x2, rel_tol=1e-4):
                getcontext().prec = 50
                return float(
                    Decimal(self.b.num) / Decimal(2) ** self.b.denom
                    + Decimal((self.c + self.a).num)
                    / Decimal(2) ** (self.c + self.a).denom
                    / Decimal(2).sqrt()
                )
            return x1 + x2

    def sde(self) -> int | float:
        """Return the smallest denominator exponent (sde) of base √2 of the ring element.

        The sde of the ring element d ∈ D[ω] is the smallest integer value k such that d * (√2)^k ∈ Z[ω].
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
            while all([coeff % 2 == 0 for coeff in coeffs]) and any(coeffs):
                coeffs = [coeff // 2 for coeff in coeffs]
                sde -= 2
        while coeffs[0] % 2 == coeffs[2] % 2 and coeffs[1] % 2 == coeffs[3] % 2 and any(coeffs):
            alpha: int = (coeffs[1] - coeffs[3]) // 2
            beta: int = (coeffs[2] + coeffs[0]) // 2
            gamma: int = (coeffs[1] + coeffs[3]) // 2
            delta: int = (coeffs[2] - coeffs[0]) // 2
            coeffs = [alpha, beta, gamma, delta]
            sde -= 1
        return sde

    def complex_conjugate(self) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Return the complex conjugate of the ring element."""
        return Domega(a=-self.c, b=-self.b, c=-self.a, d=self.d).convert(type(self))

    def sqrt2_conjugate(self) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Return the √2 conjugate of the ring element."""
        return Domega(a=-self.a, b=self.b, c=-self.c, d=self.d).convert(type(self))

    def __repr__(self) -> str:
        """Define the string representation of the class."""
        sign: Callable[[D], str] = lambda coeff: "+" if coeff.num >= 0 else "-"
        value: Callable[[D], str] = lambda coeff: str(coeff) if coeff.num >= 0 else str(-coeff)
        omega: Callable[[int], str] = lambda index: "ω" + str(index) if index > 0 else ""
        if all([coeff.num == 0 for coeff in self]):
            return "0"
        else:
            output: str = ""
            for index, coeff in enumerate(self):
                if coeff.num != 0:
                    if coeff.num < 0:
                        if coeff == -1 and index != 3:
                            output += sign(coeff) + " " + omega(3 - index) + " "
                        else:
                            output += sign(coeff) + " " + value(coeff) + omega(3 - index) + " "
                    elif coeff.num > 0 and len(output) == 0:
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
        return math.isclose(self.real(), complex(nb).real) and math.isclose(
            self.imag(), complex(nb).imag
        )

    def __neq__(self, nb: Any) -> bool:
        """Define the != operation of the class."""
        return not self.__eq__(nb)

    def __neg__(self) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the negation of the ring element."""
        return Domega(-self.a, -self.b, -self.c, -self.d).convert(type(self))

    def __complex__(self) -> complex:
        """Define the complex representation of the class."""
        return self.real() + 1.0j * self.imag()

    def __add__(self, nb: Any) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the summation operation for the ring elements."""
        if isinstance(nb, (D, Integral)):
            return Domega(self.a, self.b, self.c, self.d + nb).convert(
                output_type(type(self), type(nb))
            )
        elif issubclass(type(nb), Domega):
            return Domega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d).convert(
                output_type(type(self), type(nb))
            )
        raise TypeError(
            f"Summation operation is not defined between {self.__class__.__name__} and {type(nb).__name__}."
        )

    def __radd__(self, nb: Any) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the right summation of integers and D objects with the Domega class."""
        return self.__add__(nb)

    def __iadd__(self, nb: Any) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: Any) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the subtraction operation for the Domega class."""
        if not isinstance(nb, ((Integral, D, Zsqrt2, Dsqrt2, Zomega, Domega))):
            raise TypeError(
                f"Subtraction operation is not defined between {self.__class__.__name__} and {type(nb).__name__}."
            )
        return self.__add__(-nb)

    def __rsub__(self, nb: Any) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the right subtraction of integers and D objects with the Domega class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: Any) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the in-place subtraction for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Any) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the multiplication operation for the Domega class."""
        if isinstance(nb, (D, Integral)):
            return Domega(self.a * nb, self.b * nb, self.c * nb, self.d * nb).convert(
                output_type(type(self), type(nb))
            )
        elif isinstance(nb, (Zsqrt2, Dsqrt2, Zomega, Domega)):
            a: D = (self.a * nb.d) + (self.b * nb.c) + (self.c * nb.b) + (self.d * nb.a)
            b: D = -(self.a * nb.a) + (self.b * nb.d) + (self.c * nb.c) + (self.d * nb.b)
            c: D = -(self.a * nb.b) + -(self.b * nb.a) + (self.c * nb.d) + (self.d * nb.c)
            d: D = -(self.a * nb.c) + -(self.b * nb.b) + -(self.c * nb.a) + (self.d * nb.d)
            return Domega(a, b, c, d).convert(output_type(type(self), type(nb)))
        raise TypeError(
            f"Product operation is not defined between {self.__class__.__name__} and {type(nb).__name__}."
        )

    def __rmul__(self, nb: Any) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the right multiplication of integers and D objects with the Domega class."""
        return self.__mul__(nb)

    def __imul__(self, nb: Any) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
        """Define the in-place multiplication for the class."""
        return self.__mul__(nb)

    def __pow__(self, power: int) -> Zsqrt2 | Dsqrt2 | Zomega | Domega:
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
                    multinomial_coefficient = math.factorial(power) // math.prod(
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
    """Class to do symbolic computation with elements of the ring of cyclotomic integers of degree 8 Z[ω].

    The ring element has the form aω^3 + bω^2 + cω + d, where ω = (1 + i)/√2 is the eight root of unity.
    The coefficients a, b, c, d are integers.

    The ring element can also be expressed as alpha + i*beta, where i = √-1, and alpha and beta are numbers in te ring D[√2].
    alpha = d + (c-a)/2 √2 and beta = b + (c+a)/2 √2.

    Attributes:
        a (D): ω^3 coefficient of the ring element.
        b (D): ω^2 coefficient of the ring element.
        c (D): ω^1 coefficient of the ring element.
        d (D): ω^0 coefficient of the ring element.
    """

    def __init__(self, a: int, b: int, c: int, d: int) -> None:
        """Initialize the Zomega class.

        Args:
            a (int): ω^3 coefficient of the ring element.
            b (int): ω^2 coefficient of the ring element.
            c (int): ω^1 coefficient of the ring element.
            d (int): ω^0 coefficient of the ring element.

        Raises:
            TypeError: If the class arguments are not integers.
        """
        for arg in (a, b, c, d):
            if not isinstance(arg, Integral):
                raise TypeError(
                    f"Class arguments must be integers, but got {arg} of type {type(arg).__name__}"
                )
        super().__init__((a, 0), (b, 0), (c, 0), (d, 0))

    def __float__(self) -> float:
        """Define the float representation of the class."""
        return self.real()

    def __lt__(self, nb: Any) -> bool:
        """Define the < operation for the class."""
        return float(self) < nb

    def __le__(self, nb: Any) -> bool:
        """Define the <= operation for the class."""
        return float(self) < nb or self == nb

    def __gt__(self, nb: Any) -> bool:
        """Define the > operation for the class."""
        return float(self) > nb

    def __ge__(self, nb: Any) -> bool:
        """Define the >= operation for the class."""
        return float(self) > nb or self == nb


class Dsqrt2(Domega):
    """Class to do symbolic computation with element in the ring of quadratic dyadic fractions D[√2].

    The ring element has the form p + q√2, where p, q are dyadic fractions of the form m / 2^n,
    where m is an integer and n is a positive integer.
    The coefficients will be automatically reduced when the class is initialized.

    Attributes:
        p (D): Rational coefficient of the ring element.
        q (D): √2 coefficient of the ring element.

    Raises:
        TypeError: If the class arguments are not 2-tuples of integers (num, denom) or D objects.
        ValueError: If the denominator exponent is negative.
    """

    def __init__(self, p: tuple[int, int] | D, q: tuple[int, int] | D) -> None:
        """Initialize the Dsqrt2 class.

        Args:
            p (tuple[int, int] | D): Rational coefficient of the ring element.
            q (tuple[int, int] | D): √2 coefficient of the ring element.

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
        self._p: D = p if isinstance(p, D) else D(p[0], p[1])
        self._q: D = q if isinstance(q, D) else D(q[0], q[1])
        super().__init__(-self._q, D(0, 0), self._q, self._p)

    @property
    def p(self) -> D:
        """Rational coefficient of the ring element."""
        return self._p

    @property
    def q(self) -> D:
        """√2 coefficient of the ring element."""
        return self._q

    def __repr__(self) -> str:
        """Define the string representation of the ring element."""
        repr: str = ""
        if self.p != 0:
            repr += str(self.p)
            if self.q != 0:
                if self.q > 0:
                    repr += f"+{self.q if self.q != 1 else ''}√2"
                elif self.q < 0:
                    repr += f"-{-self.q if self.q != -1 else ''}√2"
        elif self.q != 0:
            if self.q == -1:
                repr += "-"
            repr += f"{self.q if self.q != 1 and self.q != -1 else ''}√2"
        else:
            repr += str(0)
        return repr

    def __getitem__(self, index: int) -> D:
        """Return the coefficients of the ring element from their index."""
        return [self.p, self.q][index]

    def __iter__(self) -> Iterator:
        """Allow iteration in the class coefficients."""
        return iter([self.p, self.q])

    def __float__(self) -> float:
        """Define the float representation of the class."""
        return self.real()

    def __lt__(self, nb: Any) -> bool:
        """Define the < operation for the class."""
        return float(self) < nb

    def __le__(self, nb: Any) -> bool:
        """Define the <= operation for the class."""
        return float(self) < nb or self == nb

    def __gt__(self, nb: Any) -> bool:
        """Define the > operation for the class."""
        return float(self) > nb

    def __ge__(self, nb: Any) -> bool:
        """Define the >= operation for the class."""
        return float(self) > nb or self == nb


class Zsqrt2(Zomega):
    """A simple class to do symbolic computation with elements of the ring Z[√2].

    The ring element has the form p + q√2, where p and q are integers.

    Attributes:
        p (int): Integer coefficient of the ring element.
        q (int): √2 coefficient of the ring element.
    """

    def __init__(self, p: int, q: int) -> None:
        """Initialize the ring element.

        Args:
            p (int): Integer coefficient of the ring element.
            q (int): √2 coefficient of the ring element.

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
        super().__init__(a=-q, b=0, c=q, d=p)

    @property
    def p(self) -> int:
        """Integer coefficient of the ring element."""
        return self._p

    @property
    def q(self) -> int:
        """√2 coefficient of the ring element."""
        return self._q

    def __repr__(self) -> str:
        """Define the string representation of the ring element."""
        return Dsqrt2.__repr__(self)

    def __getitem__(self, index: int) -> int:
        """Return the coefficients of the ring element from their index."""
        return [self.p, self.q][index]

    def __iter__(self) -> Iterator:
        """Allow iteration in the class coefficients."""
        return iter([self.p, self.q])


def output_type(*types: type) -> type:
    """Return the output type of class operations.

    Args:
        *types (type): Variable-length argument list of Python types.

    Returns:
        type: Output type of the operation.
    """
    if Domega in types:
        return Domega
    elif Zomega in types:
        if Dsqrt2 in types or D in types:
            return Domega
        else:
            return Zomega
    elif Dsqrt2 in types:
        return Dsqrt2
    elif Zsqrt2 in types:
        if D in types:
            return Dsqrt2
        else:
            return Zsqrt2
    else:
        raise ValueError(
            f"Conversion between {', '.join([t.__name__ for t in types])} is not supported."
        )


lamb: Zsqrt2 = Zsqrt2(1, 1)
inv_lamb: Zsqrt2 = Zsqrt2(-1, 1)
