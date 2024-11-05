from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np

from grid_algorithm_1D.Zsqrt2 import Zsqrt2


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
        if not isinstance(num, (int, np.int32, np.int64)) or not isinstance(
            denom, (int, np.int32, np.int64)
        ):
            raise TypeError(
                f"Class arguments must be of type int but received {type(num).__name__ if not isinstance(num, (int, np.int32, np.int64)) else type(denom).__name__}."
            )
        elif denom < 0:
            raise ValueError(f"denom value must be positive but got {denom}")
        self._num = num
        self._denom = denom
        self.__reduce()

    @property
    def num(self) -> int:
        return self._num

    @property
    def denom(self) -> int:
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
        else:
            return f"{self.num}/2^{self.denom}"

    def __float__(self) -> float:
        """Define the float value of the class."""
        return self.num / 2**self.denom

    def __eq__(self, nb: Any) -> bool:
        """Define the equality of the D class."""
        return math.isclose(float(self), nb)

    def __add__(self, nb: D | int) -> D:
        """Define the summation operation for the D class. Allow summation with D elements or integers."""
        if not isinstance(nb, (D, int, np.int32, np.int64)):
            raise TypeError(
                f"Summation operation is not defined for the {type(nb).__name__} class."
            )
        if isinstance(nb, D):
            if self._denom >= nb._denom:
                num = self.num + nb.num * 2 ** (self.denom - nb.denom)
                denom = self.denom
                return D(num, denom)
            else:
                num = nb.num + self.num * 2 ** (nb.denom - self.denom)
                denom = nb.denom
                return D(num, denom)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return D(self.num + nb * 2**self.denom, self.denom)

    def __radd__(self, nb: int) -> D:
        """Define the right summation of integers with the D class."""
        return self.__add__(nb)

    def __sub__(self, nb: D | int) -> D:
        """Define the subtraction operation for the D class. Allow subtraction with D elements or integers."""
        if not isinstance(nb, (D, int, np.int32, np.int64)):
            raise TypeError(
                f"Subtraction operation is not defined for the {type(nb).__name__} class."
            )
        return self.__add__(-nb)

    def __rsub__(self, nb: int) -> D:
        """Define the right subtraction of integers with the D class."""
        return -self + nb

    def __mul__(self, nb: D | int) -> D:
        """Define the product operation for the D class. Allow products with D elements or with integers."""
        if not isinstance(nb, (D, int, np.int32, np.int64)):
            raise TypeError(
                f"Product operation is not defined for the {type(nb).__name__} class."
            )
        if isinstance(nb, D):
            return D(self.num * nb.num, self.denom + nb.denom)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return D(self.num * nb, self.denom)

    def __rmul__(self, nb: int) -> D:
        """Define the right multiplication of integers with the D class."""
        return self.__mul__(nb)


class Domega:
    """Class to do symbolic computation with elements of the ring D[ω].

    The ring element has the form aω^3 + bω^2 + cω + d, where ω = (1 + i)/√2 is the eight root of unity.
    The coefficients a, b, c, d are dyadic fractions of the form m / 2^n, where m is an integer and n is a positive integer.
    The coefficient will be automatically reduced when the class is initialized.

    The ring element can also be expressed as alpha + i*beta, where i = √-1, and alpha and beta are numbers in te ring D[√2].
    alpha = d + (c-a)/2 √2 and beta = b + (c+a)/2 √2.

    Attributes:
        a (D): ω^3 coefficient of the ring element.
        b (D): ω^2 coefficient of the ring element.
        c (D): ω^1 coefficient of the ring element.
        d (D): ω^0 coefficient of the ring element.
        is_Zomega (bool): True if the ring element is also in the ring Z[ω].
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
            TypeError: If the class arguments are not tuples or D.
            TypeError: If the tuple arguments are not integers.
            TypeError: If the tuple are not of length 2.
            ValueError: If the denominator exponent is negative.
        """
        for input in (a, b, c, d):
            if isinstance(input, tuple):
                if any(
                    [
                        not isinstance(value, (int, np.int32, np.int64))
                        for value in input
                    ]
                ):
                    raise TypeError(
                        f"Tuple entries must be integers, but received {type(input[[not isinstance(value, (int, np.int32, np.int64)) for value in input].index(True)]).__name__}"
                    )
                elif len(input) != 2:
                    raise TypeError(
                        f"Tuple only takes two parameters (num, denom) but received {len(input)}"
                    )
                elif input[1] < 0:
                    raise ValueError(f"denom value must be positive but got {input[1]}")
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
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

    @property
    def is_Zomega(self) -> bool:
        return all([coeff.is_integer for coeff in self])

    def real(self) -> float:
        """Return the real part of the ring element in float representation."""
        return float(self.d) + float(self.c - self.a) / math.sqrt(2)

    def imag(self) -> float:
        """Return the imaginary part of the ring element in float representation."""
        return float(self.b) + float(self.a + self.c) / math.sqrt(2)

    def sde(self) -> int | float:
        """Return the smallest denominator exponent (sde) of base √2 of the ring element.

        The sde of the ring element d ∈ D[ω] is the smallest integer value k such that d * (√2)^k ∈ Z[ω]
        """
        sde: int = 0
        if any([coeff.denom != 0 for coeff in self]):
            k_max: int = max([coeff.denom for coeff in self])
            coeffs: list[int] = [
                coeff.num * 2 ** (k_max - coeff.denom) for coeff in self
            ]
            sde += 2 * k_max
        else:
            coeffs = [coeff.num for coeff in self]
            if not any(coeffs):
                return -math.inf
            while all([coeff % 2 == 0 for coeff in coeffs]) and any(coeffs):
                coeffs = [coeff // 2 for coeff in coeffs]
                sde -= 2
        while (
            coeffs[0] % 2 == coeffs[2] % 2
            and coeffs[1] % 2 == coeffs[3] % 2
            and any(coeffs)
        ):
            alpha: int = (coeffs[1] - coeffs[3]) // 2
            beta: int = (coeffs[2] + coeffs[0]) // 2
            gamma: int = (coeffs[1] + coeffs[3]) // 2
            delta: int = (coeffs[2] - coeffs[0]) // 2
            coeffs = [alpha, beta, gamma, delta]
            sde -= 1
        return sde

    def complex_conjugate(self) -> Domega:
        """Return the complex conjugate of the ring element."""
        return Domega(a=-self.c, b=-self.b, c=-self.a, d=self.d)

    def sqrt2_conjugate(self) -> Domega:
        """Return the √2 conjugate of the ring element."""
        return Domega(a=-self.a, b=self.b, c=-self.c, d=self.d)

    def __repr__(self) -> str:
        """Define the string representation of the class."""
        sign = lambda coeff: "+" if coeff.num >= 0 else "-"
        value = lambda coeff: str(coeff) if coeff.num >= 0 else str(-coeff)
        return f"{'' if self.a.num >= 0 else '-'}{value(self.a)} ω3 {sign(self.b)} {value(self.b)} ω2 {sign(self.c)} {value(self.c)} ω1 {sign(self.d)} {value(self.d)}"

    def __getitem__(self, i: int | slice) -> D | list[D]:
        """Return the coefficients of the ring element from their index."""
        return [self.a, self.b, self.c, self.d][i]

    def __iter__(self) -> Iterator[D]:
        """Allow iteration in the class coefficients."""
        return iter([self.a, self.b, self.c, self.d])

    def __neg__(self) -> Domega:
        """Define the negation of the ring element."""
        return Domega(-self.a, -self.b, -self.c, -self.d)

    def __complex__(self) -> complex:
        """Define the complex representation of the class."""
        return self.real() + 1.0j * self.imag()

    def __add__(self, nb: Domega | D | int) -> Domega:
        """Define the summation operation for the Domega class.

        Allow summation with int, D objects and Domega objects.
        """
        if isinstance(nb, (int, np.int32, np.int64, D)):
            return Domega(self.a, self.b, self.c, self.d + nb)
        elif isinstance(nb, Domega):
            return Domega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d)
        else:
            raise TypeError(
                f"Summation operation is not defined with {type(nb).__name__}."
            )

    def __radd__(self, nb: int | D) -> Domega:
        """Define the right summation of integers and D objects with the Domega class."""
        return self.__add__(nb)

    def __iadd__(self, nb: int | D | Domega) -> Domega:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: int | D | Domega) -> Domega:
        """Define the subtraction operation for the Domega class.

        Allow subtraction with int, D objects and Domega objects.
        """
        if not isinstance(nb, ((int, np.int32, np.int64, D, Domega))):
            raise TypeError(
                f"Subtraction operation is not defined with {type(nb).__name__}."
            )
        return self.__add__(-nb)

    def __rsub__(self, nb: int | D) -> Domega:
        """Define the right subtraction of integers and D objects with the Domega class."""
        return -self + nb

    def __isub__(self, nb: int | D | Domega) -> Domega:
        """Define the in-place subtraction for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Domega | D | int) -> Domega:
        """Define the multiplication operation for the Domega class.

        Allow multiplication with int, D objects and Domega objects.
        """
        if isinstance(nb, (int, np.int32, np.int64)):
            return Domega(
                self.a * D(nb, 0),
                self.b * D(nb, 0),
                self.c * D(nb, 0),
                self.d * D(nb, 0),
            )
        elif isinstance(nb, D):
            return Domega(self.a * nb, self.b * nb, self.c * nb, self.d * nb)
        elif isinstance(nb, Domega):
            a: D = (self.a * nb.d) + (self.b * nb.c) + (self.c * nb.b) + (self.d * nb.a)
            b: D = (
                -(self.a * nb.a) + (self.b * nb.d) + (self.c * nb.c) + (self.d * nb.b)
            )
            c: D = (
                -(self.a * nb.b) + -(self.b * nb.a) + (self.c * nb.d) + (self.d * nb.c)
            )
            d: D = (
                -(self.a * nb.c) + -(self.b * nb.b) + -(self.c * nb.a) + (self.d * nb.d)
            )
            return Domega(a, b, c, d)
        else:
            raise TypeError(
                f"Product operation is not defined with {type(nb).__name__}."
            )

    def __rmul__(self, nb: int | D) -> Domega:
        """Define the right multiplication of integers and D objects with the Domega class."""
        return self.__mul__(nb)

    def __imul__(self, nb: int | D | Domega) -> Domega:
        """Define the in-place multiplication for the class."""
        return self.__mul__(nb)

    def __pow__(self, power: int) -> Domega:
        """Define the power operation for the Domega class.

        Exponent must be positive integers.
        """
        if not isinstance(power, (int, np.int32, np.int64)):
            raise TypeError(
                f"Exponent must be an integer, but received {type(power).__name__}."
            )
        if power < 0:
            raise ValueError(
                f"Expected exponent to be a positive integer, but got {power}."
            )
        out = Domega((0, 0), (0, 0), (0, 0), (1, 0))
        for _ in range(power):
            out *= self
        return out

    def __eq__(self, other):
        """Define the equality of the Domega class."""
        if isinstance(other, Domega):
            return (
                self.a == other.a
                and self.b == other.b
                and self.c == other.c
                and self.d == other.d
            )
        return False


H_11 = Domega((-1, 1), (0, 0), (1, 1), (0, 0))
T_11 = Domega((0, 0), (0, 0), (0, 0), (1, 0))
T_12 = Domega((0, 0), (0, 0), (0, 0), (0, 0))
T_22 = Domega((0, 0), (0, 0), (1, 0), (0, 0))
T_22_inv = Domega((-1, 0), (0, 0), (0, 0), (0, 0))
H = np.array([[H_11, H_11], [H_11, -H_11]], dtype=Domega)
T = np.array([[T_11, T_12], [T_12, T_22]], dtype=Domega)
T_inv = np.array([[T_11, T_12], [T_12, T_22_inv]], dtype=Domega)

if __name__ == "__main__":
    n1 = Domega(D(1, 4), D(-5, 0), (3, 4), D(-14, 2))
    n2 = Domega(D(0, 0), D(0, 0), D(0, 0), D(1, 2))
    n3 = Domega(D(3, 2), D(5, 2), D(7, 2), D(9, 2))
    n0 = Domega((0, 0), (0, 0), (0, 0), (0, 0))
    print(n3.sde())
    print(n1)
    print((n1**0))
    print(H_11.real())
    print(T @ T @ T @ T)
    print(T_inv)
    print(T_inv @ T)
