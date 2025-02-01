from __future__ import annotations

import math
from decimal import Decimal, getcontext
from typing import Any, Callable, Iterator

import numpy as np
from cliffordplust.rings import D

__all__ = ["Domega"]

SQRT2: float = math.sqrt(2)


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
    def is_Zomega(self) -> bool:
        """True if the ring element is element of \u2124[\u03C9]."""
        return all([coeff.is_integer for coeff in self])

    @property
    def is_Dsqrt2(self) -> bool:
        """True if the ring element is element of \u2145[\u221a2]."""
        return self.b == 0 and self.c + self.a == 0

    @property
    def is_Zsqrt2(self) -> bool:
        """True if the ring element is element of \u2124[\u221a2]."""
        return (
            self.b == 0
            and self.c + self.a == 0
            and self.d.is_integer
            and ((self.c - self.a) * D(1, 1)).is_integer
        )

    @property
    def is_D(self) -> bool:
        """True if the ring element is element of \u2145."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @property
    def is_integer(self) -> bool:
        """True if the ring element is an integer."""
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d.is_integer

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
            + "\u03C91"
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
            return self.is_D and self.d == nb
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

    def __add__(self, nb: Any) -> Domega:
        """Define the summation operation for the ring elements."""
        if isinstance(nb, (Domega)):
            return Domega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d)
        elif isinstance(nb, (D, int, np.integer)):
            return Domega(self.a, self.b, self.c, self.d + nb)
        raise TypeError(
            f"Summation operation is not defined between Domega and {type(nb).__name__}."
        )

    def __radd__(self, nb: Any) -> Domega:
        """Define the right summation of integers and D objects with the Domega class."""
        return self.__add__(nb)

    def __iadd__(self, nb: Any) -> Domega:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: Any) -> Domega:
        """Define the subtraction operation for the ring element."""
        if isinstance(nb, (Domega, D, int, np.integer)):
            return self.__add__(-nb)
        raise TypeError(
            f"Subtraction operation is not defined between Domega and {type(nb).__name__}."
        )

    def __rsub__(self, nb: Any) -> Domega:
        """Define the right subtraction of integers and D objects with the Domega class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: Any) -> Domega:
        """Define the in-place subtraction for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Any) -> Domega:
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

    def __rmul__(self, nb: Any) -> Domega:
        """Define the right multiplication of integers and D objects with the Domega class."""
        return self.__mul__(nb)

    def __imul__(self, nb: Any) -> Domega:
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
