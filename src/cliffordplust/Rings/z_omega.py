from __future__ import annotations

import math
from decimal import Decimal, getcontext
from typing import Any, Callable, Iterator

import numpy as np

__all__ = ["Zomega"]

SQRT2: float = math.sqrt(2)


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
    def is_Dsqrt2(self) -> bool:
        """True if the ring element is element of \u2145[\u221a2]."""
        return self.b == 0 and self.c + self.a == 0

    @property
    def is_Zsqrt2(self) -> bool:
        """True if the ring element is element of \u2124[\u221a2]."""
        return self.b == 0 and self.c + self.a == 0 and (self.c - self.a) % 2 == 0

    @property
    def is_D(self) -> bool:
        """True if the ring element is element of \u2145."""
        return self.a == 0 and self.b == 0 and self.c == 0

    @property
    def is_integer(self) -> bool:
        """True if the ring element is an integer."""
        return self.a == 0 and self.b == 0 and self.c == 0

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

    def sde(self) -> int | float:
        """Return the smallest denominator exponent (sde) of base \u221a2 of the ring element.

        The sde of the ring element d \u2208 \u2124[\u03C9] is the smallest integer value k such that d * (\u221a2)^k \u2208 \u2124[\u03C9].
        """
        sde: int = 0

        # If all coefficients are zero, return negative infinity.
        coeffs: list[int] = [self.a, self.b, self.c, self.d]
        if not any(coeffs):
            return -math.inf

        # If the coefficients are all even, we can divide by 2 and remain in Z[ω].
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

    def __add__(self, nb: Any) -> Zomega:
        """Define the summation operation for the ring elements."""
        if isinstance(nb, Zomega):
            return Zomega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d)
        elif isinstance(nb, (int, np.integer)):
            return Zomega(self.a, self.b, self.c, self.d + nb)
        raise TypeError(
            f"Summation operation is not defined between Zomega and {type(nb).__name__}."
        )

    def __radd__(self, nb: Any) -> Zomega:
        """Define the right summation of integers with the Zomega class."""
        return self.__add__(nb)

    def __iadd__(self, nb: Any) -> Zomega:
        """Define the in-place summation operation for the class."""
        return self.__add__(nb)

    def __sub__(self, nb: Any) -> Zomega:
        """Define the subtraction operation for the ring element."""
        if isinstance(nb, (Zomega, int, np.integer)):
            return self.__add__(-nb)
        raise TypeError(
            f"Subtraction operation is not defined between Zomega and {type(nb).__name__}."
        )

    def __rsub__(self, nb: Any) -> Zomega:
        """Define the right subtraction of integers with the Zomega class."""
        return (-self).__add__(nb)

    def __isub__(self, nb: Any) -> Zomega:
        """Define the in-place subtraction for the class."""
        return self.__sub__(nb)

    def __mul__(self, nb: Any) -> Zomega:
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

    def __rmul__(self, nb: Any) -> Zomega:
        """Define the right multiplication of integers with the Zomega class."""
        return self.__mul__(nb)

    def __imul__(self, nb: Any) -> Zomega:
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
