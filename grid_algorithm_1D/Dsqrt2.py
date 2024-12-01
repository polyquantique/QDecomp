from __future__ import annotations

import numpy as np
import math
from typing import Iterator, Any

class D:
    """Class to do symbolic computation with elements of the ring of dyadic fractions D.

    The ring element has the form a / 2^k, where a is an integer and k is a positive integer.
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
            return f"{self.num}/(2^{self.denom})"

    def __float__(self) -> float:
        """Define the float value of the class."""
        return self.num / 2**self.denom

    def __eq__(self, other: Any) -> bool:
        """Define the equality of the D class."""
        if isinstance(other, D):
            return self.num == other.num and self.denom == other.denom
        elif isinstance(other, int):
            return self.num == other and self.denom == 0
        elif isinstance(other, float):
            return math.isclose(float(self), other)
        return TypeError("Non-comparable types")

    def __add__(self, other: D | int) -> D:
        """Define the summation operation for the D class. Allow summation with D elements or integers."""
        if not isinstance(other, (D, int, np.int32, np.int64)):
            raise TypeError(
                f"Summation operation is not defined for the {type(other).__name__} class."
            )
        if isinstance(other, D):
            if self._denom >= other._denom:
                num = self.num + other.num * 2 ** (self.denom - other.denom)
                denom = self.denom
                return D(num, denom)
            else:
                num = other.num + self.num * 2 ** (other.denom - self.denom)
                denom = other.denom
                return D(num, denom)
        elif isinstance(other, (int, np.int32, np.int64)):
            return D(self.num + other * 2**self.denom, self.denom)

    def __radd__(self, other: int) -> D:
        """Define the right summation of integers with the D class."""
        return self.__add__(other)

    def __sub__(self, other: D | int) -> D:
        """Define the subtraction operation for the D class. Allow subtraction with D elements or integers."""
        if not isinstance(other, (D, int, np.int32, np.int64)):
            raise TypeError(
                f"Subtraction operation is not defined for the {type(nb).__name__} class."
            )
        return self.__add__(-other)

    def __rsub__(self, other: int) -> D:
        """Define the right subtraction of integers with the D class."""
        return -self + other

    def __mul__(self, other: D | int) -> D:
        """Define the product operation for the D class. Allow products with D elements or with integers."""
        if not isinstance(other, (D, int, np.int32, np.int64)):
            raise TypeError(f"Product operation is not defined for the {type(other).__name__} class.")
        if isinstance(other, D):
            return D(self.num * other.num, self.denom + other.denom)
        elif isinstance(other, (int, np.int32, np.int64)):
            return D(self.num * other, self.denom)

    def __rmul__(self, other: int) -> D:
        """Define the right multiplication of integers with the D class."""
        return self.__mul__(other)
    
    def __pow__(self, exponent: int):
        """Raise the dyadic number to an integer power."""
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer.")
        if exponent < 0:
            raise ValueError("Exponent must be non-negative.")
        return D(self.num ** exponent, self.denom ** exponent)

class Dsqrt2:
    """Class to do symbolic computation with elements of the ring D[√2].

    The ring element has the form a + b√2. 
    The coefficients a, b are dyadic fractions of the form m / 2^n, where m is an integer and n is a positive integer.
    The coefficient will be automatically reduced when the class is initialized.

    Attributes:
        a (D): Rationnal coefficient of the ring element.
        b (D): √2 coefficient of the ring element.
        is_Zsqrt2 (bool): True if the ring element is also in the ring Z[√2].
    """
    def __init__(self, a: tuple[int, int] | D, b: tuple[int, int] | D) -> None:
        """Initialize the Dsqrt2 class.
        
        Args:
            a (tuple[int, int] | D): Rationnal coefficient of the ring element.
            b (tuple[int, int] | D): √2 coefficient of the ring element.

        Raises:
            TypeError: If `a` or `b` are not integers or instances of `D` or tuples of two integers.
        """

        # Check if `a` is of the correct type
        if not (isinstance(a, int) or isinstance(a, D) or (isinstance(a, tuple) and len(a) == 2 and all(isinstance(i, int) for i in a))):
            raise TypeError("Argument `a` must be an instance of `D` or a tuple of two integers (int, int).")
        
        # Check if `b` is of the correct type
        if not (isinstance(b, int) or isinstance(b, D) or (isinstance(b, tuple) and len(b) == 2 and all(isinstance(i, int) for i in b))):
            raise TypeError("Argument `b` must be an instance of `D` or a tuple of two integers (int, int).")
        
        if isinstance(a, D):
            self._a: D = a
        elif isinstance(a, int):
            self._a: D = D(a, 0)
        else:
            self._a: D = D(a[0], a[1])
        
        if isinstance(b, D):
            self._b: D = b
        elif isinstance(b, int):
            self._b: D = D(b, 0)
        else:
            self._b: D = D(b[0], b[1])
        
    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b
    
    @property
    def is_Zsqrt2(self) -> bool:
        return all([coeff.is_integer for coeff in self])
    
    def __repr__(self) -> str:
        return f"({self.a}) + ({self.b})√2"
    
    def __getitem__(self, i: int | slice) -> D | list[D]:
        return [self.a, self.b][i]
    
    def __iter__(self) -> Iterator[D]:
        return iter([self.a, self.b])
    
    def conjugate(self) -> Dsqrt2:
        return Dsqrt2(self.a, -self.b)
    
    def __neg__(self) -> Dsqrt2:
        return Dsqrt2(-self.a, -self.b)
    
    def __float__(self) -> float:
        """Define the float value of the class."""
        return float(self.a) + float(self.b)*(np.sqrt(2))
    
    def __eq__(self, other: Any) -> bool:
        """Define the equality of the Dsqrt2 class."""
        if isinstance(other, Dsqrt2):
            return self._a == other._a and self._b == other._b
        elif isinstance(other, (int, D)):
            return self._a == other
        elif isinstance(other, float):
            return math.isclose(float(self), float(other))
        return TypeError("Non-comparable types")
    
    def __add__(self, other: Dsqrt2 | D | int) -> Dsqrt2:
        if isinstance(other, (int, np.int32, np.int64, D)):
            return Dsqrt2(self.a + other, self.b)
        elif isinstance(other, Dsqrt2):
            return Dsqrt2(self.a + other.a, self.b + other.b)
    
    def __radd__(self, other: int | D) -> Dsqrt2:
        return self.__add__(other)
    
    def __sub__(self, other: int | D | int) -> Dsqrt2:
        return self.__add__(-other)
    
    def __rsub__(self, other: int | D) -> Dsqrt2:
        return -self + other
    
    def __mul__(self, other: Dsqrt2 | D | int | float) -> Dsqrt2 | float:
        if isinstance(other, (int, np.int32, np.int64)):
            return Dsqrt2(self.a * D(other, 0), self.b * D(other, 0))
        elif isinstance(other, D):
            return Dsqrt2(self.a * other, self.b * other)
        elif isinstance(other, Dsqrt2):
            a: D = self.a * other.a + 2 * self.b * other.b
            b: D = self.a * other.b + self.b * other.a
            return Dsqrt2(a, b)
        elif isinstance(other, float):
            return float(self) * other
        
    def __rmul__(self, other: int | D | Dsqrt2) -> Dsqrt2:
        return self.__mul__(other)
    
    def __pow__(self, exponent: int):
        """Raise the ring number to an integer power."""
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer.")
        if exponent < 0:
            raise ValueError("Exponent must be non-negative.")
        
        result = Dsqrt2(1, 0)  # Start with 1

        # Multiply self with itself exponent times
        base = self
        for _ in range(exponent):
            result = result * base  # Uses the __mul__ method already defined

        return result
    
lamb: Dsqrt2 = Dsqrt2(1, 1)

inv_lamb: Dsqrt2 = Dsqrt2(-1, 1)