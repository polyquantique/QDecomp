from __future__ import annotations

from grid_algorithm_1D.Zsqrt2 import Zsqrt2
import numpy as np
import math
from typing import Iterator, Any

class D:
    """Class to do symbolic computation with elements of the ring of dyadic fractions D.
    
    The ring element has the form a / b^k, where a is an integer and k is a positive integer. 
    The fraction will automatically be reduced in the case of a even numerator.

    Attributes:
        num (int): Numerator of the ring element.
        denum (int): Power of 2 in the denominator of the ring element.
        is_integer (bool): True if the ring element is an integer.
    """

    def __init__(self, num: int, denum: int) -> None:
        """Initialize the ring element.

        Args:
            num (int): Numerator of the ring element
            denum (int): Power of 2 in the denominator of the ring element.
        
        Raises:
            TypeError: If num or denum are not integers.
            ValueError: If denum is not positive.
        """
        if not isinstance(num, (int, np.int32, np.int64)) or not isinstance(denum, (int, np.int32, np.int64)):
            raise TypeError(f"Class arguments must be of type int but received {type(num).__name__ if not isinstance(num, (int, np.int32, np.int64)) else type(denum).__name__}.")
        elif denum < 0:
            raise ValueError(f"denum value must be positive but got {denum}")
        self._num = num
        self._denum = denum
        self.__reduce()
    
    @property
    def num(self) -> int:
        return self._num

    @property
    def denum(self) -> int:
        return self._denum
    
    @property
    def is_integer(self) -> bool:
        """Return True if the exponent in the denominator is 0, i.e. if the number is an integer."""
        return self.denum == 0
    
    def __reduce(self) -> None:
        """Reduce the fraction if the numerator is even."""
        while self.num % 2 == 0 and self.num != 0 and self.denum > 0:
            self._num //= 2
            self._denum -= 1

    def __neg__(self) -> D:
        """Define the negation of the class D."""
        return D(-self.num, self.denum)
    
    def __repr__(self) -> str:
        """Define the string representation of the class D."""
        return f"{self.num}/2^{self.denum}"
    
    def __float__(self) -> float:
        """Define the float value of the class."""
        return self.num / 2**self.denum
    
    def __eq__(self, nb: Any) -> bool:
        """Define the equality of the D class."""
        return math.isclose(float(self), nb)

    def __add__(self, nb: D | int) -> D:
        """Define the summation operation for the D class. Allow summation with D elements or integers."""
        if not isinstance(nb, (D, int, np.int32, np.int64)):
            raise TypeError(f"Summation operation is not defined for the {type(nb).__name__} class.")
        if isinstance(nb, D):
            if self._denum >= nb._denum:
                num = self.num + nb.num * 2**(self.denum - nb.denum)
                denum = self.denum
                return D(num, denum)
            else:
                num = nb.num + self.num * 2**(nb.denum - self.denum)
                denum = nb.denum
                return D(num, denum)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return D(self.num + nb * 2**self.denum, self.denum)
    
    def __radd__(self, nb: int) -> D:
        """Define the right summation of integers with the D class."""
        return self.__add__(nb)
    
    def __sub__(self, nb: D | int) -> D:
        """Define the subtraction operation for the D class. Allow subtraction with D elements or integers."""
        if not isinstance(nb, (D, int, np.int32, np.int64)):
            raise TypeError(f"Subtraction operation is not defined for the {type(nb).__name__} class.")
        return self.__add__(-nb)
    
    def __rsub__(self, nb: int) -> D:
        """Define the right subtraction of integers with the D class."""
        return -self + nb
        
    def __mul__(self, nb: D | int) -> D:
        """Define the product operation for the D class. Allow products with D elements or with integers."""
        if not isinstance(nb, (D, int, np.int32, np.int64)):
            raise TypeError(f"Product operation is not defined for the {type(nb).__name__} class.")
        if isinstance(nb, D):
            return D(self.num * nb.num, self.denum + nb.denum)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return D(self.num * nb, self.denum)
        
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
    def __init__(self, a: tuple[int, int] | D, b: tuple[int, int] | D, c: tuple[int, int] | D, d: tuple[int, int] | D) -> None:
        """Initialize the Domega class.
        
        Args:
            a (tuple[int, int] | D): ω^3 coefficient of the ring element.
            b (tuple[int, int] | D): ω^2 coefficient of the ring element.
            c (tuple[int, int] | D): ω^1 coefficient of the ring element.
            d (tuple[int, int] | D): ω^0 coefficient of the ring element.

        Raises:
        """
    
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

    def sde(self) -> int:
        if any([coeff.denum != 0 for coeff in self]):
            return 2 * max([coeff.denum for coeff in self])
        else:
            count = 0
            coeffs = [coeff.num for coeff in self]
            while all([coeff % 2 == 0 for coeff in coeffs]):
                count -= 2
                coeffs = [coeff // 2 for coeff in coeffs]
            if coeffs[1] == 0 and coeffs[3] == 0:
                if coeffs[0] % 2 == 1 and coeffs[2] % 2 == 1:
                    count -= 1
                    alpha, beta = coeffs[2] - coeffs[0], coeffs[0] + coeffs[2]
                    alpha //= 2
                    beta //= 2
                    while alpha % 2 == 0 and beta % 2 == 0 and alpha != 0 and beta != 0:
                        count -= 2
                        alpha //= 2
                        beta //= 2
            return count

    def __repr__(self) -> str:
        return f"{self.a} ω3 {self.b} ω2 + {self.c} ω1 + {self.d}"

    def __getitem__(self, i: int | slice) -> D | list[D]:
        return [self.a, self.b, self.c, self.d][i]
    
    def __iter__(self) -> Iterator[D]:
        return iter([self.a, self.b, self.c, self.d])
    
    def complex_conjugate(self) -> Domega:
        return Domega(a=-self.c, b = -self.b, c= -self.a, d=self.d)
    
    def sqrt2_conjugate(self) -> Domega:
        return Domega(a=-self.a, b=self.b, c=-self.c, d=self.d)
    
    def __neg__(self) -> Domega:
        return Domega(-self.a, -self.b, -self.c, -self.d)
    
    def __add__(self, nb: Domega | D | int) -> Domega:
        if isinstance(nb, (int, np.int32, np.int64, D)):
            return Domega(self.a, self.b, self.c, self.d + nb)
        elif isinstance(nb, Domega):
            return Domega(self.a + nb.a, self.b + nb.b, self.c + nb.c, self.d + nb.d)
    
    def __radd__(self, nb: int | D) -> Domega:
        return self.__add__(nb)
    
    def __sub__(self, nb: int | D | int) -> Domega:
        return self.__add__(-nb)
    
    def __rsub__(self, nb: int | D) -> Domega:
        return -self + nb
        
    def __mul__(self, nb: Domega | D | int) -> Domega:
        if isinstance(nb, (int, np.int32, np.int64)):
            return Domega(self.a * D(nb, 0), self.b * D(nb, 0), self.c * D(nb, 0), self.d * D(nb, 0))
        elif isinstance(nb, D):
            return Domega(self.a * nb, self.b * nb, self.c * nb, self.d * nb)
        elif isinstance(nb, Domega):
            a: D = (self.a * nb.d) + (self.b * nb.c) + (self.c * nb.b) + (self.d * nb.a)
            b: D = -(self.a * nb.a) + (self.b * nb.d) + (self.c * nb.c) + (self.d * nb.b)
            c: D = -(self.a * nb.b) + -(self.b * nb.a) + (self.c * nb.d) + (self.d * nb.c)
            d: D = -(self.a * nb.c) + -(self.b * nb.b) + -(self.c * nb.a) + (self.d + nb.d)
            return Domega(a, b, c, d)
        
    def __rmul__(self, nb: int | D) -> Domega:
        return self.__mul__(nb)
    
    def real(self) -> float:
        return float(self.d) + float(self.c - self.a)/math.sqrt(2)
    
    def imag(self) -> float:
        return float(self.b) + float(self.a + self.c) / math.sqrt(2)
    
    def __complex__(self) -> complex:
        return self.real() + 1.j * self.imag()
    
    
        

if __name__ == "__main__":
    n1 = Domega(D(2, 1), D(-5, 0), D(3, 0), D(14, 2))
    n2 = Domega(D(0, 0), D(0, 0), D(0, 0), D(1, 2))
    n3 = Domega(D(8, 0), D(0, 0), D(-8, 0), D(0, 0))
    print(n3.sde())
    print(n2.sde())
    n4 = D("f", 2.2)