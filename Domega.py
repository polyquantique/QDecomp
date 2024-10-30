from __future__ import annotations

from grid_algorithm_1D.Zsqrt2 import Zsqrt2
import numpy as np
import math
from typing import Sequence, Iterable, Iterator, Any



class D:
    def __init__(self, num: int, denum: int) -> None:
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
        return self.denum == 0
    
    def __reduce(self) -> None:
        while self.num % 2 == 0 and self.num != 0 and self.denum > 0:
            self._num //= 2
            self._denum -= 1

    def __neg__(self) -> D:
        return D(-self.num, self.denum)
    
    def __repr__(self) -> str:
        return f"{self.num}/2^{self.denum}"
    
    def __float__(self) -> float:
        return self.num / 2**self.denum
    
    def __eq__(self, nb: Any) -> bool:
        return math.isclose(float(self), nb)

    def __add__(self, nb: D | int) -> D:
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
        return self.__add__(nb)
    
    def __sub__(self, nb: D | int) -> D:
        return self.__add__(-nb)
    
    def __rsub__(self, nb: int) -> D:
        return -self + nb
        
    def __mul__(self, nb: D | int) -> D:
        if isinstance(nb, D):
            return D(self.num * nb.num, self.denum + nb.denum)
        elif isinstance(nb, (int, np.int32, np.int64)):
            return D(self.num * nb, self.denum)
        
    def __rmul__(self, nb: int) -> D:
        return self.__mul__(nb)

    
    
    

class Domega:
    def __init__(self, a: tuple[int, int] | D, b: tuple[int, int] | D, c: tuple[int, int] | D, d: tuple[int, int] | D) -> None:

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
        return max(self.a.denum, self.b.denum, self.c.denum, self.d.denum) * 2

    def __repr__(self) -> str:
        return f"{self.a} ω3 {self.b} ω2 + {self.c} ω1 + {self.d}"

    def __getitem__(self, i: int | slice) -> D | list[D]:
        return [self.a, self.b, self.c, self.d][i]
    
    def __iter__(self) -> Iterator:
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
    print(n1, n1 + 2)
    print(n1[0])
    print(n1.sde())
    n2 = Domega(D(0, 0), D(0, 0), D(0, 0), D(1, 2))
    print(n2.sde())
    n3 = Domega(D(-2, 0), D(0, 0), D(-2, 0) + 4, D(0, 0))
    print(n3)
    print(complex(n3))
    print(n3.sde())
