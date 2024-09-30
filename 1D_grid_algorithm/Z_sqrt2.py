from __future__ import annotations

import math
from typing import Optional, Union


class Z_sqrt2:
    def __init__(self, a: int, b: int) -> None:
        self.a: int = a
        self.b: int = b
        for input in (a, b):
            if not isinstance(input, int):
                raise TypeError(
                    f"Expected inputs to be of type int, but got {type(input).__name__}."
                )

    def __getitem__(self, i: int) -> int:
        return (self.a, self.b)[i]

    def __repr__(self) -> str:
        if self.b >= 0:
            return f"{self.a} + {self.b}√2"
        else:
            return f"{self.a} - {-self.b}√2"

    def __add__(self, nb: Union[Z_sqrt2, int]) -> Z_sqrt2:
        if isinstance(nb, Z_sqrt2):
            return Z_sqrt2(self.a + nb.a, self.b + nb.b)
        elif isinstance(nb, int):
            return Z_sqrt2(self.a + nb, self.b)
        else:
            raise TypeError(
                f"'{type(self).__name__}' + '{type(nb).__name__}' operation is not defined"
            )

    def __iadd__(self, nb: Union[Z_sqrt2, int]) -> Z_sqrt2:
        return self + nb

    def __sub__(self, nb: Union[Z_sqrt2, int]) -> Z_sqrt2:
        if isinstance(nb, Z_sqrt2):
            return Z_sqrt2(self.a - nb.a, self.b - nb.b)
        elif isinstance(nb, int):
            return Z_sqrt2(self.a - nb, self.b)
        else:
            raise TypeError(
                f"'{type(self).__name__}' - '{type(nb).__name__}' operation is not defined"
            )

    def __isub__(self, nb: Union[Z_sqrt2, int]) -> Z_sqrt2:
        return self - nb

    def __mul__(self, nb: Union[Z_sqrt2, int]) -> Z_sqrt2:
        if isinstance(nb, Z_sqrt2):
            return Z_sqrt2(self.a * nb.a + 2 * self.b * nb.b, self.a * nb.b + self.b * nb.a)
        elif isinstance(nb, int):
            return Z_sqrt2(self.a * nb, self.b * nb)
        else:
            raise TypeError(
                f"'{type(self).__name__}' * '{type(nb).__name__}' operation is not defined"
            )

    def __imul__(self, nb: Union[Z_sqrt2, int]) -> Z_sqrt2:
        return self * nb

    def __pow__(self, n: int) -> Z_sqrt2:
        if not isinstance(n, int):
            raise TypeError(f"Expected power to be of type int, but got {type(n).__name__}.")
        elif n < 0:
            raise ValueError(f"Expected power to be a positive integer, but got {n}.")
        pow_out = Z_sqrt2(1, 0)
        for i in range(n):
            pow_out = self * pow_out
        return pow_out

    def __ipow__(self, nb: int) -> Z_sqrt2:
        return self**nb

    def __neg__(self) -> Z_sqrt2:
        return Z_sqrt2(-self.a, -self.b)

    def __float__(self) -> float:
        return self.a + self.b * math.sqrt(2)

    def __round__(self, precision: Optional[int] = None) -> Union[int, float]:
        if precision is None:
            return round(float(self))
        else:
            return round(float(self), precision)

    def __floor__(self) -> int:
        return math.floor(float(self))

    def __ceil__(self) -> int:
        return math.ceil(float(self))

    def conjugate(self) -> Z_sqrt2:
        return Z_sqrt2(self.a, -self.b)


lamb: Z_sqrt2 = Z_sqrt2(1, 1)
inv_lamb: Z_sqrt2 = -lamb.conjugate()

if __name__ == "__main__":
    x = Z_sqrt2(1, 1)
    y = Z_sqrt2(2, 5)
    print(x**-1)
