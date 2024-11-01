from __future__ import annotations

from Dsqrt2 import D, Dsqrt2
import numpy as np
from typing import Union, Iterable, Sequence

class G_op:
    """Class to do symbolic computation with matrices of elements of the ring D[√2].

    

    Attributes:
        
    """
    def __init__(self, elements) -> None:
        """Initialize the G_op class.
        
        Args:

        Raises:
        """
        flat_elements = self._flatten_elements(elements)
        if len(flat_elements) != 4:
            raise ValueError("G_op requires exactly 4 elements.")
        
        # Check that each element is int, D, or Dsqrt2
        for element in flat_elements:
            if not isinstance(element, (int, D, Dsqrt2)):
                raise TypeError("Matrix elements must be of type int, D, or Dsqrt2.")
        
        self.elements = flat_elements

    def _flatten_elements(self, elements):
        """Flatten nested iterables to ensure exactly 4 elements."""
        if isinstance(elements, np.ndarray):
            elements = elements.flatten().tolist()
        return elements

    def __repr__(self) -> str:
        return f"[({self.elements[0]}) ({self.elements[1]})] [({self.elements[2]}) ({self.elements[3]})]"
    
    def __neg__(self) -> G_op:
        return G_op([-self.elements[0], -self.elements[1], -self.elements[2], -self.elements[3]])
    
    def det(self) -> Dsqrt2:
        return self.elements[0] * self.elements[3] - self.elements[1] * self.elements[2]
    
    def dag(self) -> G_op:
        return G_op([self.elements[0], self.elements[2], self.elements[1], self.elements[3]])
    
    def inv(self) -> G_op:
        determinant = self.det()
        if determinant == Dsqrt2(0, 0):
            return ValueError("Determinant must be non-zero")
        elif determinant == Dsqrt2(1, 0):
            return G_op([self.elements[3], -self.elements[1], -self.elements[2], self.elements[0]])
        else:
            return NotImplemented

    def __add__(self, other: int | D | Dsqrt2 | G_op) -> G_op:
        """Define matrix addition."""
        if not isinstance(other, G_op):
            return "Both elements must be 2x2 matrices"
        return G_op([self.elements[i] + other.elements[i] for i in range(4)])
    
    def __radd__(self, other: int | D | Dsqrt2 | G_op) -> G_op:
        return self.__add__(other)
    
    def __sub__(self, other: int | D | Dsqrt2 | G_op) -> G_op:
        """Define matrix subtraction."""
        return self.__add__(-other)
    
    def __rsub__(self, other: int | D | Dsqrt2 | G_op) -> G_op:
        return -self.__add__(other)

    def __mul__(self, other: int | D | Dsqrt2 | G_op) -> G_op:
        """Define matrix multiplication."""
        if isinstance(other, (int, D, Dsqrt2)):
            return G_op([other*self.elements[0], other*self.elements[1], other*self.elements[2], other*self.elements[3]])
        elif isinstance(other, G_op):
            return G_op([
                self.elements[0] * other.elements[0] + self.elements[1] * other.elements[2], self.elements[0] * other.elements[1] + self.elements[1] * other.elements[3],
                self.elements[2] * other.elements[0] + self.elements[3] * other.elements[2], self.elements[2] * other.elements[1] + self.elements[3] * other.elements[3]
            ])
        else:
            return TypeError("Product must be with G_op, Dsqrt2, D, or int")
        
    def __rmul__(self, other: int | D | Dsqrt2 | G_op) -> G_op:
        return self.__mul__(other)
    
    def __pow__(self, exponent: int) -> G_op:
        """Raise the matrix to an integer power."""
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer.")
        if exponent < 0:
            raise ValueError("Exponent must be non-negative.")
        
        result = G_op([1, 0, 0, 1])  # Start with identity

        # Multiply self with itself exponent times
        base = self
        for _ in range(exponent):
            result = result * base  # Uses the __mul__ method already defined

        return result

    def as_numpy_matrix(self):
        """Return as a 2x2 numpy matrix."""
        return np.matrix([[self.elements[0], self.elements[1]], [self.elements[2], self.elements[3]]])
    
I: G_op = G_op([1, 0, 0, 1])
    
R: G_op = G_op([Dsqrt2(0, D(1, 1)), -Dsqrt2(0, D(1, 1)), Dsqrt2(0, D(1, 1)), Dsqrt2(0, D(1, 1))])

K: G_op = G_op([Dsqrt2(-1, D(1, 1)), -Dsqrt2(0, D(1, 1)), Dsqrt2(1, D(1, 1)), Dsqrt2(0, D(1, 1))])

X: G_op = G_op([0, 1, 1, 0])

Z: G_op = G_op([1, 0, 0, -1])

A: G_op = G_op([1, -2, 0, 1])

B: G_op = G_op([1, Dsqrt2(0, 1), 0, 1])