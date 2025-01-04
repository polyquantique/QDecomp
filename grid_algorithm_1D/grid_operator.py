from __future__ import annotations

import numpy as np
from Rings import D, Zsqrt2, Dsqrt2, Domega, lamb, inv_lamb
from typing import Union

class grid_operator:
    """
    
    """
    def __init__(self, G) -> None:
        # Automatically convert input to np.matrix if necessary
        if isinstance(G, list):
            if len(G) == 4 and all(isinstance(e, (int, D, Zsqrt2, Dsqrt2)) for e in G):
                # Convert flat 4-element list to 2x2 matrix
                G = np.matrix([[G[0], G[1]], [G[2], G[3]]], dtype=object)
            elif (
                len(G) == 2 
                and all(len(row) == 2 for row in G)
                and all(isinstance(e, (int, D, Zsqrt2, Dsqrt2)) for row in G for e in row)
            ):
                # Convert 2x2 nested list to matrix
                G = np.matrix(G, dtype=object)
            else:
                raise ValueError(
                    "G must be a 4-element flat list or a 2x2 nested list with valid elements."
                )

        # Ensure G is a numpy matrix
        if not isinstance(G, np.matrix):
            raise TypeError("G must be a numpy matrix or convertible to one.")

        # Validate shape
        if G.shape != (2, 2):
            raise ValueError("G must be 2x2 in size.")

        # Validate each element
        for element in G.flatten().tolist()[0]:
            if not isinstance(element, (int, D, Zsqrt2, Dsqrt2)):
                raise TypeError(f"Element {element} must be an int, D, Zsqrt2, or Dsqrt2.")

        self.G = G
        self.a = G[0, 0]
        self.b = G[0, 1]
        self.c = G[1, 0]
        self.d = G[1, 1]

    def __repr__(self) -> str:
        G = self.G
        return f"{G}"
    
    def __neg__(self) -> grid_operator:
        G = self.G
        return grid_operator(-G)
    
    def det(self) -> Union[int, D, Zsqrt2, Dsqrt2]:
        G = self.G
        det_G = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
        return det_G
    
    def dag(self) -> grid_operator:
        G = self.G
        return np.transpose(G)
    
    def conjugate(self):
        G = self.G
        # Initialize G_conj with the same shape as G, using np.array for flexibility
        G_conj = np.zeros_like(G, dtype=object)  # or dtype=G.dtype if you need consistency in types

        for i in range(G.shape[0]):  # Iterate over rows
            for j in range(G.shape[1]):  # Iterate over columns
                element = G[i, j]
                if isinstance(element, (int, D)):
                    G_conj[i, j] = element  # No change for integers or D types
                elif isinstance(element, (Zsqrt2, Dsqrt2)):
                    G_conj[i, j] = element.sqrt2_conjugate()  # Apply conjugation
                else:
                    raise TypeError(f"Invalid type at G[{i}, {j}]: {type(element)}")
    
        return grid_operator(G_conj)  # Return the conjugated grid
    
    def inv(self) -> grid_operator:
        determinant = self.det()
        if determinant == 0:
            return ValueError("Determinant must be non-zero")
        elif determinant == 1:
            return grid_operator([self.d, -self.b, -self.c, self.a])
        elif determinant == -1:
            return grid_operator([-self.d, self.b, self.c, -self.a])
        else:
            return NotImplemented
        
    def __add__(self, other: grid_operator) -> grid_operator:
        """"""
        if not isinstance(other, grid_operator):
            return "Both elements must be grid operators"
        G = self.G
        G_p = other.G
        return grid_operator(G + G_p)
    
    def __radd__(self, other: grid_operator) -> grid_operator:
        return self.__add__(other)
    
    def __sub__(self, other: grid_operator) -> grid_operator:
        """"""
        return self.__add__(-other)
    
    def __rsub__(self, other: grid_operator) -> grid_operator:
        return -self.__add__(other)

    def __mul__(self, other: int | D | Dsqrt2 | grid_operator | np.matrix | float) -> grid_operator | np.matrix:
        """"""
        G = self.G
        if isinstance(other, (int, D, Dsqrt2)):
            return grid_operator(other * G)
        elif isinstance(other, float):
            return other * G
        elif isinstance(other, grid_operator):
            G_p = other.G
            return grid_operator(G * G_p)
        elif isinstance(other, np.matrix):
            return G * other
        else:
            raise TypeError("Product must be with a valid type")
        
    def __rmul__(self, other: int | D | Dsqrt2 | grid_operator | np.matrix | float) -> grid_operator | np.matrix:
        """"""
        G = self.G
        if isinstance(other, (float, int, D, Dsqrt2)):
            return self.__mul__(other)
        elif isinstance(other, grid_operator):
            G_p = other.G
            return grid_operator(G_p * G)
        elif isinstance(other, np.matrix):
            return other * G
        else:
            raise TypeError("Product must be with a valid type")
    
    def __pow__(self, exponent: int) -> grid_operator:
        """"""
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer.")
        if exponent < 0:
            base = self.inv()
        else:
            base = self

        result = grid_operator([1, 0, 0, 1])  # Start with identity

        for _ in range(abs(exponent)):
            result = result * base  # Uses the __mul__ method already defined

        return result
    
I: grid_operator = grid_operator([1, 0, 0, 1])
    
R: grid_operator = grid_operator([Dsqrt2(D(0, 0), D(1, 1)), -Dsqrt2(D(0, 0), D(1, 1)), Dsqrt2(D(0, 0), D(1, 1)), Dsqrt2(D(0, 0), D(1, 1))])

K: grid_operator = grid_operator([Dsqrt2(D(-1, 0), D(1, 1)), -Dsqrt2(D(0, 0), D(1, 1)), Dsqrt2(D(1, 0), D(1, 1)), Dsqrt2(D(0, 0), D(1, 1))])

X: grid_operator = grid_operator([0, 1, 1, 0])

Z: grid_operator = grid_operator([1, 0, 0, -1])

A: grid_operator = grid_operator([1, -2, 0, 1])

B: grid_operator = grid_operator([1, Zsqrt2(0, 1), 0, 1])
