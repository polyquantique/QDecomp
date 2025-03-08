# Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
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

from __future__ import annotations

from typing import Union

import numpy as np
import mpmath as mp
from cliffordplust.rings.rings import *

"""
This file defines the `Grid_Operator` class. Grid operators are defined in section 5.3 of 
https://arxiv.org/pdf/1403.2975, but this class applies to any matrix with elements inside 
the ring D[\u221a2]. Grid operators are a key component to solving the grid problem, as they 
are needed to bring the uprightness of the ellipses pair (represented by a state) to at least 1/6. 
"""

class Grid_Operator:
    """Class to initialize a grid operator.

    A grid operator is defined in detail in Section 5.3 of https://arxiv.org/pdf/1403.2975.
    This module uses the Rings.py module in order to properly define the ring elements of the grid operators,
    which are used to reduce the uprightness of the given ellipse pair.

    Attributes:
        G (np.ndarray): ndarray form of the grid operator
    """

    def __init__(self, G) -> None:
        """Initialize the object with a 2x2 numpy array.

        Args:
            G (list, np.ndarray): A 4-element flat list, a 2x2 nested list, or a 2x2 numpy.ndarray
            containing elements of type Dsqrt2.

        Raises:
            ValueError: If G is not a 4-element flat list, a 2x2 nested list, or a 2x2 array.
            TypeError: If elements of G are not of valid types.
            ValueError: If G is not 2x2 in size.
        """
        # Automatically convert input to np.ndarray if necessary
        if isinstance(G, list):
            if len(G) == 4:
                # Convert flat 4-element list to 2x2 ndarray
                G = np.array(G, dtype=object).reshape((2, 2))
            elif (
                len(G) == 2
                and all(len(row) == 2 for row in G)
            ):
                # Convert 2x2 nested list to ndarray
                G = np.array(G, dtype=object)
            else:
                raise ValueError(
                    "G must be a 4-element flat list or a 2x2 nested list with valid elements."
                )

        # Ensure G is a numpy ndarray
        if not isinstance(G, np.ndarray):
            raise TypeError("G must be a numpy ndarray or convertible to one.")

        # Validate shape
        if G.shape != (2, 2):
            raise ValueError(f"G must be of shape (2, 2). Got shape {G.shape}.")

        # Validate each element  
        for element in G.flatten():  
            if not isinstance(element, (int, D, Zsqrt2, Dsqrt2)):  
                raise TypeError(f"Element {element} must be an int, D, Zsqrt2, or Dsqrt2. Got type {type(element)}.")
        
        # Convert to Dsqrt2
        G = np.vectorize(Dsqrt2.from_ring)(G)
            
        self.G = G
        self.a = G[0, 0]
        self.b = G[0, 1]
        self.c = G[1, 0]
        self.d = G[1, 1]

    def __repr__(self) -> str:  
        """Define the string representation of the grid operator"""  
        return str(self.G)

    def __neg__(self) -> Grid_Operator:
        """Define the negation of the grid operator"""
        return Grid_Operator(-self.G)

    def det(self) -> Union[int, D, Zsqrt2, Dsqrt2]:
        """Computes the determinant of the grid operator"""
        return self.a * self.d - self.b * self.c

    def dag(self) -> Grid_Operator:
        """Define the dag operation of the grid operator"""
        # Since G is Real, the dag operation is the transpose operation
        return Grid_Operator([self.a, self.c, self.b, self.d])

    def conjugate(self):
        """Define the conjugation of the grid operator"""
        G = self.G
        G_conj = np.zeros_like(G, dtype=object)

        for i in range(2):  # Iterate over rows
            for j in range(2):  # Iterate over columns
                element = G[i, j]
                if isinstance(element, (Zsqrt2, Dsqrt2)):
                    G_conj[i, j] = element.sqrt2_conjugate()  # Apply conjugation
                else:
                    G_conj[i, j] = element  # No change for int or D types

        return Grid_Operator(G_conj)  # Return the conjugated grid

    def inv(self) -> Grid_Operator:
        """Define the inversion of the grid operator"""
        determinant = self.det()
        if determinant == 0:
            raise ValueError("Determinant must be non-zero")
        elif determinant == 1:
            return Grid_Operator([self.d, -self.b, -self.c, self.a])
        elif determinant == -1:
            return Grid_Operator([-self.d, self.b, self.c, -self.a])
        else:
            raise ValueError(
                "The inversion is not defined for grid operators with determinant different from -1 or 1"
            )

    def as_float(self) -> np.ndarray:
        return np.array(self.G, dtype=float)
    
    def as_mpmath(self) -> np.ndarray:
        return np.vectorize(lambda x: x.mpfloat())(self.G)

    def __add__(self, other: Grid_Operator) -> Grid_Operator:
        """Define the summation operation of the grid operator"""
        if not isinstance(other, Grid_Operator):
            raise "The elements must be grid operators"
        return Grid_Operator(self.G + other.G)

    def __sub__(self, other: Grid_Operator) -> Grid_Operator:
        """Define the subtraction operation of the grid operator"""
        return self + (-other)

    def __mul__(self, other: int | D | Zsqrt2 | Dsqrt2 | Grid_Operator) -> Grid_Operator:
        """Define the multiplication operation of the grid operator"""
        if isinstance(other, (int, D, Zsqrt2, Dsqrt2)):
            return Grid_Operator([other * self.a, other * self.b, other * self.c, other * self.d])
        elif isinstance(other, Grid_Operator):
            G = self.G
            G_p = other.G
            return Grid_Operator(G @ G_p)
        else:
            raise TypeError("Product must be with a valid type")

    def __rmul__(self, other: int | D | Zsqrt2 | Dsqrt2 | Grid_Operator) -> Grid_Operator:
        """Define the right multiplication operation of the grid operator"""
        if isinstance(other, (int, D, Zsqrt2, Dsqrt2)):
            return self.__mul__(other)
        elif isinstance(other, Grid_Operator):
            G = self.G
            G_p = other.G
            return Grid_Operator(G_p @ G)
        else:
            raise TypeError("Product must be with a valid type")

    def __pow__(self, exponent: int) -> Grid_Operator:
        """Define the exponentiation of the grid operator"""
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer.")
        if exponent < 0:
            base = self.inv()
        else:
            base = self

        result = Grid_Operator([1, 0, 0, 1])  # Start with identity

        for _ in range(abs(exponent)):
            result = result * base  # Uses the __mul__ method already defined

        return result


I: Grid_Operator = Grid_Operator([1, 0, 0, 1])

R: Grid_Operator = Grid_Operator(
    [
        Dsqrt2(D(0, 0), D(1, 1)),
        -Dsqrt2(D(0, 0), D(1, 1)),
        Dsqrt2(D(0, 0), D(1, 1)),
        Dsqrt2(D(0, 0), D(1, 1)),
    ]
)

K: Grid_Operator = Grid_Operator(
    [
        Dsqrt2(D(-1, 0), D(1, 1)),
        -Dsqrt2(D(0, 0), D(1, 1)),
        Dsqrt2(D(1, 0), D(1, 1)),
        Dsqrt2(D(0, 0), D(1, 1)),
    ]
)

X: Grid_Operator = Grid_Operator([0, 1, 1, 0])

Z: Grid_Operator = Grid_Operator([1, 0, 0, -1])

A: Grid_Operator = Grid_Operator([1, -2, 0, 1])

B: Grid_Operator = Grid_Operator([1, Zsqrt2(0, 1), 0, 1])
