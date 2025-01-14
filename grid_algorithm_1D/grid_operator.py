# Copyright 2022-2023 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
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
from Rings import D, Domega, Dsqrt2, Zsqrt2, inv_lamb, lamb


class grid_operator:
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
            G: A 4-element flat list, a 2x2 nested list, or a 2x2 numpy.ndarray
            containing elements of type int, D, Zsqrt2, or Dsqrt2.

        Raises:
            ValueError: If G is not a 4-element flat list, a 2x2 nested list, or a 2x2 array.
            TypeError: If elements of G are not of valid types.
            ValueError: If G is not 2x2 in size.
        """
        # Automatically convert input to np.ndarray if necessary
        if isinstance(G, list):
            if len(G) == 4 and all(isinstance(e, (int, D, Zsqrt2, Dsqrt2)) for e in G):
                # Convert flat 4-element list to 2x2 ndarray
                G = np.array([[G[0], G[1]], [G[2], G[3]]], dtype=object)
            elif (
                len(G) == 2
                and all(len(row) == 2 for row in G)
                and all(isinstance(e, (int, D, Zsqrt2, Dsqrt2)) for row in G for e in row)
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
            raise ValueError("G must be 2x2 in size.")

        # Validate each element
        for element in G.flatten():
            if not isinstance(element, (int, D, Zsqrt2, Dsqrt2)):
                raise TypeError(f"Element {element} must be an int, D, Zsqrt2, or Dsqrt2.")

        # Assign attributes
        self.G = G

    def __repr__(self) -> str:
        """Define the string representation of the grid operator"""
        G = self.G
        return f"{G}"

    def __neg__(self) -> grid_operator:
        """Define the negation of the grid operator"""
        G = self.G
        return grid_operator(-G)

    def det(self) -> Union[int, D, Zsqrt2, Dsqrt2]:
        """Computes the determinant of the grid operator"""
        G = self.G
        return G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]

    def dag(self) -> grid_operator:
        """Define the dag operation of the grid operator"""
        G = self.G
        # Since G is Real, the dag operation is the transpose operation
        return np.transpose(G)

    def conjugate(self):
        """Define the conjugation of the grid operator"""
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
        """Define the inversion of the grid operator"""
        determinant = self.det()
        if determinant == 0:
            raise ValueError("Determinant must be non-zero")
        elif determinant == 1:
            return grid_operator([self.d, -self.b, -self.c, self.a])
        elif determinant == -1:
            return grid_operator([-self.d, self.b, self.c, -self.a])
        else:
            raise ValueError("The inversion is not defined for grid operators with determinant different from -1 or 1")

    def __add__(self, other: grid_operator | np.matrix) -> grid_operator | np.matrix:
        """Define the summation operation of the grid operator"""
        if not isinstance(other, (grid_operator, np.matrix)):
            return "Both elements must be grid operators or numpy matrices"
        G = self.G
        if isinstance(other, np.matrix):
            G_p = np.matrix([[float(other.a), float(other.b)], [float(other.c), float(other.d)]])
            return G + G_p
        else:
            G_p = other.G
            return grid_operator(G + G_p)

    def __radd__(self, other: grid_operator | np.matrix) -> grid_operator | np.matrix:
        """Define the right summation operation of the grid operator"""
        return self.__add__(other)

    def __sub__(self, other: grid_operator | np.matrix) -> grid_operator | np.matrix:
        """Define the substraction operation of the grid operator"""
        return self.__add__(-other)

    def __rsub__(self, other: grid_operator | np.matrix) -> grid_operator | np.matrix:
        """Define the right substraction operation of the grid operator"""
        return -self.__add__(other)

    def __mul__(
        self, other: int | D | Zsqrt2 | Dsqrt2 | grid_operator | np.matrix | float
    ) -> grid_operator | np.matrix:
        """Define the multiplication operation of the grid operator"""
        G = self.G
        if isinstance(other, (int, D, Zsqrt2, Dsqrt2)):
            return grid_operator(other * G)
        elif isinstance(other, (float, np.matrix)):
            float_G = np.matrix([[float(self.a), float(self.b)], [float(self.c), float(self.d)]])
            return float_G * other
        elif isinstance(other, grid_operator):
            G_p = other.G
            return grid_operator(G * G_p)
        else:
            raise TypeError("Product must be with a valid type")

    def __rmul__(
        self, other: int | D | Dsqrt2 | grid_operator | np.matrix | float
    ) -> grid_operator | np.matrix:
        """Define the right multiplication operation of the grid operator"""
        G = self.G
        if isinstance(other, (int, D, Zsqrt2, Dsqrt2)):
            return self.__mul__(other)
        elif isinstance(other, grid_operator):
            G_p = other.G
            return grid_operator(G_p * G)
        elif isinstance(other, (float, np.matrix)):
            float_G = np.matrix([[float(self.a), float(self.b)], [float(self.c), float(self.d)]])
            return other * float_G
        else:
            raise TypeError("Product must be with a valid type")

    def __pow__(self, exponent: int) -> grid_operator:
        """Define the exponentiation of the grid operator"""
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

R: grid_operator = grid_operator(
    [
        Dsqrt2(D(0, 0), D(1, 1)),
        -Dsqrt2(D(0, 0), D(1, 1)),
        Dsqrt2(D(0, 0), D(1, 1)),
        Dsqrt2(D(0, 0), D(1, 1)),
    ]
)

K: grid_operator = grid_operator(
    [
        Dsqrt2(D(-1, 0), D(1, 1)),
        -Dsqrt2(D(0, 0), D(1, 1)),
        Dsqrt2(D(1, 0), D(1, 1)),
        Dsqrt2(D(0, 0), D(1, 1)),
    ]
)

X: grid_operator = grid_operator([0, 1, 1, 0])

Z: grid_operator = grid_operator([1, 0, 0, -1])

A: grid_operator = grid_operator([1, -2, 0, 1])

B: grid_operator = grid_operator([1, Zsqrt2(0, 1), 0, 1])
