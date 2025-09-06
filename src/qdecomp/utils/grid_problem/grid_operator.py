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

"""
This module defines the :class:`GridOperator`.

Grid operators are a central concept introduced in Section 5.3 of :cite:`grid_problem_ross`. However, this class is generalized to work with any matrix
whose elements lie in the ring :math:`D[\sqrt{2}]`.

To efficiently solve a general 2D grid problem, it is necessary to find the upright bounding box of the
regions of interest, since grid problems can only be solved on upright rectangular domains. It is important
to asses how closely this upright rectangle conforms to the shape inside it. The uprightness of an
arbitrary shape :math:`A` is defined as follows:

.. math::

    Up(A) = \\frac{\\text{area}(A)}{\\text{area}(BBox(A))}.

This is an important tool to evaluate how well a bounding box represents its corresponding shape.

Since the input error :math:`\\varepsilon` is small and the width of the slice :math:`\\mathcal{R}_\\epsilon`
is proportional to :math:`\\epsilon^2`, solving the grid problem on the bounding box of this slice is
extremely inefficient, because :math:`Up(\\mathcal{R}_\\varepsilon) \\to 0`. To solve this issue, grid
operators can be introduced.

A grid operator :math:`G` is such that for :math:`u \\in \\mathbb{Z}[\\omega]`, :math:`G(u) \\in \\mathbb{Z}[\\omega]`.
Moreover, if :math:`\\det(G) = \\pm 1`, then :math:`G` is called a *special grid operator*. Grid operators are
extremely useful for solving grid problems on arbitrary shapes, because if :math:`G` is a special grid operator,
then the problem

.. math::

    \\text{Find } G(u) \\in G(\\mathcal{R}_\\varepsilon) \\text{ such that } (G(u))^\\bullet \\in G^\\bullet (\\bar{\\mathcal{D}}),

is *equivalent* to the initial problem.

Furthermore, if we define :math:`E` as the smallest ellipse that encapsulates :math:`\\mathcal{R}_\\varepsilon`,
then there exists a special grid operator :math:`G`, such that:

.. math::

    Up(G(E)) \\geq \\frac{1}{6}, \\quad Up(G^\\bullet (\\bar{\\mathcal{D}})) \\geq \\frac{1}{6}.

This class defines grid operators, which will be useful in the grid problem algorithm as a whole.

For more information on the use of states, see Section 5.3 of :cite:`grid_problem_ross`.
"""

from __future__ import annotations

from typing import Union

import mpmath as mp
import numpy as np

from qdecomp.rings import *

__all__ = ["GridOperator", "I", "R", "K", "X", "Z", "A", "B"]


class GridOperator:
    """
    Class to represent a grid operator used in solving grid problems over the complex plane.

    A grid operator is a linear (though not continuous) transformation of the complex plane that preserves
    the grid structure defined by the ring :math:`\\mathbb{D}[\\omega]`. These operators are instrumental
    in improving the efficiency of grid problem algorithms.

    In particular, grid problems typically involve searching for solutions within narrow, non-rectangular
    regions of the complex plane. Since grid problems can only be solved efficiently over upright rectangular
    domains, applying a grid operator allows us to widen and reshape these regions into more tractable forms,
    thus enabling more effective enumeration of solutions.

    Grid operators are discussed in detail in Section 5.3 of :cite:`grid_problem_ross`.

    Parameters:
        G (np.ndarray): ndarray form of the grid operator
    """

    def __init__(self, G: np.ndarray) -> None:
        """Initialize the object with a 2x2 numpy array.

        Args:
            G (list | np.ndarray): A 4-element flat list, a 2x2 nested list, or a 2x2 numpy.ndarray
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
            elif len(G) == 2 and all(len(row) == 2 for row in G):
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
                raise TypeError(
                    f"Element {element} must be an int, D, Zsqrt2, or Dsqrt2. Got type {type(element)}."
                )

        # Convert to Dsqrt2
        G = np.vectorize(Dsqrt2.from_ring)(G)

        self.G = G
        self.a = G[0, 0]
        self.b = G[0, 1]
        self.c = G[1, 0]
        self.d = G[1, 1]

    def __repr__(self) -> str:
        """
        Return the string representation of the grid operator.

        Returns:
            str: A string representation of the grid operator matrix.
        """
        return str(self.G)

    def __neg__(self) -> GridOperator:
        """
        Return the negation of the grid operator.

        Returns:
            GridOperator: A new GridOperator instance with all elements negated.
        """
        return GridOperator(-self.G)

    def det(self) -> Union[int, D, Zsqrt2, Dsqrt2]:
        """
        Compute the determinant of the grid operator.

        Returns:
            int | D | Zsqrt2 | Dsqrt2: The determinant of the grid operator,
            depending on the types of its elements.
        """
        return self.a * self.d - self.b * self.c

    def dag(self) -> GridOperator:
        """
        Compute the hermitian conjugate of the grid operator.

        Returns:
            GridOperator: A new grid operator that is the hermitian conjugate of the current one.
        """
        # Since G is Real, the dag operation is the transpose operation
        return GridOperator([self.a, self.c, self.b, self.d])

    def conjugate(self):
        """Define the conjugation of the grid operator"""
        G = self.G
        G_conj = np.zeros_like(G, dtype=object)

        for i in range(2):  # Iterate over rows
            for j in range(2):  # Iterate over columns
                element = G[i, j]
                G_conj[i, j] = element.sqrt2_conjugate()  # Apply conjugation

        return GridOperator(G_conj)  # Return the conjugated grid

    def inv(self) -> GridOperator:
        """
        Return the inverse of the grid operator.

        The inversion is defined only for grid operators with determinant ±1.

        Returns:
            GridOperator: The inverse of the current grid operator.

        Raises:
            ValueError: If the determinant is 0, or not equal to ±1.
        """
        determinant = self.det()
        if determinant == 0:
            raise ValueError("Determinant must be non-zero")
        elif determinant == 1:
            return GridOperator([self.d, -self.b, -self.c, self.a])
        elif determinant == -1:
            return GridOperator([-self.d, self.b, self.c, -self.a])
        else:
            raise ValueError(
                "The inversion is not defined for grid operators with determinant different from -1 or 1"
            )

    def as_float(self) -> np.ndarray:
        """
        Return a float-valued approximation of the grid operator matrix.

        Returns:
            np.ndarray: A 2x2 NumPy array with float entries corresponding to the grid operator.
        """
        return np.array(self.G, dtype=float)

    def as_mpfloat(self) -> np.ndarray:
        """
        Return a high-precision float (mpmath) representation of the grid operator matrix.

        Returns:
            np.ndarray: A 2x2 NumPy array with entries converted to mpmath floating-point values.
        """
        return np.vectorize(lambda x: x.mpfloat())(self.G)

    def __add__(self, other: GridOperator) -> GridOperator:
        """
        Define the addition of two grid operators.

        Returns:
            GridOperator: The element-wise sum of the two grid operators.

        Raises:
            TypeError: If the operand is not an instance of GridOperator.
        """
        if not isinstance(other, GridOperator):
            raise "The elements must be grid operators"
        return GridOperator(self.G + other.G)

    def __sub__(self, other: GridOperator) -> GridOperator:
        """
        Define the subtraction operation of the grid operator.

        This is implemented by adding the negation of the other grid operator.

        Returns:
            GridOperator: The element-wise difference of the two grid operators.
        """
        return self + (-other)

    def __mul__(self, other: int | D | Zsqrt2 | Dsqrt2 | GridOperator) -> GridOperator:
        """
        Define the multiplication operation of the grid operator.

        This supports multiplication by scalars (int, D, Zsqrt2, Dsqrt2) or another grid operator.

        Returns:
            GridOperator: The result of the multiplication.

        Raises:
            TypeError: If the operand is not a valid type (int, D, Zsqrt2, Dsqrt2, or GridOperator).
        """
        if isinstance(other, (int, D, Zsqrt2, Dsqrt2)):
            num = Dsqrt2.from_ring(other)
            return GridOperator([num * self.a, num * self.b, num * self.c, num * self.d])
        elif isinstance(other, GridOperator):
            G = self.G
            G_p = other.G
            return GridOperator(G @ G_p)
        else:
            raise TypeError(
                "Product must be with a valid type (int, D, Zsqrt2, Dsqrt2) or GridOperator. Got {type(other)}."
            )

    def __rmul__(self, other: int | D | Zsqrt2 | Dsqrt2 | GridOperator) -> GridOperator:
        """
        Define the `right` multiplication operation of the grid operator.

        This supports multiplication by scalars (int, D, Zsqrt2, Dsqrt2) or another grid operator.

        Returns:
            GridOperator: The result of the multiplication.

        Raises:
            TypeError: If the operand is not a valid type (int, D, Zsqrt2, Dsqrt2, or GridOperator).
        """
        if isinstance(other, (int, D, Zsqrt2, Dsqrt2)):
            return self.__mul__(other)
        else:
            raise TypeError("Product must be with a valid type")

    def __pow__(self, exponent: int) -> GridOperator:
        """
        Define the exponentiation of the grid operator.

        This performs repeated multiplication of the grid operator by itself for the given exponent.

        Returns:
            GridOperator: The grid operator raised to the given exponent.

        Raises:
            TypeError: If the exponent is not an integer.
        """
        # Accept exponent if it is close to an integer, otherwise raise
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer.")

        if exponent < 0:
            base = self.inv()
            exp = -exponent
        else:
            base = self
            exp = exponent

        # Compute the power
        nth_power = base
        result = GridOperator([1, 0, 0, 1])  # Identity operator

        while exp:
            if exp & 1:
                result *= nth_power
            nth_power *= nth_power
            exp >>= 1

        return result


"""
This section is adapted from Appendix A in Ross et al. (2014).

To identify the appropriate grid operator for the algorithm, the following special grid 
operators are required:

.. math::

    R = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
        1 & -1 \\\\
        1 & 1
    \\end{pmatrix} \\quad
    A = \\begin{pmatrix}
        1 & -2 \\\\
        0 & 1
    \\end{pmatrix} \\quad
    B = \\begin{pmatrix}
        1 & \\sqrt{2} \\\\
        0 & 1
    \\end{pmatrix}

.. math::

    K = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
        -\\lambda^{-1} & -1 \\\\
        \\lambda & 1
    \\end{pmatrix} \\quad
    X = \\begin{pmatrix}
        0 & 1 \\\\
        1 & 0
    \\end{pmatrix} \\quad
    Z = \\begin{pmatrix}
        1 & 0 \\\\
        0 & -1
    \\end{pmatrix}

These operators can be combined in various orders to determine the grid operator needed for the algorithm.
As such, they are fundamental to the grid problem and essential for further computations in the algorithm.
"""

I: GridOperator = GridOperator([1, 0, 0, 1])

R: GridOperator = GridOperator(
    [
        Dsqrt2(D(0, 0), D(1, 1)),
        -Dsqrt2(D(0, 0), D(1, 1)),
        Dsqrt2(D(0, 0), D(1, 1)),
        Dsqrt2(D(0, 0), D(1, 1)),
    ]
)

K: GridOperator = GridOperator(
    [
        Dsqrt2(D(-1, 0), D(1, 1)),
        -Dsqrt2(D(0, 0), D(1, 1)),
        Dsqrt2(D(1, 0), D(1, 1)),
        Dsqrt2(D(0, 0), D(1, 1)),
    ]
)

X: GridOperator = GridOperator([0, 1, 1, 0])

Z: GridOperator = GridOperator([1, 0, 0, -1])

A: GridOperator = GridOperator([1, -2, 0, 1])

B: GridOperator = GridOperator([1, Zsqrt2(0, 1), 0, 1])
