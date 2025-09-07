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
This file defines the :class:`State` class, which is a key component in solving the grid problem.
Specifically, it is used to ensure that the uprightness of the ellipse pair is augmented to
at least 1/6.

For more information on the use of states, see appendix A of :cite:`grid_problem_ross`.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from qdecomp.rings import *
from qdecomp.utils.grid_problem.grid_operator import GridOperator

SQRT2 = mp.sqrt(2)

__all__ = [
    "State",
    "special_sigma",
    "inv_special_sigma",
    "special_tau",
    "inv_special_tau",
]

"""
.. note::

    In the definitions of :math:`\\sigma` and :math:`\\tau`, matrix exponentiation is involved.
    When applying the shift, these matrices are often raised to an integer power. To ensure
    numerical precision, it is preferable to compute the matrix exponentiation exactly,
    rather than relying on approximations.

    After exponentiating, the result should be multiplied by the square root of
    :math:`\\lambda` raised to the corresponding integer power.
"""

special_sigma: GridOperator = GridOperator([LAMBDA, 0, 0, 1])
inv_special_sigma: GridOperator = GridOperator([INVERSE_LAMBDA, 0, 0, 1])
special_tau: GridOperator = GridOperator([1, 0, 0, -LAMBDA])
inv_special_tau: GridOperator = GridOperator([1, 0, 0, -INVERSE_LAMBDA])


class State:
    """
    Class to initialize a state given a pair of 2×2 matrices.

    A state is of the form :math:`(A, B)`, where both :math:`A` and :math:`B` are
    2×2 real matrices with determinant 1. These matrices correspond to ellipses,
    with the matrices encoding the dimensions and orientation of each ellipse.

    This class is based on Appendix A of :cite:`grid_problem_ross`,
    and is used in the context of achieving at least 1/6 uprightness for both ellipses
    associated with a state.

    Parameters:
        A (np.ndarray): First matrix of the state.
        B (np.ndarray): Second matrix of the state.
        z (float): Exponent of :math:`\\lambda` in A.
        zeta (float): Exponent of :math:`\\lambda` in B.
        e (float): Diagonal component of A.
        epsilon (float): Diagonal component of B.
        b (float): Antidiagonal component of A.
        beta (float): Antidiagonal component of B.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray) -> None:
        """Initialize the state class.

        Args:
            A (np.ndarray): First matrix of the class.
            B (np.ndarray): Second matrix of the class.

        Raises:
            TypeError: If A or B cannot be converted to a numpy array.
            TypeError: If the elements of A or B are not mp.mpf.
            ValueError: If A or B are not 2x2 matrices.
            ValueError: If A or B are not symmetric matrices.
        """
        # Ensure A and B are numpy arrays
        A = np.array(A, dtype=object)
        B = np.array(B, dtype=object)

        # Check that both matrices are 2x2
        if A.shape != (2, 2) or B.shape != (2, 2):
            raise ValueError("Both A and B must be 2x2 matrices.")

        # Ensure that A and B contain mp.mpf
        if not np.all(np.vectorize(lambda x: isinstance(x, mp.mpf))(A)):
            raise TypeError("The elements of A must be mp.mpf")

        if not np.all(np.vectorize(lambda x: isinstance(x, mp.mpf))(B)):
            raise TypeError("The elements of B must be mp.mpf")

        # Check if A and B are symmetric
        if not np.isclose(float(A[0, 1]), float(A[1, 0])):
            raise ValueError("Matrix A must be symmetric.")
        if not np.isclose(float(B[0, 1]), float(B[1, 0])):
            raise ValueError("Matrix B must be symmetric.")

        # Assign the matrices to attributes
        self.A = A
        self.B = B

        # Normalize the determinants of A and B to 1
        self.__reduce()

    def __reduce(self) -> None:
        """Reduce both determinants to 1"""
        A = self.A
        detA = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        B = self.B
        detB = B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0]

        if detA <= 0 or detB <= 0:
            raise ValueError("The determinant of A and B must be positive and non-zero")

        if mp.almosteq(1, detA):
            pass
        else:
            self.A = (1 / mp.sqrt(detA)) * A

        if mp.almosteq(1, detB):
            pass
        else:
            self.B = (1 / mp.sqrt(detB)) * B

    def __repr__(self) -> str:
        """Returns a string representation of the object"""
        return f"({self.A}, {self.B})"

    @property
    def z(self) -> float:
        """
        Exponent of :math:`\\lambda` and :math:`\\lambda^{-1}` for the first matrix of the state.

        Returns:
            float: The exponent :math:`z` computed from the diagonal entries of matrix :math:`A`.
        """
        return -0.5 * mp.log(self.A[0, 0] / self.A[1, 1]) / mp.log(1 + SQRT2)

    @property
    def zeta(self) -> float:
        """
        Exponent of :math:`\\lambda` and :math:`\\lambda^{-1}` for the second matrix of the state.

        Returns:
            float: The exponent :math:`z` computed from the diagonal entries of matrix :math:`B`.
        """
        return -0.5 * mp.log(self.B[0, 0] / self.B[1, 1]) / mp.log(1 + SQRT2)

    @property
    def e(self) -> float:
        """
        Diagonal elements of the first matrix of the state.

        Returns:
            float: The diagonal component :math:`e` of matrix :math:`A`.
        """
        return self.A[0, 0] * (1 + SQRT2) ** self.z

    @property
    def epsilon(self) -> float:
        """
        Diagonal elements of the second matrix of the state.

        Returns:
            float: The diagonal component :math:`\\epsilon` of matrix :math:`B`.
        """
        return self.B[0, 0] * (1 + SQRT2) ** self.zeta

    @property
    def b(self) -> float:
        """
        Anti-diagonal elements of the first matrix of the state.

        Returns:
            float: The anti-diagonal component :math:`b` of matrix :math:`A`.
        """
        return self.A[0, 1]

    @property
    def beta(self) -> float:
        """
        Anti-diagonal elements of the second matrix of the state.

        Returns:
            float: The anti-diagonal component :math:`\\beta` of matrix :math:`B`.
        """
        return self.B[0, 1]

    @property
    def skew(self) -> float:
        """
        Compute the skew of the state.

        The skew measures how upright the ellipses of the state are.
        A higher skew indicates lower uprightness.

        Returns:
            float: The skew value :math:`b^2 + \\beta^2`.
        """
        return self.b**2 + self.beta**2

    @property
    def bias(self) -> float:
        """
        Compute the bias of the state.

        The bias measures how difficult it is to reduce the skew.
        To reduce the skew in a quantifiable manner, the bias must be between -1 and 1.

        Returns:
            float: The bias value :math:`\\zeta - z`.
        """
        return self.zeta - self.z

    def transform(self, G: GridOperator) -> State:
        """
        Apply a grid operator to the state.

        This computes the transformation of both ellipses in the state
        under the action of a given grid operator :math:`G`. The operator
        acts on the first matrix :math:`A` and its conjugate acts on the second
        matrix :math:`B`.

        Parameters:
            G (GridOperator): The grid operator to apply to the state.

        Returns:
            State: The transformed state after applying the grid operator.
        """
        if not isinstance(G, GridOperator):
            raise TypeError("G must be a grid operator")
        G_conj = G.conjugate()
        new_A = (G.dag()).as_mpfloat() @ self.A @ G.as_mpfloat()
        new_B = (G_conj.dag()).as_mpfloat() @ self.B @ G_conj.as_mpfloat()
        return State(new_A, new_B)

    def shift(self, k: int) -> State:
        """
        Apply the shift operation by an integer :math:`k` to the state.

        This operation adjusts the bias of the state while preserving its skew. It is
        used to bring the bias into a desired range (e.g., between -1 and 1) without
        affecting how upright the ellipses are.

        Parameters:
            k (int): The shift amount. A positive value increases the bias,
                    while a negative value decreases it.

        Returns:
            State: The shifted state with the same skew and adjusted bias.

        Raises:
            ValueError: If `k` is not an integer.
        """

        # Accept k if it is very close to an integer, otherwise raise
        if not isinstance(k, int):
            raise ValueError("exponent must be an integer")

        if k >= 0:
            # kth power of sigma
            sigma_k = (special_sigma**k).as_mpfloat() * mp.sqrt((INVERSE_LAMBDA**k).mpfloat())
            # kth power of tau
            tau_k = (special_tau**k).as_mpfloat() * mp.sqrt((INVERSE_LAMBDA**k).mpfloat())

        else:
            # Since k is negative, we have to take the inverse
            sigma_k = (inv_special_sigma**-k).as_mpfloat() * mp.sqrt((LAMBDA**-k).mpfloat())
            tau_k = (inv_special_tau**-k).as_mpfloat() * mp.sqrt((LAMBDA**-k).mpfloat())

        shift_A = sigma_k @ self.A @ sigma_k
        shift_B = tau_k @ self.B @ tau_k

        return State(shift_A, shift_B)
