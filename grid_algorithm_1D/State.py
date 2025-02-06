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
This file defines the `State` class, which is a key component in solving the grid problem. 
Specifically, it is used to ensure that the uprightness of the ellipse pair is augmented to 
at least 1/6. 

The `State` class depends on the `Grid_Operator` class, which is defined in the `grid_operator.py` file. 
The primary operations that can be performed on a state, along with the detailed definition and 
usage of a state, are explained in Appendix A of the paper: https://arxiv.org/pdf/1403.2975.
"""

from __future__ import annotations

import math

import numpy as np
from grid_operator import Grid_Operator
from Rings import Zsqrt2, inv_lamb, lamb


class State:
    """Class to initialize a state given a pair of 2x2 matrices.

    A state is of the form (A, B), where A and B are both 2x2 matrices and have
    a determinant of 1. A state is assigned to a pair of ellipses with the matrices
    A and B representing their dimensions and angles. This module is pulled from
    Appendix A of https://arxiv.org/pdf/1403.2975. This module is useful in the
    context of achieving at least 1/6 uprightness for both ellipses of the state.

    Attributes:
        A (np.ndarray): First matrix of the state.
        B (np.ndarray): Second matrix of the state.
        z (float): Exponent of \u03BB in A.
        zeta (float): Exponent of \u03BB in B.
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
            TypeError: If the elements of A or B cannot be converted to floats.
            ValueError: If A or B are not 2x2 matrices.
            ValueError: If A or B are not symmetric matrices.
        """
        # Ensure A and B are numpy arrays
        try:
            A = np.array(A, dtype=float)
            B = np.array(B, dtype=float)
        except Exception:
            raise TypeError("A and B must be convertible to numpy arrays of floats.")

        # Check that both matrices are 2x2
        if A.shape != (2, 2) or B.shape != (2, 2):
            raise ValueError("Both A and B must be 2x2 matrices.")

        # Check if A and B are symmetric
        if not np.isclose(A[0, 1], A[1, 0]):
            raise ValueError("Matrix A must be symmetric.")
        if not np.isclose(B[0, 1], B[1, 0]):
            raise ValueError("Matrix B must be symmetric.")

        # Assign the matrices to attributes
        self.A = A
        self.B = B

        # Normalize the determinants of A and B to 1
        self.__reduce()

    def __reduce(self) -> None:
        """Reduce both determinants to 1"""
        A = self.A
        detA = np.linalg.det(A)
        B = self.B
        detB = np.linalg.det(B)
        if np.isclose(detA, 0) or np.isclose(detB, 0) or detA < 0 or detB < 0:
            raise ValueError("The determinant of A and B must be positive and non-zero")
        if np.isclose(detA, 1):
            pass
        else:
            self.A = (1 / math.sqrt(detA)) * A
        if np.isclose(detB, 1):
            pass
        else:
            self.B = (1 / math.sqrt(detB)) * B

    def __repr__(self) -> str:
        """Returns a string representation of the object"""
        return f"({self.A}, {self.B})"

    @property
    def z(self) -> float:
        """Refer to (34) in https://arxiv.org/pdf/1403.2975"""
        A = self.A
        return -0.5 * math.log(A[0, 0] / A[1, 1]) / math.log(1 + math.sqrt(2))

    @property
    def zeta(self) -> float:
        """Refer to (34) in https://arxiv.org/pdf/1403.2975"""
        B = self.B
        return -0.5 * math.log(B[0, 0] / B[1, 1]) / math.log(1 + math.sqrt(2))

    @property
    def e(self) -> float:
        """Refer to (34) in https://arxiv.org/pdf/1403.2975"""
        z = self.z
        A = self.A
        return A[0, 0] * (1 + math.sqrt(2)) ** z

    @property
    def epsilon(self) -> float:
        """Refer to (34) in https://arxiv.org/pdf/1403.2975"""
        zeta = self.zeta
        B = self.B
        return B[0, 0] * (1 + math.sqrt(2)) ** zeta

    @property
    def b(self) -> float:
        """Refer to (34) in https://arxiv.org/pdf/1403.2975"""
        A = self.A
        return float(A[0, 1])

    @property
    def beta(self) -> float:
        """Refer to (34) in https://arxiv.org/pdf/1403.2975"""
        B = self.B
        return float(B[0, 1])

    @property
    def skew(self) -> float:
        """Refer to (34) in https://arxiv.org/pdf/1403.2975"""
        return self.b**2 + self.beta**2

    @property
    def bias(self) -> float:
        """Refer to (34) in https://arxiv.org/pdf/1403.2975"""
        return self.zeta - self.z

    def transform(self, G: Grid_Operator) -> State:
        """Computes the action of a grid operator on a state"""
        A = self.A
        B = self.B
        if not isinstance(G, Grid_Operator):
            raise TypeError("G must be a grid operator")
        G_conj = G.conjugate()
        new_A = G.dag().as_float() @ A @ G.as_float()
        new_B = G_conj.dag().as_float() @ B @ G_conj.as_float()
        return State(new_A, new_B)

    def shift(self, k: int) -> State:
        """Computes the 'shift by k' operation on a state"""
        A = self.A
        B = self.B
        if not isinstance(k, int):
            raise ValueError("k must be an integer")
        if k >= 0:
            # kth power of sigma
            sigma_k = (special_sigma**k).as_float() * math.sqrt(float(inv_lamb**k))
            # kth power of tau
            tau_k = (special_tau**k).as_float() * math.sqrt(float(inv_lamb**k))
        else:
            # Since k is negative, we have to take the inverse
            sigma_k = (inv_special_sigma**-k).as_float() * (math.sqrt(float(lamb)) ** -k)
            tau_k = (inv_special_tau**-k).as_float() * (math.sqrt(float(lamb)) ** -k)
        shift_A = sigma_k @ A @ sigma_k
        shift_B = tau_k @ B @ tau_k
        return State(shift_A, shift_B)


"""Important grid operators

In the definitions of sigma and tau, matrices are involved. When applying the shift, 
sigma and tau often need to be raised to an integer power. To ensure precision, it is 
preferable to compute the matrix exponentiation exactly, rather than approximating. 

After computing the exponentiation, the result is multiplied by the square root of lambda 
raised to the corresponding integer power. 
"""

special_sigma: Grid_Operator = Grid_Operator([lamb, 0, 0, 1])
inv_special_sigma: Grid_Operator = Grid_Operator([inv_lamb, 0, 0, 1])
special_tau: Grid_Operator = Grid_Operator([1, 0, 0, -lamb])
inv_special_tau: Grid_Operator = Grid_Operator([1, 0, 0, -inv_lamb])
