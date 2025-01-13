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

from Rings import Zsqrt2, lamb, inv_lamb
from Grid_Operator import grid_operator
import numpy as np
import math

class state:
    """Class to initialize a state given a pair of 2X2 matrices. 

    A state is of the form (A, B), where A and B are both 2X2 matrices and have 
    a determinant of 1. This module is pulled from Appendix A of https://arxiv.org/pdf/1403.2975. 
    This module is useful in the context of achieving at least 1/6 uprightness 
    for both ellipses of the state. 

    Attributes:
        A (np.matrix): First matrix of the state.
        B (np.matrix): Second matrix of the state.
        z (float): Exponent of λ in A
        zeta (float): Exponent of λ in B
        e (float): Diagonal component of A
        epsilon (float): Diagonal component of B
        b (float): Antidiagonal component of A
        beta (float): Antidiagonal component of B
    """
    def __init__(self, A: np.matrix, B: np.matrix) -> None:
        """Initialize the state class.
        
        Args:
            A (np.matrix): First matrix of the class
            B (np.matrix): Second matrix of the class

        Raises:
            TypeError: If A or B cannot be converted to a numpy matrix.
            ValueError: If A or B are not 2x2 matrices.
            TypeError: If the elements of A or B cannot be converted to floats.
            ValueError: If A or B are not symmetric matrices.
        """
        
        # Attempt to convert A and B to numpy matrices
        try:
            A = np.asmatrix(A)
            B = np.asmatrix(B)
        except Exception:
            raise TypeError("A and B must be convertible to numpy matrices.")
    
        # Check that both matrices are 2x2
        if A.shape != (2, 2) or B.shape != (2, 2):
            raise ValueError("Both A and B must be 2x2 matrices.")
    
        # Ensure elements in A can be converted to float
        try:
            A = A.astype(float)
        except ValueError:
            raise TypeError("Elements in matrix A must be convertible to type float.")

        # Ensure elements in B can be converted to float
        try:
            B = B.astype(float)
        except ValueError:
            raise TypeError("Elements in matrix B must be convertible to type float.")
    
        # Check if A and B are symmetric
        if not np.isclose(A[0, 1], A[1, 0]):
            raise ValueError("Matrix A must be symmetric.")
        if not np.isclose(B[0, 1], B[1, 0]):
            raise ValueError("Matrix B must be symmetric.")
        
        self.A = A
        self.B = B
        self.__reduce()

    def __reduce(self) -> None:
        """Reduce both determinants to 1"""
        A = self.A
        detA = np.linalg.det(A)
        B = self.B
        detB = np.linalg.det(B)
        if np.isclose(detA, 1):
            self.A = A
        else:
            self.A = (1/detA)*A
        if np.isclose(detB, 1):
            self.B = B
        else:
            self.B = (1/detB)*B

    def __repr__(self) -> str:
        """Returns a string representation of the object"""
        return f"({self.A}, {self.B})"
    
    @property
    def z(self) -> float:
        A = self.A
        return -0.5*np.log(A[0, 0]/A[1, 1])/np.log(1 + np.sqrt(2))
    
    @property
    def zeta(self) -> float:
        B = self.B
        return -0.5*np.log(B[0, 0]/B[1, 1])/np.log(1 + np.sqrt(2))
    
    @property
    def e(self) -> float:
        z = self.z
        A = self.A
        return A[0,0] * (1 + np.sqrt(2)) ** z
    
    @property
    def epsilon(self) -> float:
        zeta = self.zeta
        B = self.B
        return B[0,0] * (1 + np.sqrt(2)) ** zeta
    
    @property
    def b(self) -> float:
        A = self.A
        return float(A[0, 1])
    
    @property
    def beta(self) -> float:
        B = self.B
        return float(B[0, 1])
    
    @property
    def skew(self) -> float:
        b = self.b
        beta = self.beta
        return b ** 2 + beta ** 2
    
    @property
    def bias(self) -> float:
        z = self.z
        zeta = self.zeta
        return zeta - z

    def transform(self, G: grid_operator) -> state:
        """Returns the action of a grid operator on a state"""
        A = self.A
        B = self.B
        if not isinstance(G, grid_operator):
            raise TypeError("G must be a grid operator")
        G_conj = G.conjugate()
        new_A = G.dag() * A * G
        new_B = G_conj.dag() * B * G_conj
        return state(new_A, new_B)
    
    def shift(self, k: int) -> grid_operator:
        """Returns the 'shift by k' action on a state"""
        A = self.A
        B = self.B
        if not isinstance(k, int):
            raise ValueError("k must be an integer")
        if k >= 0:
            sigma_k = (special_sigma ** k) * (math.sqrt(float(inv_lamb)) ** k)
            tau_k = (special_tau ** k) * (math.sqrt(float(inv_lamb)) ** k)
        else:
            sigma_k = (inv_special_sigma ** -k) * (math.sqrt(float(lamb)) ** -k)
            tau_k = (inv_special_tau ** -k) * (math.sqrt(float(lamb)) ** -k)
        shift_A = sigma_k * A * sigma_k
        shift_B = tau_k * B * tau_k
        return state(shift_A, shift_B)
    
special_sigma: grid_operator = grid_operator([lamb, 0, 0, 1])
inv_special_sigma: grid_operator = grid_operator([inv_lamb, 0, 0, 1])
special_tau: grid_operator = grid_operator([1, 0, 0, -lamb])
inv_special_tau: grid_operator = grid_operator([1, 0, 0, -inv_lamb])