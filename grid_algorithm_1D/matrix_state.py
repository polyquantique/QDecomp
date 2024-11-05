from __future__ import annotations

from Dsqrt2_2x2matrix import G_op, Z
from Dsqrt2 import D, Dsqrt2, lamb, inv_lamb
import numpy as np
import math

class matrix_state:
    """Class to do operations with matrix states

    Attributes
    """
    def __init__(self, A: np.matrix, B: np.matrix) -> None:
        """Initialize the matrix_state class.
        
        Args:

        Raises:
        """
        # Ensure A and B are numpy matrices
        if not (isinstance(A, np.matrix) and isinstance(B, np.matrix)):
            raise TypeError("A and B must be of type np.matrix")
        
        # Check that both matrices are 2x2
        if A.shape != (2, 2) or B.shape != (2, 2):
            raise ValueError("Both A and B must be 2x2 matrices.")
        
        # Check that all elements in A can be converted to float, without converting if they are integers
        if not np.issubdtype(A.dtype, np.integer):
            try:
                A = A.astype(float)
            except ValueError:
                raise TypeError("Elements in matrix A must be convertible to type float.")

        # Check that all elements in B can be converted to float, without converting if they are integers
        if not np.issubdtype(B.dtype, np.integer):
            try:
                B = B.astype(float)
            except ValueError:
                raise TypeError("Elements in matrix B must be convertible to type float.")
        
        # Check if A and B are symmetric by verifying that A[0, 1] == A[1, 0] and B[0, 1] == B[1, 0]
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
        return f"({self.A}, {self.B})"
    
    @property
    def z(self) -> float:
        A = self.A
        return -0.5*np.log(A[0, 0]/A[1, 1])/np.log(1 + np.sqrt(2))
    
    @property
    def gamma(self) -> float:
        B = self.B
        return -0.5*np.log(B[0, 0]/B[1, 1])/np.log(1 + np.sqrt(2))
    
    @property
    def e(self) -> float:
        z = self.z
        A = self.A
        return A[0,0] * (1 + np.sqrt(2)) ** z
    
    @property
    def epsilon(self) -> float:
        z = self.gamma
        B = self.B
        return B[0,0] * (1 + np.sqrt(2)) ** z
    
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
        gamma = self.gamma
        return gamma - z
    
    def transform(self, G: G_op) -> matrix_state:
        """To do: error message if wrong type, and doc strings"""
        A = self.A
        B = self.B
        G_conj = G.conjugate()
        new_A = G.dag() * A * G
        new_B = G_conj.dag() * B * G_conj
        state = matrix_state(new_A, new_B)
        return state
    
    def shift(self, k: int) -> matrix_state:
        """To do: error message if wrong type, and doc strings"""
        A = self.A
        B = self.B
        sigma = math.sqrt(float(inv_lamb)) * G_op([lamb, 0, 0, 1])
        tau = math.sqrt(float(inv_lamb)) * G_op([1, 0, 0, -lamb])
        if k >= 0:
            sigma_k = sigma ** k
            tau_k = tau ** k
        else:
            sigma_inv = math.sqrt(float(inv_lamb)) * G_op([1, 0, 0, lamb])
            tau_inv = math.sqrt(float(inv_lamb)) * G_op([lamb, 0, 0, -1])
            sigma_k = sigma_inv ** -k
            tau_k = tau_inv ** -k
        shift_A = sigma_k * A * sigma_k
        shift_B = tau_k * B * tau_k
        state = matrix_state(shift_A, shift_B)
        return state