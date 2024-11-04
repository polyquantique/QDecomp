from __future__ import annotations

from Dsqrt2_2x2matrix import G_op
from Dsqrt2 import D, Dsqrt2, lamb, inv_lamb
import numpy as np

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
        
        # Check that all elements in A and B are int or float
        if not (np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating)):
            raise TypeError("Elements in the matrix must be of type float or int.")
        if not (np.issubdtype(B.dtype, np.integer) or np.issubdtype(B.dtype, np.floating)):
            raise TypeError("Elements in the matrix must be of type float or int.")
        
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
        return A[0, 1]
    
    @property
    def beta(self) -> float:
        B = self.B
        return B[0, 1]
    
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
        G_conj = G_op.conjugate(G)
        new_A = G_op.dag(G) * A * G
        new_B = G_op.dag(G_conj) * B * G_conj
        state = matrix_state(new_A, new_B)
        return state
    
    def shift(self, k: int) -> matrix_state:
        """To do: error message if wrong type, and doc strings"""
        A = self.A
        B = self.B
        sigma_k_mod = G_op([lamb, 0, 0, 1]) ** k
        tau_k_mod = G_op([1, 0, 0, -lamb]) ** k
        inv_lamb_k = inv_lamb ** k
        shift_A = inv_lamb_k * sigma_k_mod * A * sigma_k_mod
        shift_B = inv_lamb_k * tau_k_mod * B * tau_k_mod
        state = matrix_state(shift_A, shift_B)
        return state