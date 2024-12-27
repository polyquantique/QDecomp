from __future__ import annotations

import numpy as np
import math

class state:
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
