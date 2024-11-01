from __future__ import annotations

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
            raise TypeError("Elements in A must be of type float or int.")
        if not (np.issubdtype(B.dtype, np.integer) or np.issubdtype(B.dtype, np.floating)):
            raise TypeError("Elements in B must be of type float or int.")
        
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
        return f"(A={self.A}, B={self.B})"