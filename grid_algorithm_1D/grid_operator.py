from __future__ import annotations 

from Rings import Dsqrt2
import numpy as np

class grid_operator:
    """Class to do symbolic computation with matrices of elements of the ring D[√2].
    
    The matrices are of the form [[a b], [c d]], where a, b, c, d are 

    Attributes:
        
    """
    def __init__(self, grid_operator):
        """
        
        """

        # Convert input to NumPy array for consistency
        grid_operator = np.array(grid_operator)

        # Validate shape
        if grid_operator.shape != (2, 2):
            raise ValueError("The grid operator must be 2x2 in size.")

        # Validate each element's type
        if not all(isinstance(element, Dsqrt2) for element in grid_operator.flatten()):
            raise TypeError(f"All elements in the matrix must be of type {Dsqrt2.__name__}.")

        self.grid_operator = grid_operator

    def __repr__(self) -> str:
        return str(grid_operator)
    
I = grid_operator([[1, 0], [0, 1]])
print(I)