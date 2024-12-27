from __future__ import annotations 

from Rings import Dsqrt2
import numpy as np

class grid_operator:
    """Class to do symbolic computation with matrices of elements of the ring D[√2].
    
    The matrices are of the form [[a b], [c d]], where a, b, c, d are 

    Attributes:
        
    """
    def __init__(self, grid_operator) -> None:
        """
        
        """
        # Convert input to NumPy matrix for consistency
        grid_operator = np.matrix(grid_operator)

        # Validate shape
        if grid_operator.shape != (2, 2):
            raise ValueError("The grid operator must be 2x2 in size.")

        # Validate each element
        for element in grid_operator.flatten().tolist()[0]:
            if not isinstance(element, (int, Dsqrt2)):
                raise TypeError(f"Element {element} must be an int or Dsqrt2.")

        self.grid_operator = grid_operator

    def __repr__(self) -> str:
        grid_operator = self.grid_operator
        return str(grid_operator)
        
I = [[1, 0], [0, 1]]