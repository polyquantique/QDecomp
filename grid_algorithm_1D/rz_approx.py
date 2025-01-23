import numpy as np

from grid_problem import find_points, find_grid_operator

import sys

sys.path.append("../CliffordPlusT")

from Steiner_ellipse.steiner_ellipse import steiner_ellipse_def

def z_rotational_approximation(epsilon: float, theta: float) -> np.ndarray:
    points = find_points(epsilon, theta)
    E, p_p = steiner_ellipse_def(points)
    print(E)
    # I = np.array([[1, 0], [0, 1]], dtype=float)
    # inv_G, G = find_grid_operator(E, I)

z_rotational_approximation()