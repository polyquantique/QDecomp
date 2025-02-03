import numpy as np
import math

from grid_problem import find_points, find_grid_operator
from grid_operator import Grid_Operator

import sys

sys.path.append("CliffordPlusT")

from Steiner_ellipse.steiner_ellipse import steiner_ellipse_def, ellipse_bbox

def z_rotational_approximation(epsilon: float, theta: float) -> np.ndarray:
    p1, p2, p3 = find_points(epsilon, theta)
    E, _ = steiner_ellipse_def(p1, p2, p3)
    p_p = (1 - epsilon**2 / 2) * np.array([math.cos(theta), -math.sin(theta)])
    I = np.array([[1, 0], [0, 1]], dtype=float)
    inv_G, G = find_grid_operator(E, I)
    G_conj = G.conjugate()
    n = 0
    solution = False
    while solution == False:
        bbox_1 = ellipse_bbox(G.dag().as_float() @ E @ G.as_float(), p_p)
        bbox_2 = ellipse_bbox(G_conj.dag().as_float() @ I @ G_conj.as_float(), np.zeros(2))
        A = math.sqrt(2 ** n) * bbox_1
        B = (-math.sqrt(2)) ** n * bbox_2


z_rotational_approximation(1e-4, math.pi / 6)