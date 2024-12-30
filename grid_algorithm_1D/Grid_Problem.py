import numpy as np
import math
from Rings import Domega, D, Dsqrt2, lamb, inv_lamb
from Grid_Operator import grid_operator, I, R, K, X, Z, A, B
from State import state, special_sigma, inv_special_sigma


def find_grid_operator(A: np.matrix, B: np.matrix) -> grid_operator:
    """To do: docstrings, comments and error messages"""
    initial_state = state(A, B)
    initial_state_bias = initial_state.bias
    inv_grid_op = I
    if abs(initial_state_bias) > 1:
        k_lower = (-1 - initial_state_bias)/2
        k = math.ceil(k_lower)
        new_state = initial_state.shift(k)
    else:
        new_state = initial_state
        k = 0
    while new_state.skew >= 15:
        G_i = find_special_grid_operator(new_state)
        inv_grid_op *= G_i
        new_state = new_state.transform(G_i)
    if k < 0:
        inv_grid_op = (inv_special_sigma ** abs(k)) * inv_grid_op * (special_sigma ** abs(k))
    else:
        inv_grid_op = (special_sigma ** k) * inv_grid_op * (inv_special_sigma ** k)
    grid_op = inv_grid_op.inv()
    return grid_op

def find_special_grid_operator(ellipse_state: state) -> grid_operator:
    """To do: docstrings, comments and error messages"""
    special_grid_operator = I
    z = ellipse_state.z
    gamma = ellipse_state.gamma
    b = ellipse_state.b
    beta = ellipse_state.beta
    if beta <= 0:
        special_grid_operator *= Z
    if abs(z) <= 0.8 and abs(gamma) <= 0.8:
        special_grid_operator *= R
    elif b >= 0:
        if gamma <= -z:
            special_grid_operator *= X
        if z <= 0.3 and gamma >= 0.8:
            special_grid_operator *= K
        elif z >= 0.3 and gamma >= 0.3:
            c = min(z, gamma)
            if math.floor(float(lamb) ** c / 2) < 1:
                n = 1
            else:
                n = math.ceil(float(lamb) ** c / 4)
            special_grid_operator *= A ** n
        elif z >= 0.8 and gamma <= 0.3:
            special_grid_operator *= K.conjugate()
        else:
            ValueError("To do")
    else:
        if gamma <= -z:
            special_grid_operator *= X
        elif z >= -0.2 and gamma >= -0.2:
            c = min(z, gamma)
            if math.floor(float(lamb) ** c / math.sqrt(2)) < 1:
                n = 1
            else:
                n = math.ceil(float(lamb) ** c / math.sqrt(8))
            special_grid_operator *= A ** n
        else:
            ValueError("To do")
    return special_grid_operator

print(find_grid_operator(np.matrix([[1, 0], [0, 1]]), np.matrix([[10000, 1e-20], [1e-20, 10000]])))