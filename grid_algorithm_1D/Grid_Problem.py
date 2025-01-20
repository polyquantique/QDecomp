import math

import numpy as np
from grid_operator import A, B, I, K, R, X, Z, Grid_Operator
from Rings import D, Domega, Dsqrt2, inv_lamb, lamb
from state import inv_special_sigma, special_sigma, State


def find_grid_operator(A: np.matrix, B: np.matrix) -> Grid_Operator:
    """Find the grid operator which reduces the skew of a state to less than 15.

    Args:
        A (np.matrix): The 2x2 matrix defining the first ellipse.
        B (np.matrix): The 2x2 matrix defining the second ellipse.

    Returns:
        grid_op (Grid_Operator): The resulting grid operator that satisfies the initial criteria.

    Raises:
        TypeError: If `A` or `B` are not of type `np.matrix`.
        ValueError: If `A` or `B` are not 2x2 matrices or are incompatible with the `State` class.
    """

    initial_state = State(A, B)
    initial_state_bias = initial_state.bias
    # Set the inverse grid operator to the identity
    inv_grid_op = I
    if abs(initial_state_bias) > 1:
        # Find the value of k and apply the shift
        k_upper = (1 - initial_state_bias) / 2
        k = math.floor(k_upper)
        new_state = initial_state.shift(k)
    else:
        new_state = initial_state
        k = 0
    while new_state.skew >= 15:
        # Finds G_i such that the skew is reduced by at least 10%
        G_i = find_special_grid_operator(new_state)
        inv_grid_op = G_i * inv_grid_op
        new_state = new_state.transform(G_i)
    if k < 0:
        inv_grid_op = (inv_special_sigma**-k) * inv_grid_op * (special_sigma**-k)
    else:
        inv_grid_op = (special_sigma**k) * inv_grid_op * (inv_special_sigma**k)
    grid_op = inv_grid_op.inv()
    return grid_op


def find_special_grid_operator(ellipse_state: State) -> Grid_Operator:
    """Find the special grid operator which reduces the skew of a state by at least 10%.

    Args:
        ellipse_state (State): The state defined by a pair of ellipses.

    Returns:
        special_grid_operator (Grid_Operator): The resulting grid operator that satisfies the initial criteria.

    Raises:
        ValueError: If the algorithm encountered unaccounted-for values of z and zeta
    """

    # Initialize the special grid operator as the identity operator
    special_grid_operator = I

    # Extract parameters from the state
    z = ellipse_state.z
    zeta = ellipse_state.zeta
    b = ellipse_state.b
    beta = ellipse_state.beta

    # Flip b and beta when beta is negative
    if beta <= 0:
        special_grid_operator = Z * special_grid_operator

    # Refer to Figure 7 of Appendix A in https://arxiv.org/pdf/1403.2975 for this part (it really helps)
    if abs(z) <= 0.8 and abs(zeta) <= 0.8:
        special_grid_operator = R * special_grid_operator
    elif b >= 0:
        if zeta <= -z:
            special_grid_operator = X * special_grid_operator
        if z <= 0.3 and zeta >= 0.8:
            special_grid_operator = K * special_grid_operator
        elif z >= 0.3 and zeta >= 0.3:
            c = min(z, zeta)
            if math.floor(float(lamb) ** c / 2) < 1:
                n = 1
            else:
                n = math.ceil(float(lamb) ** c / 4)
            special_grid_operator = A**n * special_grid_operator
        elif z >= 0.8 and zeta <= 0.3:
            special_grid_operator = K.conjugate() * special_grid_operator
        else:
            # The algorithm should never reach this line
            ValueError("Encountered unaccounted-for values for the state parameters in Step-Lemma")
    else:
        if zeta <= -z:
            special_grid_operator = X * special_grid_operator
        elif z >= -0.2 and zeta >= -0.2:
            c = min(z, zeta)
            if math.floor(float(lamb) ** c / math.sqrt(2)) < 1:
                n = 1
            else:
                n = math.ceil(float(lamb) ** c / math.sqrt(8))
            special_grid_operator = B**n * special_grid_operator
        else:
            # The algorithm should never reach this line
            ValueError("Encountered unaccounted-for values for the state parameters in Step-Lemma")
    return special_grid_operator
