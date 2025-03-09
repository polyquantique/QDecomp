# Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
import mpmath as mp

import numpy as np

from cliffordplust.grid_problem.grid_operator import A, B, I, K, R, X, Z, Grid_Operator
from cliffordplust.rings.rings import *
from cliffordplust.grid_problem.state import inv_special_sigma, special_sigma, State


def find_points(epsilon: float, theta: float) -> np.ndarray:
    """Find the three points which define the slice \u211B_\u03B5 as shown in Section 7.2 of 
    https://arxiv.org/pdf/1403.2975.

    Args:
        theta (float): The angle \u03B8 (0 \u2264 \u03B8 < 4\u03C0) of the R_z gate.
        epsilon (float): The error of the approximation of R_z.

    Returns:
        tuple: A tuple containing three points (p1, p2, p3), where:
            - p1 (array[float]): The origin point [cos(\u03B8 / 2), sin(\u03B8 / 2)].
            - p2 (array[float]): The first computed point on the slice.
            - p3 (array[float]): The second computed point on the slice.

    Raises:
        ValueError: If theta is not in the range [0, 4\u03C0].
        ValueError: If epsilon is not less than 0.5.
        ValueError: If theta or epsilon cannot be converted to floats.

    """

    # Attempt to convert the input parameters to floats
    try:
        theta = float(theta)
        epsilon = float(epsilon)
    except (ValueError, TypeError):
        raise TypeError("Both theta and epsilon must be convertible to floats.")

    # Verify the value of theta
    if theta > 4 * math.pi or theta < 0:
        raise ValueError("The value of theta must be between 0 and 4\u03C0.")

    # Verify the value of epsilon
    if epsilon >= 0.5:
        raise ValueError("The maximal allowable error is 0.5.")
    
    theta = mp.mpf(theta)
    epsilon = mp.mpf(epsilon)

    # Compute important values only once
    sine = mp.sin(theta / 2)
    cosine = mp.cos(theta / 2)
    p1 = [mp.mpf(0), mp.mpf(0)]

    """
    The calculations needed to find the points are the solving of a trivial quadratic equation which is not 
    solved in detail here. Refer to Section 7.2 of https://arxiv.org/pdf/1403.2975 to understand the problem 
    statement. Note that in order to have sufficient precision on the values of the points, the intersection 
    between the unit disk and the vector z is set to be the origin. 
    """
    # Handle the special case where the sine is 0
    if sine == 0:
        p2 = [-(cosine * epsilon**2) / 2, epsilon * mp.sqrt(1 - epsilon**2 / 4)]
        p3 = [-(cosine * epsilon**2) / 2, -epsilon * mp.sqrt(1 - epsilon**2 / 4)]
    else:
        # Set the proper values for x2 and x3 in order to find the points p2 and p3
        delta = epsilon * mp.sqrt(4 / (sine**2) - epsilon**2 / (sine**2))
        b = (epsilon**2 * cosine) / (sine**2)
        denom = 2 * cosine**2 / sine**2 + 2
        x2 = (-b + delta) / denom
        x3 = (-b - delta) / denom
        p2 = [x2, cosine * x2 / sine + (epsilon**2 / 2) / sine]
        p3 = [x3, cosine * x3 / sine + (epsilon**2 / 2) / sine]

    r = np.array([cosine, -sine])

    return np.array(p1) + r, np.array(p2) + r, np.array(p3) + r


def find_grid_operator(A: np.ndarray, B: np.ndarray) -> Grid_Operator:
    """Find the grid operator which reduces the skew of a state to less than 15.

    Args:
        A (np.ndarray): The 2x2 matrix defining the first ellipse.
        B (np.ndarray): The 2x2 matrix defining the second ellipse.

    Returns:
        grid_op (Grid_Operator): The resulting grid operator that satisfies the initial criteria.

    Raises:
        TypeError: If `A` or `B` are not of type `np.ndarray` or convertable to it.
        ValueError: If `A` or `B` are not 2x2 matrices or are incompatible with the `State` class.
    """

    state = State(A, B)

    # Set the inverse grid operator to the identity
    inv_grid_op = I
    
    while state.skew >= 15:
        # Adjust the bias
        if abs(state.bias) > 1:
            # Find the value of k and apply the shift
            k_upper = (1 - float(state.bias)) / 2
            k = math.floor(k_upper)
            temp_state = state.shift(k)
        else:
            temp_state = state
            k = 0

        # Finds G_i such that the skew is reduced by at least 10%
        G_i = find_special_grid_operator(temp_state)
        if k < 0:
            G_i = (inv_special_sigma**-k) * G_i * (special_sigma**-k)
        else:
            G_i = (special_sigma**k) * G_i * (inv_special_sigma**k)
        inv_grid_op = inv_grid_op * G_i
        state = state.transform(G_i)
        
    grid_op = inv_grid_op.inv()
    return inv_grid_op, grid_op


def find_special_grid_operator(state: State) -> Grid_Operator:
    """Find the special grid operator which reduces the skew of a state by at least 10%.

    Args:
        state (State): The state defined by a pair of ellipses.

    Returns:
        special_grid_operator (Grid_Operator): The resulting special grid operator that satisfies the initial criteria.

    Raises:
        ValueError: If the algorithm encountered unaccounted-for values of z and zeta
    """

    # Initialize the special grid operator as the identity operator
    special_grid_operator = I

    # Flip b and beta when beta is negative
    if state.beta < 0:
        special_grid_operator = special_grid_operator * Z
        state = state.transform(Z)
    
    # Flip the signs of both z and zeta 
    if state.zeta < -state.z:
            special_grid_operator = special_grid_operator * X
            state = state.transform(X)
            
    # Refer to Figure 7 of Appendix A in https://arxiv.org/pdf/1403.2975 for this part (it really helps)
    if abs(state.z) <= 0.8 and abs(state.zeta) <= 0.8:
        special_grid_operator = special_grid_operator * R
    elif state.b >= 0:
        if state.z <= 0.3 and state.zeta >= 0.8:
            special_grid_operator = special_grid_operator * K
        elif state.z >= 0.3 and state.zeta >= 0.3:
            c = min(state.z, state.zeta)
            n = max(1, math.floor(float(LAMBDA) ** c / 2))
            special_grid_operator = special_grid_operator * A**n
        elif state.z >= 0.8 and state.zeta <= 0.3:
            special_grid_operator = special_grid_operator * K.conjugate()
        else:
            # The algorithm should never reach this line
            raise ValueError("Encountered unaccounted-for values for the state parameters in Step-Lemma")
    else:
        if state.z >= -0.2 and state.zeta >= -0.2:
            c = min(state.z, state.zeta)
            n = max(1, math.floor(float(LAMBDA) ** c / math.sqrt(2)))
            special_grid_operator = special_grid_operator * B**n
        else:
            # The algorithm should never reach this line
            raise ValueError("Encountered unaccounted-for values for the state parameters in Step-Lemma")
    return special_grid_operator