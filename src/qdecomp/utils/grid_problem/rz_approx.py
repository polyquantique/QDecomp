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

"""
This file runs the entire z-rotational approximation algorithm to find the associated sequence.

It contains the :func:`initialization` function, which is only to visually lighten the code and the
:func:`z_rotational_approximation` function which is the main function of this file. Given an angle and
an error, it finds an approximation of the associated z-rotation by solving for potential values
of u, and then checking if there exists a valid associated value for t using the ``Diophantine equation`` module. When u and t are found,
it returns the Clifford+T approximation of the z-rotation.
"""

import math

import mpmath as mp
import numpy as np

from qdecomp.rings.rings import *
from qdecomp.utils.diophantine import solve_xi_eq_ttdag_in_d
from qdecomp.utils.grid_problem.grid_algorithms import solve_grid_problem_2d
from qdecomp.utils.grid_problem.grid_problem import find_grid_operator, find_points
from qdecomp.utils.steiner_ellipse import ellipse_bbox, is_inside_ellipse, steiner_ellipse_def

__all__ = ["initialization", "z_rotational_approximation"]

# Define Identity
I = np.array([[mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(1)]], dtype=object)


def initialization(theta: float, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Initializes important parameters necessary to find the appropriate Rz approximation.

    This function calculates points based on the angle and the error provided, finds the
    necessary ellipse, and determines grid operators and bounding boxes for further
    computations.

    Args:
        theta (float): Angle of the z-rotational gate.
        epsilon (float): Maximum allowable error.

    Returns:
        tuple: A tuple containing:
            - bbox_1 (tuple): Bounding box coordinates for the transformed ellipse.
            - bbox_2 (tuple): Bounding box coordinates for the transformed unit disk.
    """
    # Initializes the ellipses using the points
    p1, p2, p3 = find_points(theta, epsilon)
    E, p_p = steiner_ellipse_def(p1, p2, p3)

    # Find the grid operator using the ellipses
    inv_gop, gop = find_grid_operator(E, I)
    inv_gop_conj = inv_gop.conjugate()

    # Transform the ellipse using the grid operator
    mod_E = (inv_gop.dag()).as_mpfloat() @ E @ inv_gop.as_mpfloat()
    mod_D = (inv_gop_conj.dag()).as_mpfloat() @ I @ inv_gop_conj.as_mpfloat()

    # Finds the bounding boxes
    bbox_1 = ellipse_bbox(mod_E, p_p)
    bbox_2 = ellipse_bbox(mod_D, np.array([mp.mpf(0), mp.mpf(0)]))

    return bbox_1, bbox_2


def z_rotational_approximation(theta: float, epsilon: float) -> np.ndarray:
    """
    Finds the z-rotational approximation up to an error :math:`\\varepsilon`.

    This function finds an approximation of a z-rotational inside the Clifford+T group.

    Args:
        theta (float): Angle :math:`\\theta` of the z-rotational gate.
        epsilon (float): Maximum allowable error :math:`\\varepsilon`.

    Returns:
        np.ndarray: Approximation :math:`M` of a z-rotational inside the Clifford+T subset.

    Raises:
        ValueError: If :math:`\\theta` is not in the range :math:`[0, 4\\pi]`.
        ValueError: If :math:`\\varepsilon \\geq 0.5`.
        ValueError: If :math:`\\theta` or :math:`\\varepsilon` cannot be converted to floats.
    """

    # Attempt to convert the input parameters to floats
    try:
        theta = float(theta)
        epsilon = float(epsilon)
    except (ValueError, TypeError):
        raise TypeError("Both theta and epsilon must be convertible to floats.")

    # Normalize the value of theta
    theta = theta % (4 * math.pi)

    # Verify the value of epsilon
    if epsilon >= 0.5:
        raise ValueError(f"The maximal allowable error is 0.5. Got {epsilon}.")

    # Checks if the angle is trivial
    exponent = round(2 * theta / math.pi)
    if np.isclose(0, theta):
        return np.array(
            [
                [Domega.from_ring(1), Domega.from_ring(0)],
                [Domega.from_ring(0), Domega.from_ring(1)],
            ],
            dtype=object,
        )
    elif np.isclose(2 * theta / math.pi, exponent):
        T = np.array(
            [
                [Domega(-D(1, 0), D(0, 0), D(0, 0), D(0, 0)), Domega.from_ring(0)],
                [Domega.from_ring(0), Domega(D(0, 0), D(0, 0), D(1, 0), D(0, 0))],
            ],
            dtype=object,
        )
        M = T**exponent
        return M

    # Run the initialization function
    bbox_1, bbox_2 = initialization(theta, epsilon)

    # Initialize the exact solution vector in order to evaluate the error later
    z = np.array([mp.cos(theta / 2), -mp.sin(theta / 2)])

    # Define this important value
    delta = mp.mpf(1) - mp.mpf(0.5 * epsilon**2)

    n = 0
    while True:
        # Varies if odd or even
        odd = n % 2
        if odd:
            const = Dsqrt2(D(0, 0), D(1, int((n + 1) / 2)))
        else:
            const = D(1, int(n / 2))

        # Initialize the bounding boxes using n
        A = mp.sqrt(2**n) * bbox_1
        if odd:
            bbox_2_flip = np.flip(bbox_2, axis=1)
            B = -mp.sqrt(2**n) * bbox_2_flip
        else:
            B = mp.sqrt(2**n) * bbox_2

        # For every solution found
        for cand in solve_grid_problem_2d(A, B):
            # Ensure the solution was not already found previously
            is_double = abs(cand.a - cand.c) % 2 == 1 or abs(cand.b - cand.d) % 2 == 1
            if n == 0 or is_double:
                # Find u as Domega and as mpfloat
                u = Domega.from_ring(cand) * Domega.from_ring(const)
                u_float = np.array([u.mp_real(), u.mp_imag()])
                # Find the conjugate of u
                u_conj = u.sqrt2_conjugate()
                u_conj_float = np.array([u_conj.mp_real(), u_conj.mp_imag()])
                # Compute the dot product and the lower bound
                dot = np.dot(u_float, z)
                # If the solution u is valid
                if (
                    dot < 1
                    and dot > delta
                    and u_float[0] ** 2 + u_float[1] ** 2 < 1
                    and is_inside_ellipse(u_conj_float, I, np.zeros(2))
                ):
                    # Run the diophantine module
                    xi = 1 - u.complex_conjugate() * u
                    t = solve_xi_eq_ttdag_in_d(Dsqrt2.from_ring(xi))
                    if t is None:
                        # No associated t values exists
                        pass
                    else:
                        # The solution is found and returned!
                        M = np.array([[u, -t.complex_conjugate()], [t, u.complex_conjugate()]])
                        return M

        n += 1
        if n > 1000000:
            raise ValueError(  # pragma: no cover
                "The algorithm did not find a solution after 1 million iterations. "
                "Try increasing the error for the calculations."
            )
