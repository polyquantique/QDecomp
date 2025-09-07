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
This module contains the main function to decompose single qubit gates (SQG) into a sequence of Clifford+T gates up to a given tolerance :math:`\\varepsilon`.
This module combines the functions from the :mod:`qdecomp.decompositions.rz` and :mod:`qdecomp.decompositions.zyz` modules to achieve this goal.

**Example**

    .. code-block:: python

        >>> from scipy.stats import unitary_group
        >>> from qdecomp.decompositions import sqg_decomp

        # Decompose a random single qubit gate with tolerance 0.001
        >>> sqg = unitary_group.rvs(2, random_state=42)
        >>> sequence, alpha = sqg_decomp(sqg, epsilon=0.001, add_global_phase=True)
        >>> print(sequence, alpha)
        sequence : T H S T H S T [...] S H S W W W W
        alpha : 0.27

        # Decompose a random single qubit gate with tolerance 0.001 up to a global phase
        >>> sqg = unitary_group.rvs(2, random_state=42)
        >>> sequence, _ = sqg_decomp(sqg, epsilon=0.001, add_global_phase=False)
        >>> print(sequence)
        T H S T H S T [...] Z T H Z S H S
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from qdecomp.utils import QGate
from qdecomp.utils.exact_synthesis import exact_synthesis_alg, optimize_sequence
from qdecomp.utils.grid_problem import z_rotational_approximation

__all__ = [
    "zyz_decomp",
    "rz_decomp",
    "sqg_decomp",
]


def zyz_decomp(U: NDArray) -> tuple[float, ...]:
    """
    Any single qubit gate can be decomposed into a series of three rotations around the Z, Y, and Z axis
    and a global phase factor:

    .. math:: U = e^{i \\alpha} R_z(\\theta_2) R_y(\\theta_1) R_z(\\theta_0),

    where :math:`R_z` and :math:`R_y` are the rotation gates around the Z and Y axes, respectively. This is known as the **ZYZ decomposition**.

    This function performs this decomposition on a given unitary 2 x 2 matrix.
    It returns the three rotation angles :math:`\\theta_0,  \\theta_1, \\theta_2` and the phase :math:`\\alpha`.
    For more details, see :cite:`decomp_crooks`.

    Args:
        U (NDArray): A 2 x 2 unitary matrix.

    Returns:
        tuple[float, ...]: (t0, t1, t2, alpha), the three rotation angles (rad) and the global phase (rad)

    Raises:
        ValueError: If the input matrix is not 2 x 2.
        ValueError: If the input matrix is not unitary.

    Examples:

    .. code-block:: python

        >>> import numpy as np
        >>> from qdecomp.decompositions import zyz_decomp

        # Define rotation and phase matrices
        >>> ry = lambda teta: np.array([[np.cos(teta / 2), -np.sin(teta / 2)],
                                        [np.sin(teta / 2), np.cos(teta / 2)]])
        >>> rz = lambda teta: np.array([[np.exp(-1.0j * teta / 2), 0],
                                        [0, np.exp(1.0j * teta / 2)]])
        >>> phase = lambda alpha: np.exp(1.0j * alpha)

        # Create a unitary matrix U
        >>> a = complex(1, 1) / np.sqrt(3)
        >>> b = np.sqrt(complex(1, 0) - np.abs(a) ** 2)  # Ensure that U is unitary
        >>> alpha = np.pi/3
        >>> U = np.exp(1.0j * alpha) * np.array([[a, -b.conjugate()], [b, a.conjugate()]])

        # Compute the decomposition of U
        >>> t0, t1, t2, alpha_ = zyz_decomp(U)

        # Recreate U from the decomposition
        >>> U_calculated = phase(alpha_) * rz(t2) @ ry(t1) @ rz(t0)

        # Print the results
        >>> print("U =\\n", U)
        U =
         [[-0.21132487+0.78867513j -0.28867513-0.5j       ]
         [ 0.28867513+0.5j         0.78867513+0.21132487j]]
        >>> print("U_calculated =\\n", U_calculated)
        U_calculated =
         [[-0.21132487+0.78867513j -0.28867513-0.5j       ]
         [ 0.28867513+0.5j         0.78867513+0.21132487j]]
        >>> print(f"Error = {np.linalg.norm(U - U_calculated)}")
        Error = 1.0007415106216802e-16
    """
    # Convert U to a np.ndarray if it is not already
    U = np.asarray(U)

    # Check the input matrix
    if not U.shape == (2, 2):
        raise ValueError(f"The input matrix must be 2x2. Got a matrix with shape {U.shape}.")

    det = np.linalg.det(U)
    if not np.isclose(abs(det), 1):
        raise ValueError(f"The input matrix must be unitary. Got a matrix with determinant {det}.")

    # Compute the global phase and the special unitary matrix V
    alpha = np.atan2(det.imag, det.real) / 2  # det = exp(2 i alpha)
    V = np.exp(-1.0j * alpha) * U  # V = exp(-i alpha)*U is a special unitary matrix

    # Avoid divisions by zero if U is diagonal
    if np.isclose(abs(V[0, 0]), 1, rtol=1e-14, atol=1e-14):
        t2 = -2 * np.angle(V[0, 0])
        return 0, 0, t2, alpha

    # Compute the first rotation angle
    if abs(V[0, 0]) >= abs(V[0, 1]):
        t1 = 2 * np.acos(abs(V[0, 0]))
    else:
        t1 = 2 * np.asin(abs(V[0, 1]))

    # Useful variables for the next steps
    V11_ = V[1, 1] / np.cos(t1 / 2)
    V10_ = V[1, 0] / np.sin(t1 / 2)

    a = 2 * np.atan2(V11_.imag, V11_.real)
    b = 2 * np.atan2(V10_.imag, V10_.real)

    # The following system of equations is solved to find t0 and t2
    # t0 - t2 = a
    # t0 + t2 = b
    t0 = (a - b) / 2
    t2 = (a + b) / 2

    return t0, t1, t2, alpha


def rz_decomp(angle: float, epsilon: float, add_global_phase=False) -> str:
    """
    This function decomposes a :math:`R_z` gate of the form

    .. math:: 

        R_z = \\begin{pmatrix}
        e^{-i\\theta / 2} & 0  \\\\
        0 & e^{i\\theta / 2}
        \\end{pmatrix},

    into a sequence of Clifford+T gates where :math:`\\theta` is the rotation angle around the Z axis.
    The decomposition is up to a given tolerance :math:`\\varepsilon`.
    The algorithm implemented in this function is based on the one presented by Ross and Selinger in :cite:`decomp_ross`.
    Note: when the `add_global_phase` argument is set to `True`, the sequence includes global phase gates :math:`W = e^{i\\pi/4}`.

    This function uses the :mod:`qdecomp.utils.exact_synthesis`, :mod:`qdecomp.utils.grid_problem` and :mod:`qdecomp.utils.diophantine` modules to achieve this goal.

    Args:
        angle (float): The angle of the RZ gate in radians.
        epsilon (float): The tolerance for the approximation based on the operator norm.
        add_global_phase (bool): If `True`, adds global phase gates W to the sequence (default: `False`).

    Returns:
        sequence (str): The sequence of Clifford+T gates that approximates the RZ gate.

    **Example**
    
        .. code-block:: python
    
            >>> from qdecomp.decompositions import rz_decomp
            >>> from math import pi
            
            # Decompose a RZ gate with angle pi/128 and tolerance 0.001 exactly
            >>> sequence = rz_decomp(epsilon=0.001, angle=pi/128, add_global_phase=True)
            >>> print(sequence)
            H S T H S T H [...] Z S W W W W W W
    
            # Decompose a RZ gate with angle pi/128 and tolerance 0.001 up to a global phase
            >>> sequence = rz_decomp(epsilon=0.001, angle=pi/128, add_global_phase=False)
            >>> print(sequence)
            H S T H S T H [...] Z S H S T H Z S
    """
    # Find the approximation of Rz gates in terms of Domega elements
    domega_matrix = z_rotational_approximation(epsilon=epsilon, theta=angle)

    # Convert the Domega matrix to a string representation
    sequence = exact_synthesis_alg(domega_matrix, insert_global_phase=add_global_phase)
    optimized_sequence = optimize_sequence(sequence)

    # Test if TUTdag has less T than U (Optimization of T-count)
    tut_sequence = "T" + sequence + "TTTTTTT"
    tut_optimized_sequence = optimize_sequence(tut_sequence)

    # Compare the number of T gates in the two sequences
    if tut_optimized_sequence.count("T") < optimized_sequence.count("T"):
        optimized_sequence = tut_optimized_sequence

    sequence = " ".join(optimized_sequence)
    return sequence


def sqg_decomp(
    sqg: Union[np.ndarray, QGate], epsilon: float, add_global_phase: bool = False
) -> tuple[str, float]:
    """
    Decomposes any single qubit gate (SQG) into its optimal sequence of Clifford+T gates up to a given error.

    Args:
        sqg (Union[np.ndarray, QGate]): The matrix representation of the single qubit gate to decompose.
        epsilon (float): The error tolerance for the decomposition.
        add_global_phase (bool): If `True`, adds the global phase to the sequence and return statements (default: `False`).

    Returns:
        tuple(str, float): A tuple containing the sequence of gates that approximates the input SQG and the global phase **alpha** associated with the zyz decomposition of the gate.
        (if add_global_phase is False, **alpha** is set to 0 and the sequence doesn't contain any global phase gate W).

    Raises:
        ValueError: If the input is a QGate object with no epsilon value set.
        ValueError: If the input is not a QGate object or a 2x2 matrix.
    """
    # Check if the input is a QGate object, if yes, the matrix to the input
    if isinstance(sqg, QGate):
        if sqg.epsilon is None:
            raise ValueError("The QGate object has no epsilon value set.")

        epsilon = sqg.epsilon
        sqg = sqg.matrix

    # Check if the input is a 2x2 matrix
    if sqg.shape != (2, 2):
        raise ValueError("The input must be a 2x2 matrix, got shape: " + str(sqg.shape))

    zyz_result = zyz_decomp(sqg)
    alpha = zyz_result[3]
    angles = zyz_result[:-1]
    sequence = ""
    for i, angle in enumerate(angles):
        # Adjust angle to be in the range [0, 4*pi]
        angle = angle % (4 * np.pi)

        # If angle is 0, sequence is identity and skip decomposition
        if angle == 0:
            continue

        # If it is second angle of angles, consider gate to be Y
        if i == 1:
            rz_sequence = rz_decomp(epsilon=epsilon, angle=angle, add_global_phase=add_global_phase)
            sequence += " H S H " + rz_sequence + " H S S S H "

        # Else, consider gate to be Z
        else:
            rz_sequence = rz_decomp(epsilon=epsilon, angle=angle, add_global_phase=add_global_phase)
            sequence += rz_sequence

    return sequence, alpha
