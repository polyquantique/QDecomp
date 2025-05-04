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

import numpy as np

from qdecomp.decompositions import rz_decomp, zyz_decomposition
from qdecomp.utils import QGate


def sqg_decomp(
    sqg: np.ndarray | QGate, epsilon: float, add_gloabal_phase: bool = False
) -> tuple[str, float]:
    """
    Decomposes any single qubit gate (SQG) into its sequence of Clifford+T gates

    Args:
        sqg (np.array || QGate]): The single qubit gate to decompose.
        epsilon (float): The tolerance for the decomposition
        add_global_phase (bool): If `True`, adds the global phase to the sequence and return statements (default: `False`).

    Returns:
        tuple(str, float): Tuples containing the sequence of gates that approximates the input SQG and the global phase associated with the zyz decomposition of the gate

    Raises:
        ValueError: If the input is not a 2x2 matrix
        ValueError: If the input is a QGate object with no epsilon value set
        ValueError: If the input is not a QGate object or a 2x2 matrix
    """
    # Check if the input is a QGate object, if yes, the matrix to the input
    if isinstance(sqg, QGate):
        if sqg.epsilon is None:
            raise ValueError("The QGate object has no epsilon value set.")

        else:
            epsilon = sqg.epsilon
            sqg = sqg.matrix

    # Check if the input is a 2x2 matrix
    if sqg.shape != (2, 2):
        raise ValueError("The input must be a 2x2 matrix, got shape: " + str(sqg.shape))

    angles = zyz_decomposition(sqg)
    alpha = angles[3]
    angles = angles[:-1]
    sequence = ""
    for angle in angles:
        # Adjust angle to be in the range [0, 4*pi]
        if angle < 0:
            angle = angle + 4 * np.pi

        # If angles is 0, sequence is identity and skip decomposition
        if np.allclose(angle, 0):
            continue

        # If it is second angle of angles, consider gate to be Y
        if np.allclose(angle, angles[1] + 4 * np.pi if angles[1] < 0 else angles[1]):
            rz_sequence = rz_decomp(
                epsilon=epsilon, angle=angle, add_global_phase=add_gloabal_phase
            )
            sequence = sequence + " H S H " + rz_sequence + " H S S S H "

        # Else, consider gate to be Z
        else:
            rz_sequence = rz_decomp(
                epsilon=epsilon, angle=angle, add_global_phase=add_gloabal_phase
            )
            sequence = sequence + rz_sequence

    return sequence, alpha
