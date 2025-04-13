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
from cliffordplust.exact_synthesis import *
from cliffordplust.decompositions import zyz_decomposition, rz_decomposition
from cliffordplust.circuit import QGate


from typing import Union
from scipy.stats import unitary_group


def sqg_decomp(sqg: Union[np.array, QGate], epsilon: float = 0.01) -> str:
    """
    Decomposes any single qubit gate (SQG) into its sequence of Clifford+T gates

    Args:
        sqg (Union[np.array, QGate]): The single qubit gate to decompose.
        epsilon (float): The tolerance for the decomposition (default: 0.01)

    Returns:
        str: The sequence of gates that approximates the input SQG.

    """
    # Check if the input is a QGate object, if yes, the matrix to the input
    if isinstance(sqg, QGate):
        epsilon = sqg.epsilon
        sqg = sqg.matrix

    # Check if the input is a 2x2 matrix
    if sqg.shape != (2, 2):
        raise ValueError("The input must be a 2x2 matrix, got shape: " + str(sqg.shape))

    angles = zyz_decomposition(sqg)
    alpha = angles[3]
    angles = angles[:-1]
    sequence = ""
    for angle in angles[::-1]:
        # Adjust angle to be in the range [0, 4*pi]
        if angle < 0:
            angle = angle + 4 * np.pi

        # If angles is 0, sequence is identity and skip decomposition
        if np.allclose(angle, 0):
            continue

        # If it is second angle of angles, consider gate to be Y
        if np.allclose(angle, angles[1] + 4 * np.pi if angles[1] < 0 else angles[1]):
            rz_sequence = rz_decomposition(epsilon=epsilon, angle=angle)
            sequence = sequence + " H S H " + rz_sequence + " H S S S H "

        else:
            rz_sequence = rz_decomposition(epsilon=epsilon, angle=angle)
            sequence = sequence + rz_sequence
    return sequence, alpha


if __name__ == "__main__":

    def ry(teta):
        return np.array(
            [[np.cos(teta / 2), -np.sin(teta / 2)], [np.sin(teta / 2), np.cos(teta / 2)]]
        )

    def rz(teta):
        return np.array([[np.exp(-1.0j * teta / 2), 0], [0, np.exp(1.0j * teta / 2)]])

    def phase(alpha):
        return np.exp(1.0j * alpha)

    # # Create unitary matrix
    np.random.seed(42)  # For reproducibility
    epsilon = 1e-3
    U = unitary_group.rvs(2)
    U = np.array(
        [
            [-0.68321574 + 0.52428132j, -0.49707281 + 0.10613191j],
            [-0.10627305 - 0.49704265j, -0.19929413 + 0.8378165j],
        ]
    )
    sqg = QGate.from_matrix(U, matrix_target=(0,), epsilon=epsilon)

    # # Decomposition of the SQG
    sequence, alpha = sqg_decomp(U, epsilon)
    sqg.set_decomposition(sequence, epsilon=epsilon)
    print(sqg.sequence)
    print(f"Initial matrix : {sqg.approx_matrix} ")
    print(f"Decomposed matrix : {phase(alpha) * sqg.matrix} ")
# #### Test all decomposition parts ####
# # Test the ZYZ decomposition
# t0, t1, t2, alpha_ = zyz_decomposition(U)
# U_calculated = phase(alpha_) * rz(t0) @ ry(t1) @ rz(t2)
# # print("ZYZ : ", U_calculated)
# assert np.allclose(U, U_calculated, atol=epsilon)
# if t0 < 0:
#     t0 = t0 + 4 * np.pi
# if t1 < 0:
#     t1 = t1 + 4 * np.pi
# if t2 < 0:
#     t2 = t2 + 4 * np.pi

# # print(f"t0 = {t0}, t1 = {t1}, t2 = {t2}")
# t0_gate = QGate.from_matrix(rz(t0), matrix_target=(0,), epsilon=epsilon)
# t0_sequence = rz_decomposition(epsilon=epsilon, angle=t0)
# t0_gate.set_decomposition(t0_sequence, epsilon=epsilon)
# assert np.allclose(t0_gate.matrix, t0_gate.approx_matrix, atol=epsilon)

# t1_gate = QGate.from_matrix(ry(t1), matrix_target=(0,), epsilon=epsilon)
# t1_sequence = rz_decomposition(epsilon=epsilon, angle=t1)
# t1_sequence = "H S H " + t1_sequence + " H S S S H"
# t1_gate.set_decomposition(t1_sequence, epsilon=epsilon)
# assert np.allclose(t1_gate.matrix, t1_gate.approx_matrix, atol=epsilon)

# t2_gate = QGate.from_matrix(rz(t2), matrix_target=(0,), epsilon=epsilon)
# t2_sequence = rz_decomposition(epsilon=epsilon, angle=t2)
# t2_gate.set_decomposition(t2_sequence, epsilon=epsilon)
# assert np.allclose(t2_gate.matrix, t2_gate.approx_matrix, atol=epsilon)

# decomposed_U = t0_gate.matrix @ t1_gate.matrix @ t2_gate.matrix
# print("Decomposed U =", decomposed_U)
# # assert np.allclose(U, decomposed_U, atol=epsilon)

# print(
#     f" t0 {t0}: {t0_gate.sequence} \n t1 {t1}: {t1_gate.sequence} \n t2 {t2}: {t2_gate.sequence}\n"
# )
# total_sequence = (
#     t2_gate.sequence + " " + t1_gate.sequence + " " + t0_gate.sequence
# )  # Fix sequence concatenation

# print(total_sequence)
# assert total_sequence == sqg.sequence
# sqg.set_decomposition(total_sequence, epsilon=epsilon)
# print("Final matrix", sqg.matrix)
# assert np.allclose(decomposed_U, sqg.matrix, atol=epsilon)
# assert np.allclose(phase(alpha_) * sqg.matrix, sqg.approx_matrix, atol=epsilon)

# t0 test
# print(t0)
# t0_gate = QGate.from_matrix(rz(t0), matrix_target=(0,), epsilon=epsilon)
# t0_sequence = rz_decomposition(epsilon=epsilon, angle=t0)
# t0_gate.set_decomposition(t0_sequence, epsilon=epsilon)
# print(t0_gate.approx_matrix)
# print(t0_gate.matrix)
# error = max(np.linalg.svd(t0_gate.matrix - t0_gate.approx_matrix, compute_uv=False))
# print(error)
