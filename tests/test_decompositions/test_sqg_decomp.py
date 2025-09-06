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
#    limitations under the License.import pytest

"""Test the sqg decomposition function."""

import numpy as np
import pytest
from scipy.stats import unitary_group

from qdecomp.decompositions import rz_decomp, sqg_decomp, zyz_decomp
from qdecomp.utils import QGate

np.random.seed(42)  # For reproducibility


# Rotation and phase matrices
def ry(teta):
    return np.array([[np.cos(teta / 2), -np.sin(teta / 2)], [np.sin(teta / 2), np.cos(teta / 2)]])


def rz(teta):
    return np.array([[np.exp(-1.0j * teta / 2), 0], [0, np.exp(1.0j * teta / 2)]])


def phase(alpha):
    return np.exp(1.0j * alpha)


@pytest.mark.parametrize(
    "a, b",
    [
        (0, 1),
        (0.5, np.sqrt(3) / 2),
        (1, 0),
        (-1, 0),
        (-0.7, np.sqrt(51) / 10),
        (complex(1, 1) / np.sqrt(2), 0),
        (complex(2, -3) / 4, np.sqrt(3) / 4),
        (1e-10, 1),
        (1 - 1e-10, np.sqrt(2e-10)),
        (1 - 1e-16, np.sqrt(2e-16)),
    ],
)
@pytest.mark.parametrize("alpha", [0, 1, np.pi, np.pi / 2, 2 * np.pi])
def test_zyz_decomp(a, b, alpha):
    """
    Test the ZYZ decomposition of a 2x2 unitary matrix.
    """
    U = np.exp(1.0j * alpha) * np.array([[a, -b.conjugate()], [b, a.conjugate()]])  # Unitary matrix

    t0, t1, t2, alpha_ = zyz_decomp(U)

    # Check that the decomposition is correct
    U_calculated = phase(alpha_) * rz(t2) @ ry(t1) @ rz(t0)

    assert np.allclose(U, U_calculated, atol=1e-7, rtol=1e-7)


def test_zyz_decomp_arbitrary_unitary():
    """
    Test the ZYZ decomposition of arbitrary unitary matrices.
    """
    unitary_generator = unitary_group(dim=2, seed=42)
    for _ in range(15):
        U = unitary_generator.rvs()  # Generate a random unitary matrix
        t0, t1, t2, alpha_ = zyz_decomp(U)

        # Check that the decomposition is correct
        U_calculated = phase(alpha_) * rz(t2) @ ry(t1) @ rz(t0)

        assert np.allclose(U, U_calculated, atol=1e-7, rtol=1e-7)


def test_zyz_decomp_unitary_error():
    """
    Test the errors raised by the zyz_decomp() function when the matrix is not unitary.
    """
    U = np.array([[1, 0], [0, 2]])
    with pytest.raises(
        ValueError, match="The input matrix must be unitary. Got a matrix with determinant"
    ):
        zyz_decomp(U)


def test_zyz_decomp_shape_error():
    """
    Test the errors raised by the zyz_decomp() function when the matrix is not 2x2.
    """
    U = np.eye(3)
    with pytest.raises(
        ValueError, match=r"The input matrix must be 2x2. Got a matrix with shape \(3, 3\)."
    ):
        zyz_decomp(U)


@pytest.mark.parametrize("epsilon", [1e-2, 1e-3, 1e-4])
@pytest.mark.parametrize("angle", [np.pi / 2, np.pi / 4, np.pi / 6, np.pi, np.pi / 8, np.pi / 12])
def test_rz_decomp_precision(angle, epsilon):
    """Test if rz_decomp returns a sequence within the desired error."""
    sequence = rz_decomp(angle=angle, epsilon=epsilon, add_global_phase=True)
    gate = QGate.from_matrix(
        np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]]),
        target=(0,),
        epsilon=epsilon,
    )
    gate.set_decomposition(sequence, epsilon)
    error = max(np.linalg.svd(gate.sequence_matrix - gate.init_matrix, compute_uv=False))
    assert error < epsilon


@pytest.mark.parametrize("angle", np.random.uniform(0, 2 * np.pi, 10))
@pytest.mark.parametrize("epsilon", [1e-2, 1e-3, 1e-4])
def test_rz_decomp_random_angle(angle, epsilon):
    """Test if the decomposition of a random angle is correct."""
    sequence = rz_decomp(angle=angle, epsilon=epsilon, add_global_phase=True)
    gate = QGate.from_matrix(
        np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]]),
        target=(0,),
        epsilon=epsilon,
    )
    gate.set_decomposition(sequence, epsilon)
    sequence_matrix = gate.sequence_matrix
    error = max(np.linalg.svd(sequence_matrix - gate.init_matrix, compute_uv=False))
    assert error < epsilon


def test_rz_decomp_identity():
    """Test decomposition of an RZ gate with a zero angle."""
    epsilon = 1e-4
    angle = 0.0
    sequence = rz_decomp(angle=angle, epsilon=epsilon)
    assert sequence == ""


def test_rz_decomp_invalid_angle():
    """Test that a non-numeric angle raises a TypeError."""
    with pytest.raises(TypeError):
        rz_decomp(1e-5, "invalid_angle")


@pytest.mark.parametrize("trial", range(10))
@pytest.mark.parametrize("epsilon", [0.01, 0.001, 0.0001])
def test_sqg_decomp_random_unitary(trial, epsilon):
    """Test the validity of the output of the sqg_decomp function or an arbitrary gate"""
    # Generate a random 2x2 unitary matrix (single qubit gate)
    U = unitary_group.rvs(2, random_state=trial)
    sqg = QGate.from_matrix(matrix=U, target=(0,), epsilon=epsilon)
    # Decompose the single qubit gate
    sequence, alpha = sqg_decomp(sqg, epsilon, add_global_phase=True)
    sqg.set_decomposition(sequence, epsilon=epsilon)

    # Account for error propagation in the decomposition (10*epsilon)
    error = max(
        np.linalg.svd(phase(alpha) * sqg.sequence_matrix - sqg.init_matrix, compute_uv=False)
    )
    assert error < 10 * epsilon


@pytest.mark.parametrize("trial", range(10))
@pytest.mark.parametrize("epsilon", [0.01, 0.001, 0.0001])
def test_sqg_decomp_zyz_random(trial, epsilon):
    """Test if the sqg_decomp returns the correct matrix associated with the zyz decomposition."""
    # Generate a random 2x2 unitary matrix (single qubit gate)
    U = unitary_group.rvs(2, random_state=trial)
    sqg = QGate.from_matrix(U, (0,), epsilon=epsilon)
    sequence, _ = sqg_decomp(sqg, epsilon, add_global_phase=True)
    sqg.set_decomposition(sequence, epsilon=epsilon)
    # Evaluate de zyz decomposition matrix
    t0, t1, t2, _ = zyz_decomp(U)
    zyz_matrix = rz(t2) @ ry(t1) @ rz(t0)

    # Account for error propagation in the decomposition (10*epsilon)
    error = max(np.linalg.svd(sqg.sequence_matrix - zyz_matrix, compute_uv=False))
    assert error < 10 * epsilon


def test_sqg_decomp_identity():
    """Test if sqg_decomp correctly handles the identity matrix."""
    identity = np.eye(2)
    sequence, _ = sqg_decomp(identity, epsilon=0.01)
    assert sequence == ""


def test_sqg_decomp_invalid_input_shape():
    """Test if sqg_decomp raises an error for non-2x2 matrices."""
    invalid_matrix = np.eye(3)  # 3x3 matrix
    with pytest.raises(ValueError, match="The input must be a 2x2 matrix"):
        sqg_decomp(invalid_matrix, epsilon=0.01)


def test_sqg_decomp_invalid_epsilon():
    """Test if sqg_decomp raises an error if epsilon is not defined in QGate object."""
    U = unitary_group.rvs(2)
    sqg = QGate.from_matrix(U, (0,))
    with pytest.raises(ValueError, match="The QGate object has no epsilon value set."):
        sqg_decomp(sqg, epsilon=0.01)
