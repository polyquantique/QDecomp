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

from qdecomp.decompositions import sqg_decomp, zyz_decomposition
from qdecomp.utils import QGate

np.random.seed(42)  # For reproducibility


# Rotation and phase matrices
def ry(teta):
    return np.array([[np.cos(teta / 2), -np.sin(teta / 2)], [np.sin(teta / 2), np.cos(teta / 2)]])


def rz(teta):
    return np.array([[np.exp(-1.0j * teta / 2), 0], [0, np.exp(1.0j * teta / 2)]])


def phase(alpha):
    return np.exp(1.0j * alpha)


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
    t0, t1, t2, _ = zyz_decomposition(U)
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
