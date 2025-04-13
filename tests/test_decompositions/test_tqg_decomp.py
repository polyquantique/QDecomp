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

"""Test the tqg decomposition function."""

import numpy as np
from scipy.stats import unitary_group
import pytest

from qdecomp.decompositions import tqg_decomp
from qdecomp.utils import QGate


np.random.seed(42)  # For reproducibility


def multiply_circuit(circuit: list[QGate]) -> np.ndarray:
    """
    Multiply a list of QGates objects to get the matrix representation of the circuit.

    Args:
        circuit (list[QGate]): The list of gates in the circuit.

    Returns:
        np.ndarray: The matrix representation of the circuit.
    """
    M = np.eye(4)
    for gate in circuit:
        if gate.matrix.shape == (2, 2):
            if gate.matrix_target == (0,):
                M = np.kron(gate.matrix, np.eye(2)) @ M
            else:
                M = np.kron(np.eye(2), gate.matrix) @ M
        else:
            M = gate.matrix @ M
    return M


@pytest.mark.parametrize("trial", range(10))
@pytest.mark.parametrize("epsilon", [0.01, 0.001, 0.0001])
def test_tqg_decomposition_random_unitary(trial, epsilon):
    """Test the tqg_decomposition function with a random unitary matrix."""
    # Test the decomposition
    U = unitary_group.rvs(4)
    decomposition = tqg_decomp(U, epsilon=epsilon)
    reconstructed = multiply_circuit(decomposition)

    # Assert the reconstructed matrix is equal to the original matrix
    if isinstance(U, QGate):
        U = U.matrix
    phase = reconstructed[0, 0] / U[0, 0]

    # account for error propagation in the decomposition (10*epsilon)
    assert np.allclose(reconstructed / phase, U, atol=10 * epsilon)
