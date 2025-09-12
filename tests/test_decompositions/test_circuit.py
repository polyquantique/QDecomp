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

"""Test the `circuit_decomp` function."""

import numpy as np
import pytest
from scipy.stats import unitary_group

from qdecomp.decompositions import circuit_decomp
from qdecomp.utils import QGate

np.random.seed(42)  # For reproducibility


@pytest.mark.parametrize("trial", range(3))
def test_circuit_decomp_output_type(trial):
    """Test that circuit_decomp returns a list of QGate objects."""

    # Create a circuit with multiple random single-qubit gates
    sqg_circuit = [
        QGate.from_matrix(
            matrix=unitary_group.rvs(2, random_state=trial), target=(i,), epsilon=0.01
        )
        for i in range(3)
    ]

    # Create a circuit with multiple random two-qubit gates
    tqg_circuit = [
        QGate.from_matrix(
            matrix=unitary_group.rvs(4, random_state=trial), target=(0, i + 1), epsilon=0.01
        )
        for i in range(3)
    ]

    # Combine the circuits to form a circuit with both single and two-qubit gates
    mixed_circuit = sqg_circuit + tqg_circuit

    # Decompose the circuits
    decomposed_sqg_circuit = circuit_decomp(sqg_circuit)
    decomposed_tqg_circuit = circuit_decomp(tqg_circuit)
    decomposed_mixed_circuit = circuit_decomp(mixed_circuit)

    # Verify that the result is a list of QGate objects
    # SQG only
    assert isinstance(decomposed_sqg_circuit, list)
    assert len(decomposed_sqg_circuit) > 0

    # TQG only
    assert isinstance(decomposed_tqg_circuit, list)
    assert len(decomposed_tqg_circuit) > 0

    # Mixed circuit
    assert isinstance(decomposed_mixed_circuit, list)
    assert len(decomposed_mixed_circuit) > 0

    for gate in decomposed_sqg_circuit:
        assert isinstance(gate, QGate), f"Expected QGate object, got {type(gate)}"
        assert gate.sequence_matrix.shape == (2, 2)
    for gate in decomposed_tqg_circuit:
        assert isinstance(gate, QGate), f"Expected QGate object, got {type(gate)}"
        assert gate.sequence_matrix.shape in [(2, 2), (4, 4)]

    for gate in decomposed_mixed_circuit:
        assert isinstance(gate, QGate), f"Expected QGate object, got {type(gate)}"
        assert gate.sequence_matrix.shape in [(2, 2), (4, 4)]


def test_circuit_decomp_empty_circuit():
    """Test circuit_decomp with an empty circuit."""
    circuit = []

    # Decompose the circuit
    decomposed_circuit = circuit_decomp(circuit)

    # Verify that the result is an empty list
    assert isinstance(decomposed_circuit, list)
    assert len(decomposed_circuit) == 0


def test_circuit_decomp_invalid_input():
    """Test that circuit_decomp raises appropriate errors for invalid inputs."""
    # Test with non-list input
    with pytest.raises(TypeError, match="Input circuit must be a list"):
        circuit_decomp("not a list")

    # Test with list containing non-QGate objects
    with pytest.raises(TypeError, match="Input circuit must be a list of QGate objects"):
        circuit_decomp([np.array([[1, 0], [0, 1]])])
