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
This module provides tools for decomposing quantum circuits into the Clifford+T gate set.

The main function in this module, `circuit_decomposition`, takes a quantum circuit represented as a list of gates. Each gate is defined by:
- A matrix representation (either 2x2 for single-qubit gates or 4x4 for two-qubit gates).
- The target qubits on which the gate acts.
- A desired error tolerance (`epsilon`) for approximating the gate.

The function returns a list of `QGate` objects, where each gate is decomposed into the Clifford+T gate set.

The `QGate` class provides additional properties, such as the matrix representation, control and target qubits, and the decomposition error, allowing users to inspect the resulting gates in detail.

### Features:

1. **Error Tolerance**:
   The decomposition process respects the desired error tolerance (`epsilon`) when approximating Rz rotationnal gates.

2. **Integration with `QGate`**:
   The resulting gates are represented as `QGate` objects, which provide a rich interface for inspecting gate properties.

### Example Usage:
```python
import numpy as np
from cliffordplust.decompositions.circuit import circuit_decomposition
from scipy.stats import unitary_group

# Define a quantum circuit
circuit = [
    (np.array([[1, 0], [0, 1j]]), (0,), 0.01),  # Single-qubit S gate
    (np.eye(4)[[0, 1, 3, 2]], (0, 1), 0.01),   # Two-qubit CNOT gate
    (unitary_group.rvs(2), (1,), 0.001),        # Random single-qubit gate
]

# Decompose the circuit
decomposed_circuit = circuit_decomposition(circuit)

# Print the decomposed gates
for gate in decomposed_circuit:
    print(gate)
    # For single-qubit gates:
    # Gate: RandomGate
    # Sequence: T H T T H ...
    # Control: None
    # Target: (1,)
    # Matrix:
    # [[...]]
    # Matrix target: (1,)

    # For two-qubit gates:
    # Gate: CNOT
    # Sequence: CNOT
    # Control: 0
    # Target: (1,)
    # Matrix:
    # [[1. 0. 0. 0.]
    #  [0. 1. 0. 0.]
    #  [0. 0. 0. 1.]
    #  [0. 0. 1. 0.]]
    # Matrix target: (0, 1)
"""
from cliffordplust.decompositions import (
    zyz_decomposition,
    cnot_decomposition,
)
from cliffordplust.grid_problem.rz_approx import z_rotational_approximation
from cliffordplust.exact_synthesis import *
from cliffordplust.circuit import QGate
import numpy as np


def circuit_decomposition(
    circuit: list[tuple[np.ndarray, tuple[int, ...], float] | list[QGate]],
) -> list[QGate]:
    """
    Decompose a quantum circuit into a list of QGate objects using the Clifford+T gate set.

    This function takes a quantum circuit represented as a list of tuples, where each tuple contains:
    - A gate matrix (either 2x2 or 4x4 `np.ndarray`).
    - The target qubits on which the gate acts (a tuple of integers, e.g., `(int,)` for single-qubit gates or `(int, int)` for two-qubit gates).
    - The desired error tolerance (`epsilon`) for approximating the gate.

    The function returns a list of `QGate` objects, where each gate is either a CNOT or a single qubit gate.
    If the gate is a single-qubit gate, its decomposition into Clifford+T gates is computed, and the resulting sequence is stored in the `sequence` property of the `QGate` object.

    The `QGate` class provides additional properties such as the matrix representation, control and target qubits, and the decomposition error.

    Args:
        circuit (list[tuple[np.ndarray, tuple[int, ...], float]]):
            The quantum circuit to decompose. Each tuple in the list represents a gate with:
            - np.ndarray: The matrix representation of the gate (2x2 for single-qubit gates, 4x4 for two-qubit gates).
            - tuple[int, ...]: The target qubits on which the gate acts.
            - float: The dseired error tolerance espilon that will be applied to each approximation of Rz rotationnal gates

    Note: Since in general 3 Rz gates are required in the decomposition of a single qubit gate, the error epsilon may not coincide with the total error between the found sequence decomposition and the initial gate.

    Returns:
        list[QGate]: A list of `QGate` objects representing the decomposed circuit in the Clifford+T gate set.

    Raises:
        TypeError: If the input circuit is not a list of tuples (np.ndarray, tuple, float).
        ValueError: If the input matrix is not 2x2 or 4x4.
        ValueError: If the target qubits do not match the size of the matrix.
    """

    # Test if input circuit is a list
    if type(circuit) is not list:
        raise TypeError(f"Input circuit must be a list, got {type(circuit)}")

    # Test if gates are correct type
    if (
        not isinstance(circuit, list)
        or not all(
            isinstance(gate, tuple)
            and len(gate) == 3
            and isinstance(gate[0], np.ndarray)
            and isinstance(gate[1], tuple)
            and isinstance(gate[2], float)
            for gate in circuit
        )
        and not all(isinstance(gate, QGate) for gate in circuit)
    ):
        raise TypeError(
            f"Input gates' type must be tuples (np.ndarray, tuple, float) or QGates, got first gate : {type(circuit[0])}"
        )

    decomposed_circuit = []
    for gate in circuit:
        if type(gate) is tuple:
            gate = QGate.from_tuple(gate)

        # if sequence is already initialized, skip decomposition
        if gate.sequence is not None or gate.epsilon is None:
            decomposed_circuit.append(gate)
            continue

        # if gate is 2x2, no CNOT decomposition is needed
        if gate.matrix.shape == (2, 2):
            # if target qubits are not a tuple with exactly one element, raise error
            if len(gate.matrix_target) != 1:
                raise ValueError(
                    f"Target qubits must be a tuple with exactly one element for one qubit gate, got {gate.matrix_target}"
                )
            cnot_decomp_lists = [gate]

        # if gate is 4x4, CNOT decomposition is needed
        elif gate.matrix.shape == (4, 4):
            # if target qubits are not a tuple with exactly two elements, raise error
            if len(gate.matrix_target) != 2:
                raise ValueError(
                    f"Target qubits must be a tuple with exactly two elements for 2 qubits gate, got {gate.matrix_target}"
                )
            cnot_decomp_lists = cnot_decomposition(gate.matrix)
        # if gate is not 2x2 or 4x4, raise error
        else:
            raise ValueError(
                f"Input gate must be a 2x2 or 4x4 matrix, got {gate.matrix.shape}"
            )

        for cnot_decomp_qgate in cnot_decomp_lists:

            # if gate sequence is already initialized, skip decomposition
            if cnot_decomp_qgate.sequence is not None:
                decomposed_circuit.append(cnot_decomp_qgate)
                continue

            # Decomposition of all single qubit gates using zyz, rz_approx and exact synthesis
            angles = zyz_decomposition(cnot_decomp_qgate.matrix)[:-1]
            approxed_sequences = []
            for angle in angles:
                # if angle is 0, decomposition is identity ("")
                if np.isclose(angle, 0, atol=gate.epsilon / 100):
                    approxed_sequences.append("")

                # if angle is not 0, follow normal decomposition
                else:
                    rz_approxed_gate = z_rotational_approximation(
                        gate.epsilon, (angle + 4 * np.pi) if angle < 0 else angle
                    )
                    approxed_sequences.append(exact_synthesis_alg(rz_approxed_gate))

            # Construct the gate decomposition sequence
            gate_sequence = (
                approxed_sequences[2]
                + "HSSSH"
                + approxed_sequences[1]
                + "HSH"
                + approxed_sequences[0]
            )
            cnot_decomp_qgate.set_decomposition(gate_sequence, gate.epsilon)
            decomposed_circuit.append(cnot_decomp_qgate)
    return decomposed_circuit
