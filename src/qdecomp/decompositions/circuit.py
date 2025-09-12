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
This module implements a helper function, simplifying the process of decomposing large circuits which contain SQG and TQG.
It uses the :mod:`qdecomp.decompositions.tqg` module to decompose each gate in the circuit.
"""

from typing import Iterable

from qdecomp.decompositions.sqg import sqg_decomp
from qdecomp.decompositions.tqg import tqg_decomp
from qdecomp.utils import QGate


def circuit_decomp(
    circuit: Iterable[QGate],
) -> list[QGate]:
    """
    Decompose a quantum circuit into the Clifford+T gate set using :class:`QGate` objects.

    Args:
        circuit (Iterable[QGate]): An iterable of QGate objects representing the quantum circuit to decompose.

    Returns:
        list[QGate]: A list of QGate objects representing the decomposed gates in the Clifford+T gate set.

    Raises:
        TypeError: If the input circuit is not a list or contains non-QGate objects.
        ValueError: If a gate in the circuit has an unsupported number of qubits (not 1 or 2).
    """

    # Test if input circuit is an iterable
    if not hasattr(circuit, "__iter__"):
        raise TypeError(f"Input circuit must be an iterable, got {type(circuit)}")

    # Initialize the decomposed circuit
    decomposed_circuit = []
    for gate in circuit:
        if not isinstance(gate, QGate):
            raise TypeError(
                f"Input circuit must be a list of QGate objects, got list index {circuit.index(gate)} of type: {type(gate)}"
            )

        if gate.num_qubits == 1:
            sequence = sqg_decomp(gate.init_matrix, epsilon=gate.epsilon)[0]
            decomposed_gate = [
                QGate.from_sequence(
                    sequence=sequence,
                    target=gate.target,
                ),
            ]
        elif gate.num_qubits == 2:
            decomposed_gate = tqg_decomp(gate.init_matrix, epsilon=gate.epsilon)
        else:
            raise ValueError(
                f"Unsupported gate size {gate.num_qubits}. Only single and two-qubit gates are supported."
            )

        decomposed_circuit.extend(decomposed_gate)

    return decomposed_circuit
