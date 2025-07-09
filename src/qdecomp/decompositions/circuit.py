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
This module contains a small function to help decompose bigger quantum circuits into the Clifford+T gate set.
It implements the :mod:`qdecomp.decompositions.tqg` module to decompose each gate in the circuit.
"""
from qdecomp.decompositions import tqg_decomp

from qdecomp.utils import QGate


def circuit_decomp(
    circuit: list[QGate],
) -> list[QGate]:
    """
    Decompose a quantum circuit into the Clifford+T gate set using QGate objects.

    Args:
        circuit (list[QGate]): A list of QGate objects representing the quantum circuit to decompose.

    Returns:
        list[QGate]: A list of QGate objects representing the decomposed gates in the Clifford+T gate set.
    """

    # Test if input circuit is a list
    if not isinstance(circuit, list):
        raise TypeError(f"Input circuit must be a list, got {type(circuit)}")

    # Initialize the decomposed circuit
    decomposed_circuit = []
    for gate in circuit:
        if not isinstance(gate, QGate):
            raise TypeError(
                f"Input circuit must be a list of QGate objects, got list index {circuit.index(gate)} of type: {type(gate)}"
            )

        decomposed_gate = tqg_decomp(gate.init_matrix, epsilon=gate.epsilon)
        decomposed_circuit.extend(decomposed_gate)

    return decomposed_circuit
