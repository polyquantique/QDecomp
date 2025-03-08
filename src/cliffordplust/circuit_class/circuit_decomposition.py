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

from cliffordplust.circuit_class import QudecompCircuit
from cliffordplust.decomposition import (
    zyz_decomposition,
    cnot_decomposition,
)  # change name when available
from cliffordplust.grid_problem.rz_approx import z_rotational_approximation
from cliffordplust.exact_synthesis import exact_synthesis_alg


# /!\ Still need to test since the CNOT decomposition is not implemented /!\


def circuit_decompostion(circuit):
    """
    Decompose a circuit into a sequence of Clifford+ gates, will add more complete description
    """
    decomposed_circuit = QudecompCircuit()
    for gate in circuit.gates:
        cnot_decomp_circuit = gate.cnot_decomposition(
            gate
        )  # Nedds to be changed once the CNOT decomposition is implemented
        for cnot_decomp_gate in cnot_decomp_circuit.gates:
            # if CNOT, no decomposition is needed
            if cnot_decomp_gate.name == "CNOT":
                cnot_decomp_gate.sequence_decomposition("CNOT")
                decomposed_circuit.append(cnot_decomp_gate)
            # if SQG, apply decomposition
            elif cnot_decomp_gate.matrix.shape == (2, 2):
                # Apply ZYZ decomposition
                angles = zyz_decomposition(cnot_decomp_gate.matrix)
                rz_approxed_gates = [
                    z_rotational_approximation(cnot_decomp_gate.epsilon, angle)
                    for angle in angles
                ]
                approxed_sequences = [
                    exact_synthesis_alg(gate) for gate in rz_approxed_gates
                ]
                gate_sequence = (
                    approxed_sequences(2)
                    + "HSSSH"
                    + approxed_sequences(1)
                    + "HSH"
                    + approxed_sequences(0)
                )
                cnot_decomp_gate.sequence_decomposition(gate_sequence)
                decomposed_circuit.append(cnot_decomp_gate)
    return decomposed_circuit
