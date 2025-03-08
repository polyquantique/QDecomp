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

from cliffordplust.circuit.qgate import QGate


class QCircuit:
    """
    Class to represent a quantum circuit composed of multiple quantum gates.

    The class contains a list of QGate objects that represent the quantum gates in the circuit.

    Attributes:
        gates (list[QGate]): List of quantum gates in the circuit
    """

    def __init__(self, gates: list[QGate]) -> None:
        """
        Initialize the QuantumCircuit with a list of quantum gates.

        :param gates: List of quantum gates
        """
        self.gates: list = gates

    def append(self, circuit: "QCircuit") -> None:
        """
        Add a quantum gate to the circuit.

        :param gate: A quantum gate to be added
        """
        self.gates += circuit

    def __repr__(self) -> str:
        """
        Return a string representation of the quantum circuit.
        """
        return f"{'--'.join(f'{str(gate)}' for gate in self.gates)}"

    def __len__(self) -> int:
        """
        Return the number of gates in the circuit.
        """
        return len(self.gates)

    def __getitem__(self, key: int) -> QGate:
        """
        Return the quantum gate at index key.

        :param key: Index of the quantum gate
        """
        return self.gates[key]

    def circuit_sequence_rep(self) -> str:
        """
        Print the Clifford+T sequence associated with the circuit decomposition.
        """
        return f"{'--'.join(f'{gate.sequence_rep()}' for gate in self.gates)}"
