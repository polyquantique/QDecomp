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

from typing import List, Union, Tuple

from cliffordplust.exact_synthesis import *


class QudecompGate:
    """
    Class to represent a quantum gate in a quantum circuit meant to be decomposed.
    The gate must be initialized with a name, np.array, target qubits, and error epsilon.
    Additionnaly, the gate can provide a sequence decomposition once the full decomposition is done

    Attributes:
        matrix (np.ndarray): Matrix representation of the quantum gate
        qubits (int | Tuple[int, int]): Target qubits for the quantum gate
        name (str): Name of the quantum gate
        epsilon (float): Desired error for the gate
        sequence_decomposition (str): Clifford+T sequence associated with the gate decomposition (default is None)
    """

    def __init__(
        self,
        matrix: np.ndarray,
        qubits: int | Tuple[int, int],
        name: str,
        epsilon: float,
    ):
        """
        Initialize the QuantumGate with a matrix, qubits, name, and epsilon.

        Args:
        :param matrix: np.array of size 4x4 or 2x2
        :param qubits: int for single qubit or tuple of size two for two qubits
        :param name: Name of the quantum gate
        :param epsilon: Desired error for the gate

        Raises:
        ValueError: If the matrix is not of size 2x2 or 4x4
        ValueError: If the matrix is of size 2x2 and qubits is not an integer
        ValueError: If the matrix is of size 4x4 and qubits is not a tuple of size two
        """
        # Test if matrix is
        if matrix.shape not in [(2, 2), (4, 4)]:
            raise ValueError("Matrix must be of size 2x2 or 4x4")

        if matrix.shape == (2, 2) and not isinstance(qubits, int):
            raise ValueError("For 2x2 matrix, qubits must be a single integer")

        if matrix.shape == (4, 4) and not (
            isinstance(qubits, tuple) and len(qubits) == 2
        ):
            raise ValueError("For 4x4 matrix, qubits must be a tuple of size two")

        self.matrix: np.ndarray = matrix
        self.qubits: int | tuple[int] = qubits
        self.name: str = name
        self.epsilon: float = epsilon
        self._sequence_decomposition: str = None

    @property
    def sequence_decomposition(self):
        return self._sequence_decomposition

    @sequence_decomposition.setter
    def sequence_decomposition(self, sequence: str) -> None:
        """
        Set the Clifford+T sequence associated with the gate decomposition.

        :param sequence: List of Clifford+T gates
        """
        self._sequence_decomposition = sequence

    def __str__(self) -> str:
        """
        Return a string representation of the quantum gate.
        """
        return f"[{self.name}, {self.qubits}, {self.epsilon}]"

    def sequence_rep(self) -> str:
        """
        Print the Clifford+T sequence associated with the gate decomposition.
        """
        if self.sequence_decomposition is None:
            raise ValueError(
                f"There is no sequence decomposition associated with QudecompGate : {str(self)}"
            )
        else:
            return f"[{self.sequence_decomposition}, {self.qubits}, {self.epsilon}]"


class QudecompCircuit:
    """
    Class to represent a quantum circuit composed of multiple quantum gates.

    The class contains a list of QuedecompGate objects that represent the quantum gates in the circuit.

    Attributes:
        gates (List[QuedecompGate]): List of quantum gates in the circuit
    """

    def __init__(self, gates: Tuple[QudecompGate]):
        """
        Initialize the QuantumCircuit with a list of quantum gates.

        :param gates: List of quantum gates
        """
        self.gates: tuple = gates

    def append(self, gate: QudecompGate) -> None:
        """
        Add a quantum gate to the circuit.

        :param gate: A quantum gate to be added
        """
        self.gates += (gate,)

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

    def __getitem__(self, key: int) -> QudecompGate:
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
