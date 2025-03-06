import numpy as np
from typing import Union, Tuple


class QudecompGate:
    def __init__(
        self,
        matrix: np.ndarray,
        qubits: Union[int, Tuple[int, int]],
        name: str,
        epsilon: float,
    ):
        """
        Initialize the QuantumGate with a matrix, qubits, name, and epsilon.

        :param matrix: np.array of size 4x4 or 2x2
        :param qubits: int for single qubit or tuple of size two for two qubits
        :param name: Name of the quantum gate
        :param epsilon: Desired error for the gate
        """
        if matrix.shape not in [(2, 2), (4, 4)]:
            raise ValueError("Matrix must be of size 2x2 or 4x4")

        if matrix.shape == (2, 2) and not isinstance(qubits, int):
            raise ValueError("For 2x2 matrix, qubits must be a single integer")

        if matrix.shape == (4, 4) and not (
            isinstance(qubits, tuple) and len(qubits) == 2
        ):
            raise ValueError("For 4x4 matrix, qubits must be a tuple of size two")

        self.matrix = matrix
        self.qubits = qubits
        self.name = name
        self.epsilon = epsilon
        self._sequence_decomposition = None

    @property
    def sequence_decomposition(self) -> None:
        return self._sequence_decomposition

    def __repr__(self) -> str:
        """
        Return a string representation of the quantum gate.
        """
        return f"QuantumGate(matrix={self.matrix}, name={self.name}, qubits={self.qubits}, epsilon={self.epsilon})"


H_gate = QudecompGate(np.array([[1, 1], [1, -1]]) / np.sqrt(2), 0, "H", 0)
