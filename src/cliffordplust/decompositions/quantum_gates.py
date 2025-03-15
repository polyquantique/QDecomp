import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from typing import Callable, Any
from cliffordplust.circuit import QGate

SQRT2 = np.sqrt(2)


class Gates:
    
    """Single qubit gates."""

    X = np.array([[0, 1], [1, 0]]) # Pauli X gate

    Y = np.array([[0, -1j], [1j, 0]]) # Pauli Y gate

    Z = np.array([[1, 0], [0, -1]]) # Pauli Z gate

    H = 1 / SQRT2 * np.array([[1, 1], [1, -1]]) # Hadamard gate

    S = np.array([[1, 0], [0, 1.0j]]) # Phase gate

    V = 1 / 2 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) # Square root of X gate

    T = np.array([[1, 0], [0, np.exp(1.0j * np.pi / 4)]]) # T gate


    """Two qubit gates."""

    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) # SWAP gate

    ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]) # iSWAP gate

    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) # CNOT gate

    CNOT1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]) # Inverted CNOT gate

    DCNOT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]) # CNOT, then inverted CNOT

    INV_DCNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]) # Inverted CNOT, then CNOT

    CY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]) # Controlled Y gate

    CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]) # Controlled Z gate

    CH = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 / SQRT2, 1 / SQRT2], [0, 0, 1 / SQRT2, -1 / SQRT2]] # Controlled Hadamard gate
    )

    MAGIC = (
        1 / SQRT2 * np.array([[1, 1.0j, 0, 0], [0, 0, 1.0j, 1], [0, 0, 1.0j, -1], [1, -1.0j, 0, 0]]) # Magic gate
    )

    
    """Parametric gates."""
    @staticmethod
    def power_pauli_y(p: float) -> NDArray[np.floating]:
        """
        Return the Pauli Y power gate.

        Args:
            p (float): Power of the Pauli Y gate.

        Returns:
            np.ndarray: Pauli Y power gate.
        """
        angle = np.pi / 2 * p
        phase = np.exp(1.0j * angle)

        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        return phase * matrix


    @staticmethod
    def power_pauli_z(p: float) -> NDArray[np.floating]:
        """
        Return the Pauli Z power gate.

        Args:
            p (float): Power of the Pauli Z matrix.

        Returns:
            np.ndarray: Pauli Z power gate.
        """
        return np.diag([1, np.exp(1.0j * np.pi * p)])
    

    @staticmethod
    def canonical_gate(tx: float, ty: float, tz: float) -> NDArray[np.floating]:
        """
        Return the matrix form of the canonical gate for the given parameters.

        Args:
            tx, ty, tz (floats): Parameters of the canonical gates

        Returns:
            np.ndarray: Matrix form of the canonical gate.
        """
        XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        YY = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
        ZZ = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        exponent = -1.0j * np.pi / 2 * (tx * XX + ty * YY + tz * ZZ)
        return expm(exponent)



class Circuit:

    @staticmethod
    def magic_decomp(q0: int, q1: int) -> list[QGate]:
        """
        Circuit implementation of the magic gate.

        Decompose the magic gate into a circuit of S, H, and CNOT gates.

        Args:
            q0 (int): First target qubit of the gate.
            q1 (int): Second target qubit of the gate.

        Returns:
            list[QGate]: List of QGate objects representing the decomposition of the magic gate.
        """
        magic_circuit = [
            QGate.from_tuple(('S', (q0,), 0)),
            QGate.from_tuple(('S', (q1,), 0)),
            QGate.from_tuple(('H', (q1,), 0)),
            QGate.from_tuple(('CNOT', (q1, q0), 0)),
        ]
        return magic_circuit
    
    @staticmethod
    def magic_dag_decomp(q0: int, q1: int) -> list[QGate]:
        """
        Circuit implementation of the hermitian conjugate of the magic gate.

        Decompose the hermitian conjugate of the magic gate into a circuit of SDAG, H, and CNOT gates.

        Args:
            q0 (int): First target qubit of the gate.
            q1 (int): Second target qubit of the gate.

        Returns:
            list[QGate]: List of QGate objects representing the decomposition of the gate.
        """
        magic_dag_circuit = [
            QGate.from_tuple(('CNOT', (q1, q0), 0)),
            QGate.from_tuple(('H', (q1,), 0)),
            QGate.from_tuple(('SDAG', (q1,), 0)),
            QGate.from_tuple(('SDAG', (q0,), 0)),
        ]
        return magic_dag_circuit
    
    @staticmethod
    def swap_decomp(q0: int, q1: int) -> list[QGate]:
        """
        Circuit implementation of the SWAP gate.

        Decompose the SWAP gate into a circuit of CNOT gates.

        Args:
            q0 (int): First target qubit of the gate.
            q1 (int): Second target qubit of the gate.

        Returns:
            list[QGate]: List of QGate objects representing the decomposition of the SWAP gate.
        """
        swap_circuit = [QGate.from_tuple(('CNOT', (q0, q1), 0)), 
                        QGate.from_tuple(('CNOT', (q1, q0), 0)), 
                        QGate.from_tuple(('CNOT', (q0, q1), 0))]
        return swap_circuit
    
    @staticmethod
    def cy_decomp(q0: int, q1: int) -> list[QGate]:
        """
        Circuit implementation of the controlled Y gate.

        Decompose the controlled Y gate into a circuit of SDAG, CNOT and S gates.

        Args:
            q0 (int): First target qubit of the gate.
            q1 (int): Second target qubit of the gate.

        Returns:
            list[QGate]: List of QGate objects representing the decomposition of the controlled Y gate.
        """
        cy_circuit = [QGate.from_tuple(('SDAG', (q1,), 0)),
                        QGate.from_tuple(('CNOT', (q0, q1), 0)),
                        QGate.from_tuple(('S', (q1,), 0))]
        return cy_circuit
    
    @staticmethod
    def cz_decomp(q0: int, q1: int) -> list[QGate]:
        """
        Circuit implementation of the controlled Z gate.

        Decompose the controlled Z gate into a circuit of H, CNOT, and H gates.

        Args:
            q0 (int): First target qubit of the gate.
            q1 (int): Second target qubit of the gate.

        Returns:
            list[QGate]: List of QGate objects representing the decomposition of the controlled Z gate.
        """
        cz_circuit = [QGate.from_tuple(('H', (q1,), 0)),
                        QGate.from_tuple(('CNOT', (q0, q1), 0)),
                        QGate.from_tuple(('H', (q1,), 0))]
        return cz_circuit
    
    @staticmethod
    def ch_decomp(q0: int, q1: int):
        """
        Circuit implementation of the controlled Hadamard gate.

        Decompose the controlled Hadamard gate into a circuit of Clifford+T gates.

        Args:
            q0 (int): First target qubit of the gate.
            q1 (int): Second target qubit of the gate.

        Returns:
            list[QGate]: List of QGate objects representing the decomposition of the controlled Hadamard gate.
        """
        ch_circuit = [QGate.from_tuple(('S', (q1,), 0)),
                QGate.from_tuple(('H', (q1,), 0)),
                QGate.from_tuple(('T', (q1,), 0)),
                QGate.from_tuple(('CNOT', (q0, q1), 0)),
                QGate.from_tuple(('TDAG', (q1,), 0)),
                QGate.from_tuple(('H', (q1,), 0)),
                QGate.from_tuple(('SDAG', (q1,), 0))]
        return ch_circuit
    
    @staticmethod
    def iswap_decomp(q0: int, q1: int) -> list[QGate]:
        """
        Circuit implementation of the iSWAP gate.

        Decompose the iSWAP gate into a circuit of Clifford+T gates.

        Args:
            q0 (int): First target qubit of the gate.
            q1 (int): Second target qubit of the gate.

        Returns:
            list[QGate]: List of QGate objects representing the decomposition of the iSWAP gate.
        """
        iswap_circuit = Circuit.swap_decomp(q0, q1) + Circuit.cz_decomp(q0, q1) + [
            QGate.from_tuple(('S', (q0,), 0)),
            QGate.from_tuple(('S', (q1,), 0)),
        ]
        return iswap_circuit

    
    @staticmethod
    def decomposition(name: str, q0, q1) -> list[QGate]:
        match name:
            case 'MAGIC':
                return Circuit.magic_decomp(q0, q1)
            case 'MAGIC_DAG':
                return Circuit.magic_dag_decomp(q0, q1)
            case 'SWAP':
                return Circuit.swap_decomp(q0, q1)
            case 'CY':
                return Circuit.cy_decomp(q0, q1)
            case 'CZ':
                return Circuit.cz_decomp(q0, q1)
            case 'CH':
                return Circuit.ch_decomp(q0, q1)
            case 'ISWAP':
                return Circuit.iswap_decomp(q0, q1)
            case _:
                raise ValueError(f"Decomposition of gate {name} not implemented.")

