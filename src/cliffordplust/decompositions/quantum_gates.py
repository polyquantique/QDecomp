import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from typing import Callable, Any

SQRT2 = np.sqrt(2)

# Single qubit gates

X = np.array([[0, 1], [1, 0]]) # Pauli X gate

Y = np.array([[0, -1j], [1j, 0]]) # Pauli Y gate

Z = np.array([[1, 0], [0, -1]]) # Pauli Z gate

H = 1 / SQRT2 * np.array([[1, 1], [1, -1]]) # Hadamard gate

S = np.array([[1, 0], [0, 1.0j]]) # Phase gate

V = 1 / 2 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) # Square root of X gate

T = np.array([[1, 0], [0, np.exp(1.0j * np.pi / 4)]]) # T gate


# Two qubit gates

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) # SWAP gate

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) # CNOT gate

CNOT1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]) # Inverted CNOT gate

DCNOT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]) # CNOT, then inverted CNOT

INV_DCNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]) # Inverted CNOT, then CNOT

CY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]) # Controlled Y gate

CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]) # Controlled Z gate

CH = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 / SQRT2, 1 / SQRT2], [0, 0, 1 / SQRT2, -1 / SQRT2]] # Controlled Hadamard gate
)


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


def power_pauli_z(p: float) -> NDArray[np.floating]:
    """
    Return the Pauli Z power gate.

    Args:
        p (float): Power of the Pauli Z matrix.

    Returns:
        np.ndarray: Pauli Z power gate.
    """
    return np.diag([1, np.exp(1.0j * np.pi * p)])


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


gates: dict[str, NDArray[np.floating]] = {
    "H": H,
    "X": X,
    "Y": Y,
    "Z": Z,
    "V": V,
    "S": S,
    "T": T,
    "SWAP": SWAP,
    "CNOT": CNOT,
    "CNOT1": CNOT1,
    "DCNOT": DCNOT,
    "InvDCNOT": INV_DCNOT,
    "CH": CH,
    "CZ": CZ,
    "CY": CY,
}

parametric_gates: dict[str, Callable[[float], NDArray[np.floating]] | Callable[[float, float, float], NDArray[np.floating]]] = {
    "PY": power_pauli_y,
    "PZ": power_pauli_z,
    "canonical": canonical_gate,
}

