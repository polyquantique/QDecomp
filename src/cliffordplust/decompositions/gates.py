import numpy as np
from scipy.linalg import expm

SQRT2 = np.sqrt(2)

H = 1 / SQRT2 * np.array([[1, 1], [1, -1]])

X = np.array([[0, 1], [1, 0]])

Y = np.array([[0, -1j], [1j, 0]])

Z = np.array([[1, 0], [0, -1]])

V = 1 / 2 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])

S = np.array([[1, 0], [0, 1.0j]])

T = np.array([[1, 0], [0, np.exp(1.0j * np.pi / 4)]])

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

CNOT1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

DCNOT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])

INV_DCNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])

CY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])

CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

CH = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 / SQRT2, 1 / SQRT2], [0, 0, 1 / SQRT2, -1 / SQRT2]]
)


def power_pauli_y(p: float) -> np.ndarray:
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


def power_pauli_z(p: float) -> np.ndarray:
    """
    Return the Pauli Z power gate.

    Args:
        p (float): Power of the Pauli Z matrix.

    Returns:
        np.ndarray: Pauli Z power gate.
    """
    return np.diag([1, np.exp(1.0j * np.pi * p)])


def canonical_gate(tx: float, ty: float, tz: float) -> np.ndarray:
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


gates = {
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
    "PY": power_pauli_y,
    "PZ": power_pauli_z,
    "canonical": canonical_gate,
}
