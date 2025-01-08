import numpy as np

# Rotation and phase matrices
Rx = lambda teta: np.array(
    [[np.cos(teta / 2), -1.0j * np.sin(teta / 2)], [-1.0j * np.sin(teta / 2), np.cos(teta / 2)]]
)
Ry = lambda teta: np.array(
    [[np.cos(teta / 2), -np.sin(teta / 2)], [np.sin(teta / 2), np.cos(teta / 2)]]
)
Rz = lambda teta: np.array([[np.exp(-1.0j * teta / 2), 0], [0, np.exp(1.0j * teta / 2)]])
phase = lambda alpha: np.exp(1.0j * alpha)


def zyz_decomposition(U):
    """
    Compute the ZYZ decomposition of a 2x2 unitary matrix U.
    U = e**(i alpha) * Rz(t2) * Ry(t1) * Rz(t0)

    :param U: A 2x2 unitary matrix.
    :return: A tuple (t0, t1, t2, alpha) containing ZYZ rotation angles (rad) and the global phase
    (rad).
    """
    det = np.linalg.det(U)

    if not np.isclose(np.abs(det), 1):
        raise ValueError(f"The input matrix must be unitary. Got a matrix with determinant {det}.")

    if not U.shape == (2, 2):
        raise ValueError(f"The input matrix must be 2x2. Got a matrix with shape {U.shape}.")

    alpha = np.arctan2(det.imag, det.real) / 2  # det = exp(2 i alpha)
    V = np.exp(-1.0j * alpha) * U  # V = exp(-i alpha)*U is a special unitary matrix

    # Avoid divisions by zero if U is diagonal
    if np.isclose(np.abs(V[0, 0]), 1, rtol=1e-16, atol=1e-16):
        t0 = 0
        t1 = 0
        t2 = -2 * np.angle(V[0, 0])
        return t0, t1, t2, alpha

    # Compute the first rotation angle
    if np.abs(V[0, 0]) >= np.abs(V[0, 1]):
        t1 = 2 * np.arccos(np.abs(V[0, 0]))
    else:
        t1 = 2 * np.arcsin(np.abs(V[0, 1]))

    # Useful variables for the next steps
    V11_ = V[1, 1] / np.cos(t1 / 2)
    V10_ = V[1, 0] / np.sin(t1 / 2)

    a = 2 * np.arctan2(V11_.imag, V11_.real)
    b = 2 * np.arctan2(V10_.imag, V10_.real)

    # The following system of equations is solved to find t0 and t2
    # t0 + t2 = a
    # t0 - t2 = b
    t0 = (a + b) / 2
    t2 = (a - b) / 2

    return t0, t1, t2, alpha
