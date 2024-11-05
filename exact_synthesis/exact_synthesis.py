import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Domega import Domega, H, T, T_inv


def exact_synthesis(U: np.array) -> str:

    if not np.all([isinstance(element, Domega) for element in U.flatten()]):
        raise TypeError("Matrix elements must be of class D[ω]")
    elif U.shape != (2, 2):
        raise TypeError("Matrix must be of size 2x2")
    elif not is_unitary(U):
        raise ValueError("Matrix must be unitary")

    sequence: str = ""
    norm_z = U[0, 0] * U[0, 0].complex_conjugate()
    s = norm_z.sde()
    print(s)
    while s > 3:
        for k in [0, 1, 2, 3]:
            U_prime = H @ T_inv ** (k) @ U
            norm_z_prime = U_prime[0, 0] * U_prime[0, 0].complex_conjugate()
            sde = norm_z_prime.sde()
            if norm_z_prime.sde() == s - 1:
                sequence += k * "T" + "H"
                s = norm_z_prime.sde()
                U = U_prime
                break
    return sequence, U


def is_unitary(matrix):
    # Calculate the conjugate transpose
    conj_transpose = conj_transpose = np.array(
        [[element.complex_conjugate() for element in row] for row in matrix.T]
    )

    # Calculate the product of the matrix and its conjugate transpose
    product = np.dot(matrix, conj_transpose)
    # Identity matrix of the same size
    D_0 = Domega((0, 0), (0, 0), (0, 0), (0, 0))
    D_1 = Domega((0, 0), (0, 0), (0, 0), (1, 0))
    identity = np.array([[D_1, D_0], [D_0, D_1]], dtype=Domega)
    # Check if the product is close to the identity matrix
    return matrix_equal(product, identity)


def matrix_equal(matrix1, matrix2):
    for row1, row2 in zip(matrix1, matrix2):
        for elem1, elem2 in zip(row1, row2):
            if not elem1 == elem2:
                return False
    return True


def apply_sequence(U, sequence):  # in reverse order :
    for char in sequence[::-1]:
        if char == "H":
            U = H @ U
        elif char == "T":
            U = T @ U
        else:
            raise ValueError("Invalid character in sequence")
    return U


if __name__ == "__main__":
    U = (
        H @ T @ T @ T @ H @ H @ T @ H @ T @ H @ H @ T @ H @ T @ H @ T @ H @ H @ H @ T
    )  # random unitary matrix with s >3

    print(f"Initial gate : {U}")
    sequence, U_f = exact_synthesis(U)
    print(f"Sequence : {sequence}")
    print(f"Matrix with s<3 : \n{U_f}")
    print(f"Final matrix : \n{apply_sequence(U_f, sequence)}")
