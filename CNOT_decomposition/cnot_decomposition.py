import numpy as np


def cnot_decomposition(U):
    """
    Decompose a 2-qubit special unitary into CNOT gates and single qubit gates.
    
    :param U: 2-qubit special unitary matrix
    :return: list of CNOT gates and single qubit gates
    """
    # Check if the input matrix is 2x2
    if U.shape != (4, 4):
        raise ValueError(f'Input matrix shape must be 4x4. Got shape {U.shape}.')
    
    # Check if the input matrix is special
    if not np.isclose(np.linalg.det(U), 1):
        raise ValueError(f'The input matrix must be special (det(U) = 1). Got det(U) = {np.linalg.det(U)}.')
    
    # Check if the matrix is unitary
    if not np.allclose(U @ U.T.conj(), np.eye(4)):
        raise ValueError(f'The input matrix must be unitary.')
    
    # Magic gate
    M = np.array([
        [1, 1.j, 0, 0],
        [0, 0, 1.j, 1],
        [0, 0, 1.j, -1],
        [1, -1.j, 0, 0]
    ]) / np.sqrt(2)

    V = M @ U @ M.T.conj()
    VV = V.T @ V

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(VV)
    D2 = np.diag(eigvals)
    Q1 = eigvecs

    print()
    print("eigvals")
    print(eigvals)
    print()
    print("eigvecs")
    print(eigvecs)
    print()
    print("D2")
    print(D2)
    print()

    VV_reconstructed = Q1.T @ D2 @ Q1
    print("VV")
    print(VV)
    print()
    print("VV_reconstructed")
    print(VV_reconstructed)
    print()
    print(f"{np.allclose(VV, VV_reconstructed) = }")
    print("Error: ", np.linalg.norm(VV - VV_reconstructed))
