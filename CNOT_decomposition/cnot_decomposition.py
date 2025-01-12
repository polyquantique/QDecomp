import numpy as np


def nearest_kronecker_product(C):
    C = C.reshape(2, 2, 2, 2)
    C = C.transpose(0, 2, 1, 3)
    C = C.reshape(4, 4)

    u, sv, vh = np.linalg.svd(C)

    A = np.sqrt(sv[0]) * u[:, 0].reshape(2, 2)
    B = np.sqrt(sv[0]) * vh[0, :].reshape(2, 2)

    return A, B


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
    D = np.sqrt(D2)

    Q1 = eigvecs.T
    Q2 = V @ Q1.T @ np.linalg.inv(D)
    
    K12 = M @ Q1 @ M.T.conj()
    K34 = M @ Q2 @ M.T.conj()

    # K12 = M.T.conj() @ Q1 @ M
    # K34 = M.T.conj() @ Q2 @ M

    A1, A2 = nearest_kronecker_product(K12)

    if not np.allclose(Q1.T @ Q1, np.eye(4)):
        print("\nQ1 is not orthogonal!!!!!!!!!!!!!!!!!!")
        print(Q1)
        print()
        print(Q1.T @ Q1)

    if not np.allclose(Q2.T @ Q2, np.eye(4)):
        print("\nQ2 is not orthogonal!!!!!!!!!!!!!!!!!!")
        print(Q2)
        print()
        print(Q2.T @ Q2)

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
    print("D")
    print(D)
    print()

    VV_reconstructed = Q1.T @ D2 @ Q1
    print("Q1")
    print(Q1)
    print()
    print("Q2")
    print(Q2)
    print()
    print("K12")
    print(K12)
    print()
    print("A1")
    print(A1)
    print()
    print("A2")
    print(A2)
    print()
    print("K34")
    print(K34)
    print()
    print("U")
    print(U)
    print()
    print("V")
    print(V)
    print()
    print("VV")
    print(VV)
    print()
    print("VV_reconstructed")
    print(VV_reconstructed)
    print()
    print(f"{np.allclose(VV, VV_reconstructed) = }")
    print("Error: ", np.linalg.norm(VV - VV_reconstructed))
