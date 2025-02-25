import math

import numpy as np
import pytest
from scipy.stats import special_ortho_group, ortho_group, unitary_group

from cliffordplust.decompositions import *


def test_power_pauli_y():
    """Test the power of Pauli-Y matrix."""
    # Trivial cases
    assert np.allclose(power_pauli_y(0), np.eye(2))
    assert np.allclose(power_pauli_y(0.5),
                    np.array([[complex(1, 1), complex(-1, -1)], [complex(1, 1), complex(1, 1)]])/2)
    assert np.allclose(power_pauli_y(1), np.array([[0, -1.j], [1.j, 0]]))
    assert np.allclose(power_pauli_y(2), np.eye(2))

    for t in [0, np.pi/2, -2, 10]:
        assert np.allclose(power_pauli_y(t) @ power_pauli_y(-t), np.eye(2))  # Check inverse
        assert np.allclose(power_pauli_y(t), power_pauli_y(t % 2))  # Check periodicity


def test_power_pauli_z():
    """Test the power of Pauli-Z matrix."""
    # Trivial cases
    assert np.allclose(power_pauli_z(0), np.eye(2))
    assert np.allclose(power_pauli_z(0.5), np.diag([1, 1.j]))
    assert np.allclose(power_pauli_z(1), np.array([[1, 0], [0, -1]]))
    assert np.allclose(power_pauli_z(2), np.eye(2))

    for t in [0, np.pi/2, -2, 10]:
        assert np.allclose(power_pauli_z(t) @ power_pauli_z(-t), np.eye(2))  # Check inverse
        assert np.allclose(power_pauli_z(t), power_pauli_z(t % 2))  # Check periodicity


@pytest.mark.parametrize("matrix, result", [
    (np.eye(2), True),
    (-np.eye(2), True),
    (-np.eye(3), False),
    (np.zeros((2, 2)), False),
    (np.arange(1, 5).reshape(2, 2), False),
    (np.ones((2, 2)), False),
    (np.array([[1, -2], [-3, 1]]), False),
    (np.array([[1.2, 3.4], [5, -1]]), False),
    (np.array([[0, 1], [1, 0]]), True),
    (np.array([[0, 1], [-1, 0]]), False),
])
def test_is_special(matrix, result):
    """Test the is_special function."""
    assert is_special(matrix) == result


@pytest.mark.parametrize("matrix, result", [
    (np.eye(2), True),
    (-np.eye(2), True),
    (-np.eye(3), True),
    (np.zeros((2, 2)), False),
    (np.arange(1, 5).reshape(2, 2), False),
    (np.ones((2, 2)), False),
    (np.array([[1, -2], [-3, 1]]), False),
    (np.array([[1.2, 3.4], [5, -1]]), False),
    (np.array([[0, 1], [1, 0]]), True),
    (np.array([[0, 1], [-1, 0]]), True),
])
def test_is_orthogonal(matrix, result):
    """Test the is_orthogonal function."""
    assert is_orthogonal(matrix) == result


@pytest.mark.parametrize("matrix, result", [
    (np.eye(2), True),
    (-np.eye(2), True),
    (-np.eye(3), True),
    (np.zeros((2, 2)), False),
    (np.arange(1, 5).reshape(2, 2), False),
    (np.ones((2, 2)), False),
    (np.array([[1, -2], [-3, 1]]), False),
    (np.array([[1.2, 3.4], [5, -1]]), False),
    (np.array([[0, 1], [1, 0]]), True),
    (np.array([[0, 1], [-1, 0]]), True),
])
def test_is_unitary(matrix, result):
    """Test the is_unitary function."""
    assert is_unitary(matrix) == result


@pytest.mark.parametrize(
    "A",
    [
        np.zeros((2, 2)),
        np.identity(2),
        np.arange(1, 5).reshape(2, 2),
        np.ones((2, 2)),
        np.array([[1, -2], [-3, 1]]),
        np.array([[1.2, 3.4], [5, -1]]),
        np.array([[np.pi, np.e], [np.log(2), np.sqrt(3)]]),
        np.array([[1, 1], [1, -1]]) / math.sqrt(2),
        np.eye(2) / math.sqrt(2),
    ],
)
@pytest.mark.parametrize(
    "B",
    [
        np.zeros((2, 2)),
        np.identity(2),
        np.arange(1, 5).reshape(2, 2),
        np.ones((2, 2)),
        np.array([[1, -2], [-3, 1]]),
        np.array([[1.2, 3.4], [5, -1]]),
        np.array([[np.pi, np.e], [np.log(2), np.sqrt(3)]]),
        np.array([[1, 1], [1, -1]]) / math.sqrt(2),
        np.eye(2) / math.sqrt(2),
    ],
)
def test_kronecker_decomposition(A, B):
    """Test the kronecker decomposition of 4x4 matrix."""
    M = np.kron(A, B)
    a, b = kronecker_decomposition(M)
    assert np.allclose(M, np.kron(a, b))


@pytest.mark.parametrize(
    "U",
    [unitary_group(dim=4, seed=42).rvs() for _ in range(40)]
    + [ortho_group(dim=4, seed=42).rvs() for _ in range(10)]
    + [
        np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),  # SWAP gate
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),  # CNOT gate
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),  # CZ gate
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1.0j], [0, 0, 1.0j, 0]]),  # CY gate
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
                [0, 0, 1 / math.sqrt(2), -1 / math.sqrt(2)],
            ]
        ),  # CH gate
        np.array([[1, 0, 0, 0], [0, 0, 1.0j, 0], [0, 1.0j, 0, 0], [0, 0, 0, 1]]),  # iSWAP gate
        np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]),  # DCNOT gate
        np.identity(4),  # Identity gate
        np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0]]),
        np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, -1, 0]]),
    ],
)
def test_canonical_decomposition(U):
    """Test the canonical decomposition of 4x4 unitary matrix."""
    A, B, t, alpha = canonical_decomposition(U)
    assert np.allclose(
        U, B @ canonical_gate(t[0], t[1], t[2]) @ A * np.exp(1.0j * alpha), rtol=1e-8
    )
    a, b = kronecker_decomposition(A)
    alpha, beta = kronecker_decomposition(B)
    assert np.allclose(A, np.kron(a, b)) and np.allclose(B, np.kron(alpha, beta))


def test_kronecker_decomposition_errors():
    """Test the raise of errors when calling kronecker decomposition with wrong arguments."""
    for M in [((1, 2), (3, 4)), [[1, 2], [3, 4]], 1, "1", 1.0]:
        with pytest.raises(TypeError, match="The input matrix must be a numpy object"):
            kronecker_decomposition(M)
    for M in [
        np.arange(1, 10).reshape(3, 3),
        np.arange(1, 13).reshape(3, 4),
        np.arange(1, 13).reshape(4, 3),
        np.arange(1, 26).reshape(5, 5),
    ]:
        with pytest.raises(ValueError, match="The input matrix must be 4x4"):
            kronecker_decomposition(M)


def test_canonical_decomposition_errors():
    """Test the raise of errors when calling canonical decomposition with wrong arguments."""
    for U in [((1, 2), (3, 4)), [[1, 2], [3, 4]], 1, "1", 1.0]:
        with pytest.raises(TypeError, match="Matrix U must be a numpy object"):
            canonical_decomposition(U)
    for U in [
        np.arange(1, 10).reshape(3, 3),
        np.arange(1, 13).reshape(3, 4),
        np.arange(1, 13).reshape(4, 3),
        np.arange(1, 26).reshape(5, 5),
    ]:
        with pytest.raises(ValueError, match="U must be 4x4"):
            canonical_decomposition(U)
    for U in [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]),  # Not unitary
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) * 1.1,  # Not unitary
    ]:
        with pytest.raises(ValueError, match="U must be unitary"):
            canonical_decomposition(U)


def test_so4_decomposition():
    """Test the SO(4) decomposition."""
    for i in range(10):
        # Use a predefined or randomly generated 4x4 matrix
        match i:
            case 0:
                U = np.eye(4)
            case 1:
                U = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            case 2:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
            case 3:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0]])
            case _:
                U = special_ortho_group(dim=4, seed=42).rvs()
        
        # Test the decomposition
        decomp = so4_decomposition(U)
        reconstructed = np.eye(4)

        splitted_decomp = list()
        for gate, target in decomp:
            # Split the gate into elementary gates
            if type(gate) is not str:
                splitted_decomp.append((gate, target))
            elif " " not in gate:
                splitted_decomp.append((gate, target))
            else:
                for g in gate.split(" "):
                    splitted_decomp.append((g, target))

        # Reconstruct the matrix
        for gate, target in splitted_decomp:
            # Transform a string gate into a np.array gate
            match gate:
                case np.ndarray():
                    pass
                
                case "CNOT":
                    if target == (0, 1):
                        gate = np.array(
                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
                    elif target == (1, 0):
                        gate = np.array(
                            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

                case "H":
                    gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                
                case "T":
                    gate = np.array([[1, 0], [0, np.exp(1.j * np.pi / 4)]])

                case "TDAG":
                    gate = np.array([[1, 0], [0, np.exp(-1.j * np.pi / 4)]])
            
                case "S":
                    gate = np.array([[1, 0], [0, 1.j]])
                
                case "SDAG":
                    gate = np.array([[1, 0], [0, -1.j]])
                
                case _:
                    assert False, f"Unknown gate {gate}"

            # Transform any 2x2 matrix into a 4x4 matrix
            if gate.shape == (2, 2):
                if target == (0, ):
                    transformed_gate = np.kron(gate, np.eye(2))
                elif target == (1, ):
                    transformed_gate = np.kron(np.eye(2), gate)
                else:
                    assert False, f"Unknown target {target}"
            
            else:
                transformed_gate = gate

            reconstructed = transformed_gate @ reconstructed

        # Assert the reconstructed matrix is equal to the original matrix
        assert np.allclose(reconstructed, U, rtol=1e-8)


def test_so4_decomposition_errors():
    """Test the raise of errors when calling SO(4) decomposition with wrong arguments."""
    # Shape error
    U = np.eye(3)
    with pytest.raises(ValueError, match="The input matrix must be 4x4. Got "):
        so4_decomposition(U)
    
    # Orthogonal error
    U = np.eye(4) * 1.1
    with pytest.raises(ValueError, match="The input matrix must be orthogonal."):
        so4_decomposition(U)
    
    # Special error
    U = np.diag([1, 1, 1, -1])
    with pytest.raises(ValueError, match="The input matrix must be special. Got det = "):
        so4_decomposition(U)

def test_o4_det_minus1_decomposition():
    """Test the O(4) decomposition."""
    for i in range(10):
        # Use a predefined or randomly generated 4x4 matrix
        match i:
            case 0:
                U = np.diag([1, 1, 1, -1])
            case 1:
                U = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            case 2:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, -1, 0]])
            case 3:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
            case _:
                U = ortho_group(dim=4, seed=42).rvs()
                if np.linalg.det(U) == 1:
                    U[:, -1] = -U[:, -1]  # To have a determinant of -1
       
        # Test the decomposition
        decomp = o4_det_minus1_decomposition(U)
        reconstructed = np.eye(4)

        splitted_decomp = list()
        for gate, target in decomp:
            # Split the gate into elementary gates
            if type(gate) is not str:
                splitted_decomp.append((gate, target))
            elif " " not in gate:
                splitted_decomp.append((gate, target))
            else:
                for g in gate.split(" "):
                    splitted_decomp.append((g, target))

        # Reconstruct the matrix
        for gate, target in splitted_decomp:
            # Transform a string gate into a np.array gate
            match gate:
                case np.ndarray():
                    pass
                
                case "CNOT":
                    if target == (0, 1):
                        gate = np.array(
                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
                    elif target == (1, 0):
                        gate = np.array(
                            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

                case "H":
                    gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                
                case "T":
                    gate = np.array([[1, 0], [0, np.exp(1.j * np.pi / 4)]])

                case "TDAG":
                    gate = np.array([[1, 0], [0, np.exp(-1.j * np.pi / 4)]])
            
                case "S":
                    gate = np.array([[1, 0], [0, 1.j]])
                
                case "SDAG":
                    gate = np.array([[1, 0], [0, -1.j]])
                
                case _:
                    assert False, f"Unknown gate {gate}"

            # Transform any 2x2 matrix into a 4x4 matrix
            if gate.shape == (2, 2):
                if target == (0, ):
                    transformed_gate = np.kron(gate, np.eye(2))
                elif target == (1, ):
                    transformed_gate = np.kron(np.eye(2), gate)
                else:
                    assert False, f"Unknown target {target}"
            
            else:
                transformed_gate = gate

            reconstructed = transformed_gate @ reconstructed

        # Assert the reconstructed matrix is equal to the original matrix
        assert np.allclose(reconstructed, U, rtol=1e-8)


def test_o4_det_minus1_decomposition_errors():
    """Test the raise of errors when calling O(4) decomposition with wrong arguments."""
    # Shape error
    U = np.eye(3)
    with pytest.raises(ValueError, match="The input matrix must be 4x4. Got "):
        o4_det_minus1_decomposition(U)
    
    # Orthogonal error
    U = np.eye(4) * 1.1
    with pytest.raises(ValueError, match="The input matrix must be orthogonal."):
        o4_det_minus1_decomposition(U)
    
    # det != -1 error
    U = np.diag([1, 1, 1, 1])
    with pytest.raises(ValueError, match="The input matrix must have a determinant of -1. Got det"):
        o4_det_minus1_decomposition(U)    


def test_u4_decomposition():
    """Test the U(4) decomposition."""
    for i in range(10):
        i = 0
        # Use a predefined or randomly generated 4x4 matrix
        print()
        print(i)
        match i:
            case 0:
                U = np.diag([1, 1, 1, -1])
            case 1:
                U = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            case 2:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, -1, 0]])
            case 3:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
            case 4:
                U = np.eye(4)
            case 5:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0]])
            case 6:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, -1, 0]])
            case _:
                U = unitary_group(dim=4, seed=42).rvs()
        
        print(np.linalg.det(U))
       
        # Test the decomposition
        decomp = u4_decomposition(U)
        reconstructed = np.eye(4)

        splitted_decomp = list()
        for gate, target in decomp:
            # Split the gate into elementary gates
            if type(gate) is not str:
                splitted_decomp.append((gate, target))
            elif " " not in gate:
                splitted_decomp.append((gate, target))
            else:
                for g in gate.split(" "):
                    splitted_decomp.append((g, target))

        # Reconstruct the matrix
        for gate, target in splitted_decomp:
            # Transform a string gate into a np.array gate
            match gate:
                case np.ndarray():
                    pass
                
                case "CNOT":
                    if target == (0, 1):
                        gate = np.array(
                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
                    elif target == (1, 0):
                        gate = np.array(
                            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

                case "H":
                    gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                
                case "T":
                    gate = np.array([[1, 0], [0, np.exp(1.j * np.pi / 4)]])

                case "TDAG":
                    gate = np.array([[1, 0], [0, np.exp(-1.j * np.pi / 4)]])
            
                case "S":
                    gate = np.array([[1, 0], [0, 1.j]])
                
                case "SDAG":
                    gate = np.array([[1, 0], [0, -1.j]])
                
                case _:
                    assert False, f"Unknown gate {gate}"

            # Transform any 2x2 matrix into a 4x4 matrix
            if gate.shape == (2, 2):
                if target == (0, ):
                    transformed_gate = np.kron(gate, np.eye(2))
                elif target == (1, ):
                    transformed_gate = np.kron(np.eye(2), gate)
                else:
                    assert False, f"Unknown target {target}"
            
            else:
                transformed_gate = gate

            reconstructed = transformed_gate @ reconstructed

        print("Calculated")
        print(reconstructed)
        print("Expected")
        print(U)
        
        print()
        print()

        # Assert the reconstructed matrix is equal to the original matrix
        assert np.allclose(reconstructed, U, rtol=1e-8)

# np.set_printoptions(precision=3, suppress=True)
# U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, -1, 0]])
# test_canonical_decomposition(U)
# test_u4_decomposition()
