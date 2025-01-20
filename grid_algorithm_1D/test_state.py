import pytest
import numpy as np
import math
from Rings import D, Zsqrt2, Dsqrt2, lamb, inv_lamb
from grid_operator import Grid_Operator
from state import State


def test_init_valid_matrices():
    A = np.array([[1, 0], [0, 1]], dtype=float)
    B = np.array([[2, 1], [1, 2]], dtype=float)

    state = State(A, B)

    assert isinstance(state, State)


def test_init_invalid_type_A():
    A = "not valid"  # Invalid type
    B = np.array([[2, 1], [1, 2]], dtype=float)

    with pytest.raises(TypeError):
        State(A, B)


def test_init_invalid_type_B():
    A = np.array([[1, 0], [0, 1]], dtype=float)
    B = "not valid"  # Invalid type

    with pytest.raises(TypeError):
        State(A, B)


def test_init_invalid_shape():
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3x3 matrix
    B = np.array([[2, 1], [1, 2]], dtype=float)

    with pytest.raises(ValueError):
        State(A, B)


def test_init_non_symmetric_matrix_A():
    A = np.array([[1, 0], [1, 1]], dtype=float)  # Not symmetric
    B = np.array([[2, 1], [1, 2]], dtype=float)

    with pytest.raises(ValueError):
        State(A, B)


def test_init_non_symmetric_matrix_B():
    A = np.array([[1, 0.5], [0.5, 1]], dtype=float)
    B = np.array([[2, 1], [0.5, 2]], dtype=float)  # Not symmetric

    with pytest.raises(ValueError):
        State(A, B)


def test_reduce_determinant_normalization():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)  # Det(A) = 1
    B = np.array([[4, 2], [2, 2]], dtype=float)  # Det(B) = 4

    state = State(A, B)

    # Check if determinants are normalized to 1
    assert np.isclose(np.linalg.det(state.A), 1)
    assert np.isclose(np.linalg.det(state.B), 1)


def test_reduce_no_change_when_determinant_is_1():
    A = np.array([[1, 0], [0, 1]], dtype=float)  # Det(A) = 1
    B = np.array([[1, 0], [0, 1]], dtype=float)  # Det(B) = 1

    state = State(A, B)

    # Check if the matrices remain unchanged when determinants are already 1
    assert np.array_equal(state.A, A)
    assert np.array_equal(state.B, B)


def test_reduce_scaling_matrices():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)  # Det(A) = 1
    B = np.array([[4.5, 2], [2, 1]], dtype=float)  # Det(B) = 0.5

    state = State(A, B)

    # After reduction, determinants of both A and B should be 1
    print(np.linalg.det(state.A), np.linalg.det(state.B))
    assert np.isclose(np.linalg.det(state.A), 1)
    assert np.isclose(np.linalg.det(state.B), 1)


def test_reduce_zero_determinant_raises_error():
    A = np.array([[0, 0], [0, 0]], dtype=float)  # Det(A) = 0
    B = np.array([[1, 1], [1, 1]], dtype=float)  # Det(B) = 0

    with pytest.raises(
        ValueError, match="The determinant of A and B must be positive and non-zero"
    ):
        State(A, B)


def test_reduce_single_matrix_zero_determinant():
    A = np.array([[0, 0], [0, 0]], dtype=float)  # Det(A) = 0
    B = np.array([[2, 1], [1, 2]], dtype=float)  # Det(B) = 2

    with pytest.raises(
        ValueError, match="The determinant of A and B must be positive and non-zero"
    ):
        State(A, B)


def test_repr():
    A = np.array([[1, 0], [0, 1]], dtype=float)
    B = np.array([[2, 1], [1, 2]], dtype=float)

    state = State(A, B)

    expected_repr = f"({state.A}, {state.B})"
    assert repr(state) == expected_repr


def test_z_property():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)  # Example matrix
    B = np.array([[2, 1], [1, 2]], dtype=float)

    state = State(A, B)

    # The exact value is based on the formula in the method
    expected_z = -0.5 * math.log(state.A[0, 0] / state.A[1, 1]) / math.log(1 + math.sqrt(2))
    assert np.isclose(state.z, expected_z)


def test_zeta_property():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)
    B = np.array([[2, 1], [1, 2]], dtype=float)  # Example matrix

    state = State(A, B)

    # The exact value is based on the formula in the method
    expected_zeta = -0.5 * math.log(state.B[0, 0] / state.B[1, 1]) / math.log(1 + math.sqrt(2))
    assert np.isclose(state.zeta, expected_zeta)


def test_e_property():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)  # Example matrix
    B = np.array([[2, 1], [1, 2]], dtype=float)

    state = State(A, B)

    expected_e = state.A[0, 0] * (1 + math.sqrt(2)) ** state.z
    assert np.isclose(state.e, expected_e)


def test_epsilon_property():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)
    B = np.array([[2, 1], [1, 2]], dtype=float)  # Example matrix

    state = State(A, B)

    expected_epsilon = state.B[0, 0] * (1 + math.sqrt(2)) ** state.zeta
    assert np.isclose(state.epsilon, expected_epsilon)


def test_b_property():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)  # Example matrix
    B = np.array([[2, 1], [1, 2]], dtype=float)

    state = State(A, B)

    expected_b = float(state.A[0, 1])
    assert np.isclose(state.b, expected_b)


def test_beta_property():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)
    B = np.array([[2, 1], [1, 2]], dtype=float)  # Example matrix

    state = State(A, B)

    expected_beta = float(state.B[0, 1])
    assert np.isclose(state.beta, expected_beta)


def test_skew_property():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)  # Example matrix
    B = np.array([[2, 1], [1, 2]], dtype=float)

    state = State(A, B)

    expected_skew = state.b**2 + state.beta**2
    assert np.isclose(state.skew, expected_skew)


def test_bias_property():
    A = np.array([[2, 0], [0, 0.5]], dtype=float)  # Example matrix
    B = np.array([[2, 1], [1, 2]], dtype=float)

    state = State(A, B)

    expected_bias = state.zeta - state.z
    assert np.isclose(state.bias, expected_bias)


def test_transform():
    A = np.array([[1, 0], [0, 1]], dtype=float)
    B = np.array([[2.5, math.sqrt(2)], [math.sqrt(2), 1]], dtype=float)
    G_op = Grid_Operator([1, Zsqrt2(0, 1), 0, 1])
    exp_A = np.array([[1, math.sqrt(2)], [math.sqrt(2), 3]], dtype=float)
    exp_B = np.array([[5 / math.sqrt(2), -3], [-3, 2 * math.sqrt(2)]], dtype=float)

    state = State(A, B)
    trans_state = state.transform(G_op)

    assert np.allclose(trans_state.A, exp_A), f"Expected {exp_A} but got {trans_state.A}"
    assert np.allclose(trans_state.B, exp_B), f"Expected {exp_B} but got {trans_state.B}"


def test_transform_error():
    A = np.array([[1, 0], [0, 1]], dtype=float)
    B = np.array([[2.5, math.sqrt(2)], [math.sqrt(2), 1]], dtype=float)
    G_op = [1, Zsqrt2(0, 1), 0, 1]

    state = State(A, B)
    with pytest.raises(TypeError, match="G must be a grid operator"):
        state.transform(G_op)


def test_shift_positive():
    A = np.array([[1, math.sqrt(2)], [math.sqrt(2), 3]], dtype=float)
    B = np.array([[5 / math.sqrt(2), -3], [-3, 2 * math.sqrt(2)]], dtype=float)
    exp_A = np.array(
        [[float(lamb**2), math.sqrt(2)], [math.sqrt(2), 3 * float(inv_lamb**2)]],
        dtype=float,
    )
    exp_B = np.array(
        [
            [5 * float(inv_lamb**2) / math.sqrt(2), -3],
            [-3, 2 * float(lamb**2) * math.sqrt(2)],
        ],
        dtype=float,
    )

    state = State(A, B)
    shifted = state.shift(2)

    assert np.allclose(shifted.A, exp_A), f"Expected {exp_A} but got {shifted.A}"
    assert np.allclose(shifted.B, exp_B), f"Expected {exp_B} but got {shifted.B}"


def test_shift_negative():
    A = np.array([[1, math.sqrt(2)], [math.sqrt(2), 3]], dtype=float)
    B = np.array([[5 / math.sqrt(2), -3], [-3, 2 * math.sqrt(2)]], dtype=float)
    exp_A = np.array(
        [[float(inv_lamb), math.sqrt(2)], [math.sqrt(2), 3 * float(lamb)]], dtype=float
    )
    exp_B = np.array(
        [[5 * float(lamb) / math.sqrt(2), 3], [3, 2 * float(inv_lamb) * math.sqrt(2)]],
        dtype=float,
    )

    state = State(A, B)
    shifted = state.shift(-1)

    assert np.allclose(shifted.A, exp_A), f"Expected {exp_A} but got {shifted.A}"
    assert np.allclose(shifted.B, exp_B), f"Expected {exp_B} but got {shifted.B}"


def test_invalid_shift():
    A = np.array([[1, 0], [0, 1]], dtype=float)
    B = np.array([[2.5, math.sqrt(2)], [math.sqrt(2), 1]], dtype=float)
    k = 1.37

    state = State(A, B)
    with pytest.raises(ValueError, match="k must be an integer"):
        state.shift(k)
