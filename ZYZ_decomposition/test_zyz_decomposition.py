import itertools

import pytest

from zyz_decomposition import *


@pytest.mark.parametrize(
    "a, alpha",
    itertools.product(
        [
            0,
            0.5,
            1,
            -1,
            -0.7,
            complex(1, 1) / np.sqrt(2),
            complex(2, -3) / 4,
            1e-10,
            1 - 1e-10,
            1 - 1e-16,
        ],
        [0, 1, np.pi, np.pi / 2, 2 * np.pi],
    ),
)
def test_zyz_decomposition(a, alpha):
    """
    Test the ZYZ decomposition of a 2x2 unitary matrix.
    """
    b = np.sqrt(complex(1, 0) - np.abs(a) ** 2)  # Find b such that U is unitary
    U = np.exp(1.0j * alpha) * np.array([[a, -b.conjugate()], [b, a.conjugate()]])  # Unitary matrix

    t0, t1, t2, alpha_ = zyz_decomposition(U)

    # Check that the decomposition is correct
    U_calculated = phase(alpha_) * Rz(t2) @ Ry(t1) @ Rz(t0)

    assert np.allclose(U, U_calculated, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(
    "U, error_message",
    [
        [
            np.array([[1, 0], [0, 2]]),
            "The input matrix must be unitary. Got a matrix with determinant",
        ],
        [np.eye(3), "The input matrix must be 2x2. Got a matrix with shape "],
    ],
)
def test_zyz_decomposition_error(U, error_message):
    """
    Test the errors raised by the zyz_decomposition() function.
    """
    with pytest.raises(ValueError, match=error_message):
        zyz_decomposition(U)
