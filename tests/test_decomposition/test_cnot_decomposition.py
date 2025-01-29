import math

import numpy as np
import pytest
from scipy.stats import unitary_group

from cliffordplust.decomposition import *


@pytest.mark.parametrize(
    "A",
    [np.random.uniform(-100, 100, (2, 2)) for _ in range(10)]
    + [np.zeros((2, 2)), np.identity(2), np.arange(1, 5).reshape(2, 2), np.ones((2, 2))]
)
@pytest.mark.parametrize(
    "B",
    [np.random.uniform(-100, 100, (2, 2)) for _ in range(10)]
    + [np.zeros((2, 2)), np.identity(2), np.arange(1, 5).reshape(2, 2), np.ones((2, 2))]
)
def test_kronecker_decomposition(A, B):
    """Test the kronecker decomposition of 4x4 matrix."""
    M = np.kron(A, B)
    a, b = kronecker_decomposition(M)
    assert np.allclose(M, np.kron(a, b))


@pytest.mark.parametrize(
    "U",
    [unitary_group.rvs(4) for _ in range(40)]
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
    ],
)
def test_canonical_decomposition(U):
    """Test the canonical decomposition of 4x4 unitary matrix."""
    A, B, t, alpha = canonical_decomposition(U)
    assert np.allclose(U, B @ can(t[0], t[1], t[2]) @ A * np.exp(1.0j * alpha), rtol=1e-8)
    a, b = kronecker_decomposition(A)
    alpha, beta = kronecker_decomposition(B)
    assert np.allclose(A, np.kron(a, b)) and np.allclose(B, np.kron(alpha, beta))


def test_kronecker_decomposition_errors():
    """Test the raise of errors when calling kronecker decomposition with wrong arguments."""
    for M in [((1, 2), (3, 4)), [[1, 2], [3, 4]], 1, "1", 1.0]:
        with pytest.raises(TypeError, match="Matrix must be a numpy object"):
            kronecker_decomposition(M)
    for M in [
        np.arange(1, 10).reshape(3, 3),
        np.arange(1, 13).reshape(3, 4),
        np.arange(1, 13).reshape(4, 3),
        np.arange(1, 26).reshape(5, 5),
    ]:
        with pytest.raises(ValueError, match="Matrix must be 4x4"):
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
