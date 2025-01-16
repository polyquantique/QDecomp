import pytest
from cnot_decomposition import *
import numpy as np
from scipy.stats import unitary_group

@pytest.mark.parametrize("A", [np.random.uniform(-100, 100, (2, 2)) for _ in range(10)])
@pytest.mark.parametrize("B", [np.random.uniform(-100, 100, (2, 2)) for _ in range(10)])
def test_kronecker_decomposition(A, B):
    """Test the kronecker decomposition of 4x4 matrix."""
    M = np.kron(A, B)
    a, b = kronecker_decomposition(M)
    assert np.allclose(M, np.kron(a, b))

@pytest.mark.parametrize("U", [unitary_group.rvs(4) for _ in range(40)])
def test_canonical_decomposition(U):
    """Test the canonical decomposition of 4x4 unitary matrix."""
    A, B, t, alpha = canonical_decomposition(U)
    assert np.allclose(U, B@can(t[0], t[1], t[2])@A*np.exp(1.j*alpha))

def test_kronecker_decomposition_errors():
    """Test the raise of errors when calling kronecker decomposition with wrong arguments."""
    for M in [((1, 2), (3, 4)), [[1, 2], [3, 4]], 1, "1", 1.0]:
        with pytest.raises(TypeError, match="Matrix must be a numpy object"):
            kronecker_decomposition(M)
    for M in [np.random.randint(-100, 100, (3, 3)), np.random.randint(-100, 100, (3, 4)), np.random.randint(-100, 100, (4, 3)), np.random.randint(-100, 100, (5, 5))]:
        with pytest.raises(ValueError, match="Matrix must be 4x4"):
            kronecker_decomposition(M)

def test_canonical_decomposition_errors():
    """Test the raise of errors when calling canonical decomposition with wrong arguments."""
    for M in [((1, 2), (3, 4)), [[1, 2], [3, 4]], 1, "1", 1.0]:
        with pytest.raises(TypeError, match="Matrix U must be a numpy object"):
            canonical_decomposition(M)
    for M in [np.random.randint(-100, 100, (3, 3)), np.random.randint(-100, 100, (3, 4)), np.random.randint(-100, 100, (4, 3)), np.random.randint(-100, 100, (5, 5)), unitary_group.rvs(4) * 2]:
        with pytest.raises(ValueError, match="M must be a 4x4 unitary matrix"):
            canonical_decomposition(M)