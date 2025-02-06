import pytest
import numpy as np
import math

from rz_approx import z_rotational_approximation

angles = np.linspace(0, 4 * math.pi, 20)
values = [1e-2]

@pytest.mark.parametrize("theta", angles)
@pytest.mark.parametrize("epsilon", values)
def test_solutions(epsilon, theta):
    U = z_rotational_approximation(epsilon, theta)
    rz = np.array([[math.exp(-theta * complex(0 + 1.j)), 0][0, math.exp(theta * complex(0 + 1.j))]])
    E = np.linalg.norm(rz - U, "fro")
    assert E <= epsilon