# Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math

import mpmath as mp
import numpy as np
import pytest

from qdecomp.utils.grid_problem.rz_approx import z_rotational_approximation

errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
angles = [2 * math.pi / 3, math.pi / 16, 6, 0, 3 * math.pi / 2, 2 * math.pi]


@pytest.mark.parametrize("theta", angles)
@pytest.mark.parametrize("epsilon", errors)
def test_rz_approx(theta, epsilon):
    """Test the z_rotational_approximation function for various epsilon and theta values."""
    # Set the decimal places for mpmath based on epsilon
    dps = int(-math.log10(epsilon**2)) + 8
    # Run the main function with the specified precision
    with mp.workdps(dps):
        U = z_rotational_approximation(theta, epsilon)
    U_complex = np.array(U, dtype=complex)
    # Calculate the expected Rz gate matrix
    rz = np.array(
        [
            [math.cos(theta / 2) - 1.0j * math.sin(theta / 2), 0],
            [0, math.cos(theta / 2) + 1.0j * math.sin(theta / 2)],
        ]
    )
    # Calculate the maximum error using SVD
    error = max(np.linalg.svd(U_complex - rz, compute_uv=False))
    # Assert that the error is within the specified epsilon
    assert error <= epsilon


def test_invalid_theta_type():
    """Test that z_rotational_approximation raises TypeError for invalid theta type."""
    with pytest.raises(TypeError):
        z_rotational_approximation("invalid_theta", 0.1)


def test_invalid_epsilon_type():
    """Test that z_rotational_approximation raises TypeError for invalid epsilon type."""
    with pytest.raises(TypeError):
        z_rotational_approximation(1.0, "invalid_epsilon")


def test_epsilon_too_large():
    """Test that z_rotational_approximation raises ValueError for epsilon >= 0.5."""
    with pytest.raises(ValueError):
        z_rotational_approximation(math.pi, 0.6)  # epsilon >= 0.5
