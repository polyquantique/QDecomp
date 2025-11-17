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

import mpmath as mp
import numpy as np
import pytest

from qdecomp.utils.grid_problem.rz_approx_cpp import rz_approx_cpp

errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
angles = [2 * np.pi / 3, np.pi / 16, 6, 0, 3 * np.pi / 2, 2 * np.pi]


@pytest.mark.parametrize("theta", angles)
@pytest.mark.parametrize("epsilon", errors)
def test_rz_approx(theta, epsilon):
    """Test the rz_approx_cpp function for various epsilon and theta values."""
    # Set the decimal places for mpmath based on epsilon
    dps = int(-np.log10(epsilon**2)) + 8
    # Run the main function with the specified precision
    with mp.workdps(dps):
        U = rz_approx_cpp(theta, epsilon)
    U_complex = np.array(U, dtype=complex)
    # Calculate the expected Rz gate matrix
    rz = np.array(
        [
            [np.cos(theta / 2) - 1.0j * np.sin(theta / 2), 0],
            [0, np.cos(theta / 2) + 1.0j * np.sin(theta / 2)],
        ]
    )
    # Calculate the maximum error using SVD
    error = max(np.linalg.svd(U_complex - rz, compute_uv=False))
    # Assert that the error is within the specified epsilon
    assert error <= epsilon


def test_invalid_theta_type():
    """Test that rz_approx_cpp raises TypeError for invalid theta type."""
    with pytest.raises(TypeError):
        rz_approx_cpp("invalid_theta", 0.1)


def test_invalid_epsilon_type():
    """Test that rz_approx_cpp raises TypeError for invalid epsilon type."""
    with pytest.raises(TypeError):
        rz_approx_cpp(1.0, "invalid_epsilon")


def test_epsilon_too_large():
    """Test that rz_approx_cpp raises ValueError for epsilon >= 0.5."""
    with pytest.raises(ValueError):
        rz_approx_cpp(np.pi, 0.6)  # epsilon >= 0.5
