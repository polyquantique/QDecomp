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

import pytest
import numpy as np
import math
import mpmath as mp

from qdecomp.utils.grid_problem import z_rotational_approximation

errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
angles = [2 * math.pi / 3, math.pi / 16, 6, 0, 3 * math.pi / 2, 2 * math.pi]


@pytest.mark.parametrize("theta", angles)
@pytest.mark.parametrize("epsilon", errors)
def test_rz_approx(epsilon, theta):
    dps = int(-math.log10(epsilon**2)) + 8
    with mp.workdps(dps):
        U = z_rotational_approximation(epsilon, theta)
    U_complex = np.array(U, dtype=complex)
    rz = np.array(
        [
            [math.cos(theta / 2) - 1.0j * math.sin(theta / 2), 0],
            [0, math.cos(theta / 2) + 1.0j * math.sin(theta / 2)],
        ]
    )
    Error = op_norm = max(np.linalg.svd(U_complex - rz, compute_uv=False))
    assert Error <= epsilon


def test_invalid_theta_type():
    with pytest.raises(TypeError):
        z_rotational_approximation(0.1, "invalid_theta")


def test_invalid_epsilon_type():
    with pytest.raises(TypeError):
        z_rotational_approximation("invalid_epsilon", 1.0)


def test_theta_out_of_range():
    with pytest.raises(ValueError):
        z_rotational_approximation(0.1, 5 * math.pi)  # theta > 4π

    with pytest.raises(ValueError):
        z_rotational_approximation(0.1, -1)  # theta < 0


def test_epsilon_too_large():
    with pytest.raises(ValueError):
        z_rotational_approximation(0.6, math.pi)  # epsilon >= 0.5
