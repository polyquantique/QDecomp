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

import numpy as np
import pytest

from qdecomp.decompositions import zyz_decomposition


# Rotation and phase matrices
def ry(teta):
    return np.array([[np.cos(teta / 2), -np.sin(teta / 2)], [np.sin(teta / 2), np.cos(teta / 2)]])


def rz(teta):
    return np.array([[np.exp(-1.0j * teta / 2), 0], [0, np.exp(1.0j * teta / 2)]])


def phase(alpha):
    return np.exp(1.0j * alpha)


@pytest.mark.parametrize(
    "a, b",
    [
        (0, 1),
        (0.5, np.sqrt(3) / 2),
        (1, 0),
        (-1, 0),
        (-0.7, np.sqrt(51) / 10),
        (complex(1, 1) / np.sqrt(2), 0),
        (complex(2, -3) / 4, np.sqrt(3) / 4),
        (1e-10, 1),
        (1 - 1e-10, np.sqrt(2e-10)),
        (1 - 1e-16, np.sqrt(2e-16)),
    ],
)
@pytest.mark.parametrize("alpha", [0, 1, np.pi, np.pi / 2, 2 * np.pi])
def test_zyz_decomposition(a, b, alpha):
    """
    Test the ZYZ decomposition of a 2x2 unitary matrix.
    """
    U = np.exp(1.0j * alpha) * np.array([[a, -b.conjugate()], [b, a.conjugate()]])  # Unitary matrix

    t0, t1, t2, alpha_ = zyz_decomposition(U)

    # Check that the decomposition is correct
    U_calculated = phase(alpha_) * rz(t2) @ ry(t1) @ rz(t0)

    assert np.allclose(U, U_calculated, atol=1e-7, rtol=1e-7)


def test_zyz_decomposition_unitary_error():
    """
    Test the errors raised by the zyz_decomposition() function when the matrix is not unitary.
    """
    U = np.array([[1, 0], [0, 2]])
    with pytest.raises(
        ValueError, match="The input matrix must be unitary. Got a matrix with determinant"
    ):
        zyz_decomposition(U)


def test_zyz_decomposition_shape_error():
    """
    Test the errors raised by the zyz_decomposition() function when the matrix is not 2x2.
    """
    U = np.eye(3)
    with pytest.raises(
        ValueError, match=r"The input matrix must be 2x2. Got a matrix with shape \(3, 3\)."
    ):
        zyz_decomposition(U)
