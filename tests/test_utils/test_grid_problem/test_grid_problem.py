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
from qdecomp.utils.grid_problem.grid_problem import *
from qdecomp.utils.steiner_ellipse import *


def test_find_points_type_error():
    with pytest.raises(TypeError, match="Both theta and epsilon must be convertible to floats."):
        find_points("", [5, 2, 4, 5])


def test_find_points_value_error_epsilon():
    with pytest.raises(ValueError, match="The maximal allowable error is 0.5."):
        find_points(2.3345, 1.5098)


angles = np.linspace(0, 4 * math.pi - 0.25, 20)
values = [0.25, 1e-2, 1e-4, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128]


@pytest.mark.parametrize("theta", angles)
@pytest.mark.parametrize("epsilon", values)
def test_find_points_valid_solutions(epsilon, theta):
    dps = int(-mp.log10(epsilon**2)) + 8  # Set the decimal places for mpmath
    with mp.workdps(dps):
        p1, p2, p3 = find_points(epsilon, theta)

        n1 = mp.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
        n2 = 2 * mp.mpf(epsilon) * mp.sqrt(mp.mpf(1) - mp.mpf(epsilon) ** 2 / 4)

    assert np.isclose(float(n1), float(n2))


errors = [0.25, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


@pytest.mark.parametrize("theta", angles)
@pytest.mark.parametrize("epsilon", errors)
def test_grid_op(epsilon, theta):
    dps = int(-mp.log10(epsilon**2)) + 8  # Set the decimal places for mpmath
    with mp.workdps(dps):
        p1, p2, p3 = find_points(epsilon, theta)
        E, p_p = steiner_ellipse_def(p1, p2, p3)
        I = np.array([[mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(1)]], dtype=object)
        initial_state = State(E, I)
        inv_gop, gop = find_grid_operator(E, I)
        inv_gop_conj = inv_gop.conjugate()
        mod_E = inv_gop.dag().as_mpfloat() @ E @ inv_gop.as_mpfloat()
        mod_D = inv_gop_conj.dag().as_mpfloat() @ inv_gop_conj.as_mpfloat()
        state = State(mod_E, mod_D)
    assert state.skew <= 15
