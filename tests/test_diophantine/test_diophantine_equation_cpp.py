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

import itertools

import pytest

from cliffordplust.diophantine import *


@pytest.mark.parametrize(
    "a, a_, b, b_", itertools.product(range(-1, 10), range(0, 3), range(-5, 4), range(0, 3))
)
def test_solve_xi_eq_ttdag_in_d_cpp(a, a_, b, b_):
    """
    Test the solve_xi_eq_ttdag_in_d() function.
    """
    xi = Dsqrt2(D(a, a_), D(b, b_))

    t_cpp = solve_xi_eq_ttdag_in_d_cpp(xi)

    if t_cpp is not None:
        recombination = t_cpp * t_cpp.complex_conjugate()

        assert recombination == Domega.from_ring(xi)
        assert float(xi) >= 0 and float(xi.sqrt2_conjugate()) >= 0  # xi is doubly positive

    else:
        t_py = solve_xi_eq_ttdag_in_d(xi)
        assert t_py is None  # Both implementations should return None if there is no solution
