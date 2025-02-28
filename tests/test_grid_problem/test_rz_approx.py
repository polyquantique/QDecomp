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