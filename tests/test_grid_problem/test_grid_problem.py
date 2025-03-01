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
from grid_problem import *

def test_find_points_type_error():
    with pytest.raises(TypeError, match="Both theta and epsilon must be convertible to floats."):
        find_points('', [5, 2, 4, 5])
    
def test_find_points_value_error_theta():
    with pytest.raises(ValueError, match="The value of theta must be between 0 and 4\u03C0."):
        find_points(0.2, 15)

def test_find_points_value_error_epsilon():
    with pytest.raises(ValueError, match="The maximal allowebale error is 0.5."):
        find_points(2.3345, 1.5098)

angles = np.linspace(0, 4 * math.pi, 20)
values = [0.25, 1e-2, 1e-4, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128]

@pytest.mark.parametrize("theta", angles)
@pytest.mark.parametrize("epsilon", values)
def test_valid_slolutions(epsilon, theta):
    p1, p2, p3 = find_points(epsilon, theta)

    n1 = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
    n2 = 2 * math.sqrt(epsilon ** 2 - epsilon ** 4 / 4)

    assert np.isclose(n1, n2)
    assert np.allclose(p1, 0)

