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
import math
import mpmath as mp
import time

from cliffordplust.grid_problem.rz_approx import z_rotational_approximation
from cliffordplust.exact_synthesis.exact_synthesis import exact_synthesis_alg

t1 = time.time()
epsilon = 1e-5
theta = 1 * math.pi / 128
mp.dps = int(-math.log10(epsilon**2)) + 8
U = z_rotational_approximation(epsilon, theta)
U_complex = np.array(U, dtype=complex)
rz = np.array([
    [math.cos(theta / 2) - 1.j * math.sin(theta / 2), 0], 
    [0, math.cos(theta / 2) + 1.j * math.sin(theta / 2)]
])
E = op_norm = max(np.linalg.svd(U_complex - rz, compute_uv=False))
print("Error: ", E)
Sequence = exact_synthesis_alg(U)
print("Sequence: ", Sequence)
t2 = time.time()
print("time: ", t2 - t1)