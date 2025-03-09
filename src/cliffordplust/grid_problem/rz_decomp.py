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

def rz_decomp(epsilon: float, theta: float):
    t1 = time.time()
    dps = int(-math.log10(epsilon**2)) + 8
    with mp.workdps(dps):
        U = z_rotational_approximation(epsilon, theta)
    U_complex = np.array(U, dtype=complex)
    rz = np.array([
        [math.cos(theta / 2) - 1.j * math.sin(theta / 2), 0], 
        [0, math.cos(theta / 2) + 1.j * math.sin(theta / 2)]
    ])
    Error = op_norm = max(np.linalg.svd(U_complex - rz, compute_uv=False))
    Sequence = exact_synthesis_alg(U)
    t2 = time.time()
    duration = t2 - t1
    return Sequence, Error, duration

# Sequence, Error, duration = rz_decomp(1e-7, 4 * math.pi / 3)
# print("Sequence: ", Sequence)
# print("Error: ", Error)
# print("time: ", duration)