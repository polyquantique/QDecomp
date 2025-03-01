import numpy as np
import math
import time

from cliffordplust.grid_problem.rz_approx import z_rotational_approximation
from cliffordplust.exact_synthesis.exact_synthesis import exact_synthesis_alg

t1 = time.time()
epsilon = 1e-3
theta = 2 * math.pi / 3
U = z_rotational_approximation(epsilon, theta)
print(U)
U_complex = np.array(U, dtype=complex)
print(U_complex)
rz = np.array([
    [np.exp(-theta / 2 * 1j), 0], 
    [0, np.exp(theta / 2 * 1j)]
])
E = op_norm = max(np.linalg.svd(U_complex - rz, compute_uv=False))
print("Error: ", E)
Sequence = exact_synthesis_alg(U)
print("Sequence: ", Sequence)
t2 = time.time()
print("time: ", t2 - t1)