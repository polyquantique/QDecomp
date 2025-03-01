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

from cliffordplust.grid_problem.grid_problem import find_points, find_grid_operator
from cliffordplust.grid_problem.grid_algorithms import solve_grid_problem_2d
from cliffordplust.rings import *
import cliffordplust.diophantine.diophantine_equation_cpp as diop
# from cliffordplust.diophantine.diophantine_equation import solve_xi_eq_ttdag_in_d
from cliffordplust.grid_problem.steiner_ellipse import steiner_ellipse_def, ellipse_bbox, is_inside_ellipse

def z_rotational_approximation(epsilon: float, theta: float) -> np.ndarray:
    p1, p2, p3 = find_points(epsilon, theta)
    E, p_p = steiner_ellipse_def(p1, p2, p3)
    z = np.array([math.cos(theta/2), -math.sin(theta/2)])
    I = np.array([[1, 0], [0, 1]], dtype=float)
    inv_gop, gop = find_grid_operator(E, I)
    gop_conj = gop.conjugate()
    inv_gop_conj = inv_gop.conjugate()
    n = 0
    i = Domega(D(0, 0), D(0, 0), D(1, 0), D(0, 0))
    solution = False
    mod_E = inv_gop.dag().as_float() @ E @ inv_gop.as_float()
    mod_D = inv_gop_conj.dag().as_float() @ inv_gop_conj.as_float()
    bbox_1 = ellipse_bbox(mod_E, p_p)
    bbox_2 = ellipse_bbox(mod_D, np.zeros(2))
    while solution == False:
        odd = n % 2
        if odd:
            const = Dsqrt2(D(0, 0), D(1, int((n + 1) / 2)))
        else:
            const = D(1, int(n / 2))
        A = math.sqrt(2 ** n) * bbox_1
        if odd:
            bbox_2_flip = np.array([[bbox_2[0, 1], bbox_2[0, 0]], [bbox_2[1, 1], bbox_2[1, 0]]])
            B = -math.sqrt(2 ** n) * bbox_2_flip
        else: 
            B = math.sqrt(2 ** n) * bbox_2
        U = solve_grid_problem_2d(A.tolist(), B.tolist())
        for candidate in U:
            v = Domega.from_ring(candidate) * Domega.from_ring(const)
            v_conj = v.sqrt2_conjugate()
            v_float = np.array([v.real(), v.imag()])
            v_conj_float = np.array([v_conj.real(), v_conj.imag()])
            if is_inside_ellipse(v_float, E, p_p) and is_inside_ellipse(v_conj_float, I, np.zeros(2)):
                v_vec = np.array([Dsqrt2(v.d, D(1, 1) * (v.c - v.a)), Dsqrt2(v.b, D(1, 1) * (v.c + v.a))])
                u_vec = v_vec
                u_float = np.array(u_vec, dtype=float)
                u = Domega.from_ring(u_vec[0]) + i * Domega.from_ring(u_vec[1])
                if np.dot(u_float, z) < 1 and np.dot(u_float, z) > 1 - 0.5 * epsilon**2:
                    print("Found candidate")
                    print(u)
                    print(u_float)
                    print(np.dot(u_float, z))
                    xi = 1 - u.complex_conjugate() * u
                    t = diop.solve_xi_eq_ttdag_in_d_cpp(Dsqrt2.from_ring(xi))
                    # t = solve_xi_eq_ttdag_in_d(Dsqrt2.from_ring(xi))
                    if t is None:
                        print("Failed")
                    else:
                        solution = True
                        M = np.array([[u, -t.complex_conjugate()], [t, u.complex_conjugate()]])
                        return M
        print(n)
        n += 1