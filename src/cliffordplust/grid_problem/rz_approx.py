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

from cliffordplust.grid_problem.grid_problem import find_points, find_grid_operator
from cliffordplust.grid_problem.grid_algorithms import solve_grid_problem_2d
from cliffordplust.rings.rings import *
# import cliffordplust.diophantine.diophantine_equation_cpp as diop
from cliffordplust.diophantine.diophantine_equation import *
from cliffordplust.grid_problem.steiner_ellipse import *

def initialization(epsilon: float, theta: float):
    p1, p2, p3 = find_points(epsilon, theta)
    E, p_p = steiner_ellipse_def(p1, p2, p3)
    I = np.array([[mp.mpf(1), mp.mpf(0)], 
              [mp.mpf(0), mp.mpf(1)]], dtype=object)
    inv_gop, gop = find_grid_operator(E, I)
    inv_gop_conj = inv_gop.conjugate()
    mod_E = (inv_gop.dag()).as_mpmath() @ E @ inv_gop.as_mpmath()
    mod_D = (inv_gop_conj.dag()).as_mpmath() @ I @ inv_gop_conj.as_mpmath()
    bbox_1 = ellipse_bbox(mod_E, p_p)
    bbox_2 = ellipse_bbox(mod_D, [mp.mpf(0), mp.mpf(0)])
    return E, p_p, bbox_1, bbox_2

def z_rotational_approximation(epsilon: float, theta: float) -> np.ndarray:
    exponent = round(2 * theta / math.pi)
    if np.isclose(2 * theta / math.pi, exponent):
        T = np.array([[Domega(D(1, 0), D(0, 0), D(0, 0), D(0, 0)), Domega.from_ring(0)], [Domega.from_ring(0), Domega(D(0, 0), D(0, 0), D(1, 0), D(0, 0))]], dtype=object)
        M = T ** exponent
        return M
    E, p_p, bbox_1, bbox_2 = initialization(epsilon, theta)
    I = np.array([[mp.mpf(1), mp.mpf(0)], 
              [mp.mpf(0), mp.mpf(1)]], dtype=object)
    z = np.array([mp.cos(theta / 2), -mp.sin(theta / 2)])
    n = 0
    solution = False
    while solution == False:
        odd = n % 2
        if odd:
            const = Dsqrt2(D(0, 0), D(1, int((n + 1) / 2)))
        else:
            const = D(1, int(n / 2))
        A = mp.sqrt(2 ** n) * bbox_1
        if odd:
            bbox_2_flip = np.array([[bbox_2[0, 1], bbox_2[0, 0]], [bbox_2[1, 1], bbox_2[1, 0]]])
            B = -mp.sqrt(2 ** n) * bbox_2_flip
        else: 
            B = mp.sqrt(2 ** n) * bbox_2
        U = solve_grid_problem_2d(A.tolist(), B.tolist())
        print(f"Found {len(U)} solutions")
        for candidate in U:
            if n > 0 and (abs(candidate.a - candidate.c) % 2 == 1 or abs(candidate.b - candidate.d) % 2 ==1):
                u = Domega.from_ring(candidate) * Domega.from_ring(const)
                u_vec = np.array([Dsqrt2(u.d, D(1, 1) * (u.c - u.a)), Dsqrt2(u.b, D(1, 1) * (u.c + u.a))])
                u_conj = u.sqrt2_conjugate()
                u_conj_vec = np.array([Dsqrt2(u_conj.d, D(1, 1) * (u_conj.c - u_conj.a)), Dsqrt2(u_conj.b, D(1, 1) * (u_conj.c + u_conj.a))])
                u_float = np.array([u_vec[0].mpfloat(), u_vec[1].mpfloat()])
                u_conj_float = np.array([u_conj_vec[0].mpfloat(), u_conj_vec[1].mpfloat()])
                dot = np.dot(u_float, z)
                delta = mp.mpf(1) - mp.mpf(0.5 * epsilon**2)
                if dot < 1 and dot > delta and is_inside_ellipse(u_conj_float, I, np.zeros(2)):
                    print("Found candidate")
                    xi = 1 - u.complex_conjugate() * u
                    # t = diop.solve_xi_eq_ttdag_in_d_cpp(Dsqrt2.from_ring(xi))
                    t = solve_xi_eq_ttdag_in_d(Dsqrt2.from_ring(xi))
                    if t is None:
                        print("Failed")
                    else:
                        solution = True
                        print(Dsqrt2.from_ring(xi))
                        M = np.array([[u, -t.complex_conjugate()], [t, u.complex_conjugate()]])
                        return M
        print("Denominator exponent: ", n)
        n += 1