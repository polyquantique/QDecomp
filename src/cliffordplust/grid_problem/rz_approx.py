import numpy as np
import math

from grid_problem import find_points, find_grid_operator
from grid_operator import Grid_Operator
from grid_algorithm_2D import solve_grid_problem_2D
from Rings import *
# import cliffordplust.diophantine.diophantine_equation_cpp as diop
from diophantine_equation import solve_xi_eq_ttdag_in_d
from steiner_ellipse import steiner_ellipse_def, ellipse_bbox, is_inside_ellipse, plot_ellipse

def z_rotational_approximation(epsilon: float, theta: float) -> np.ndarray:
    p1, p2, p3 = find_points(epsilon, theta)
    E, _ = steiner_ellipse_def(p1, p2, p3)
    p_p = (1 - epsilon**2 / 2) * np.array([math.cos(theta/2), -math.sin(theta/2)])
    I = np.array([[1, 0], [0, 1]], dtype=float)
    inv_gop, gop = find_grid_operator(E, I)
    gop_conj = gop.conjugate()
    inv_gop_conj = inv_gop.conjugate()
    n = 0
    solution = False
    mod_E = inv_gop.dag().as_float() @ E @ inv_gop.as_float()
    mod_D = inv_gop_conj.dag().as_float() @ inv_gop_conj.as_float()
    bbox_1 = ellipse_bbox(mod_E, p_p)
    bbox_2 = ellipse_bbox(mod_D, np.zeros(2))
    plot_ellipse(E, p_p)
    plot_ellipse(mod_E, p_p)
    plot_ellipse(mod_D, np.zeros(2))
    while solution == False:
        odd = n % 2
        if odd:
            const = Dsqrt2(D(0, 0), D(1, int((n + 1) / 2)))
        else:
            const = D(1, int(n / 2))
        i = Zomega(0, 0, 1, 0)
        A = math.sqrt(2 ** n) * bbox_1
        if odd:
            bbox_2_flip = np.array([[bbox_2[0, 1], bbox_2[0, 0]], [bbox_2[1, 1], bbox_2[1, 0]]])
            B = -math.sqrt(2 ** n) * bbox_2_flip
        else: 
            B = math.sqrt(2 ** n) * bbox_2
        U = solve_grid_problem_2D(A.tolist(), B.tolist())
        for candidate in U:
            u = candidate * const
            u_conj = u.sqrt2_conjugate()
            u_float = np.array([u.real(), u.imag()])
            u_conj_float = np.array([u_conj.real(), u_conj.imag()])
            if is_inside_ellipse(u_float, mod_E, p_p) and is_inside_ellipse(u_conj_float, mod_D, np.zeros(2)):
                print("Found candidate")
                print(u)
                xi = 1 - u.complex_conjugate() * u
                # t = diop.solve_xi_eq_ttdag_in_d_cpp(xi.convert(Dsqrt2))
                t = solve_xi_eq_ttdag_in_d(xi.convert(Dsqrt2))
                if t is None:
                    print("Failed")
                else:
                    solution = True
                    M = np.array([[u, -t.complex_conjugate()], [t, u.complex_conjugate()]])
                    print(M)
                    print(np.array(M, dtype=complex))
                    return M
        print(n)
        n += 1

epsilon = 1e-6
theta = 13 * math.pi / 8
U = z_rotational_approximation(epsilon, theta)
print(U)
U_complex = np.array(U, dtype=complex)
rz = np.array([
    [np.exp(-theta / 2 * 1j), 0], 
    [0, np.exp(theta / 2 * 1j)]
])
E = np.linalg.norm(rz - U_complex, 2)
print(E/epsilon)