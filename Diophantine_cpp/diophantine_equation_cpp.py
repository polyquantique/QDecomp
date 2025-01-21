import sys

sys.path.append("../../CliffordPlusT")

import ctypes
from grid_algorithm_1D.Rings import *


# Import the C++ library
path = ".\\diophantine_equation_lib.dll"
dioph_lib = ctypes.cdll.LoadLibrary(path)

# Define the output data structure
class Domega_struct(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_int),
        ("la", ctypes.c_int),
        ("b", ctypes.c_int),
        ("lb", ctypes.c_int),
        ("c", ctypes.c_int),
        ("lc", ctypes.c_int),
        ("d", ctypes.c_int),
        ("ld", ctypes.c_int),
    ]

# Configure the function input and output types
dioph_lib.solve_xi_eq_ttdag_in_d_helper.argtypes = [ctypes.c_int] * 4  # 4 integer inputs
dioph_lib.solve_xi_eq_ttdag_in_d_helper.restype = Domega_struct  # 8 integer outputs

# Wrapper to call the C++ function
def solve_xi_eq_ttdag_in_d_cpp(xi):
    res = dioph_lib.solve_xi_eq_ttdag_in_d_helper(xi.p.num, xi.p.denom, xi.q.num, xi.q.denom)
    return Domega(
        (res.a, res.la),
        (res.b, res.lb),
        (res.c, res.lc),
        (res.d, res.ld),
    )
