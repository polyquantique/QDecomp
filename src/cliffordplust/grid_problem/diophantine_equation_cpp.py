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

"""
This module solves the Diophantine equation \u03BE = t\u2020 t for t where \u03BE is given using
C++. Refer to the diophantine_equation.py file for a Python implementation.

Input: \u03Be in D[\u221A2]
Output: t in D[\u03C9] such that \u03Be = t\u2020 t
"""

import os
import platform
import ctypes
from Rings import *


# Import the C++ library
platf = platform.system()
match (platf):
    case "Windows":
        lib_file = os.path.join(os.path.dirname(__file__), "cpp", "diophantine_lib.dll")
    # case "Linux":
    #     lib_file = os.path.join(os.path.dirname(__file__), "cpp", "libdiophantine_equation_lib.so")
    # case "Darwin":
    #     lib_file = os.path.join(os.path.dirname(__file__), "cpp", "libdiophantine_equation_lib.dylib")
    case _:
        raise Exception(f"Unsupported platform: {platf}")

dioph_lib = ctypes.cdll.LoadLibrary(lib_file)


# Define the output data structure
class Domega_struct(ctypes.Structure):
    _fields_ = [
        ("has_solution", ctypes.c_bool),
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
dioph_lib.solve_xi_eq_ttdag_in_d_helper.restype = Domega_struct  # 1 bool and 8 integer outputs


# Helper to call the C++ function
def solve_xi_eq_ttdag_in_d_cpp(xi: Dsqrt2) -> Domega | None:
    """
    Solve the equation \u03BE = t * t\u2020 or t where \u2020 denotes the complex conjugate. \u03BE
    is an element of D[\u221A2] and t is an element of D[\u03C9]. This function returns the first
    solution of the equation. If no solution exists, the function returns None.

    Note: this function is equivalent to the solve_xi_eq_ttdag_in_d() function in the
    diophantine_equation.py file, but it is implemented in C++. It is roughly 150 times faster.

    Args:
        xi (Dsqrt2): A number

    Returns:
        Domega of None: A number t for which \u03BE = t * t\u2020, or None if no solution exists
    """
    # Solve the diophantine equation using the C++ library
    res = dioph_lib.solve_xi_eq_ttdag_in_d_helper(xi.p.num, xi.p.denom, xi.q.num, xi.q.denom)

    # If there is no solution, return None
    if not res.has_solution:
        return None
    
    # Otherwise, return the solution
    return Domega(
        (res.a, res.la),
        (res.b, res.lb),
        (res.c, res.lc),
        (res.d, res.ld),
    )

# print(solve_xi_eq_ttdag_in_d_cpp(Dsqrt2((13, 1), (2, 1))))
# print(solve_xi_eq_ttdag_in_d_cpp(Dsqrt2((13, 2), (2, 1))))
