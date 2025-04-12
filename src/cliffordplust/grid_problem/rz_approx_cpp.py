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
Missing docsstring.
"""

import os
import sys

path_list = os.path.dirname(__file__).split(os.sep)
sys.path.append(os.sep.join(path_list[:-3]))

import ctypes
import platform
import numpy as np

from cliffordplust.rings.rings import *
from cliffordplust.grid_problem.rz_approx import initialization

# Import the C++ library
platf = platform.system()
match (platf):
    case "Windows":
        lib_file = os.path.join(os.path.dirname(__file__), "cpp", "rz_approx_lib.dll")
    # case "Linux":
    #     lib_file = os.path.join(os.path.dirname(__file__), "cpp", "librz_approx_lib.so")
    # case "Darwin":
    #     lib_file = os.path.join(os.path.dirname(__file__), "cpp", "librz_approx_lib.dylib")
    case _:
        raise Exception(f"Unsupported platform: {platf}")

rz_approx_lib = ctypes.cdll.LoadLibrary(lib_file)


# Define the Domega data structure
class Domega_struct(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_longlong),
        ("la", ctypes.c_ushort),
        ("b", ctypes.c_longlong),
        ("lb", ctypes.c_ushort),
        ("c", ctypes.c_longlong),
        ("lc", ctypes.c_ushort),
        ("d", ctypes.c_longlong),
        ("ld", ctypes.c_ushort),
    ]

def Domega_struct_to_Domega(domega_struct: Domega_struct) -> Domega:
    """
    Convert a Domega_struct to a Domega object.

    Args:
        domega_struct (Domega_struct): The Domega_struct object to convert.

    Returns:
        Domega: The converted Domega object.
    """
    return Domega(
        (domega_struct.a, domega_struct.la),
        (domega_struct.b, domega_struct.lb),
        (domega_struct.c, domega_struct.lc),
        (domega_struct.d, domega_struct.ld),
    )

# Define the result type for the C++ function
class Rz_approx_struct(ctypes.Structure):
    _fields_ = [("u", Domega_struct), ("t", Domega_struct)]

# C++ data types aliases
db = ctypes.c_double  # long double
db_2 = db * 2  # long double [2]
db_2_2 = db * 2 * 2  # long double [2][2]

# Configure the function input and output types
rz_approx_lib.rz_approx_helper.argtypes = [
    db,  # theta
    db_2_2,  # ellipse
    db_2,  # point
    db_2_2,  # bbox1
    db_2_2,  # bbox2
    db  # epsilon
]
rz_approx_lib.rz_approx_helper.restype = Rz_approx_struct  # 2 Domega_struct


# Helper to call the C++ function
def rz_approx_cpp(epsilon: float, theta: float) -> np.ndarray:
    """
    Missing docstring.
    """
    # Initialize the parameters for the C++ function
    E, p_p, bbox_1, bbox_2 = initialization(epsilon, theta)

    E = E.astype(float)
    p_p = p_p.astype(float)
    bbox_1 = bbox_1.astype(float)
    bbox_2 = bbox_2.astype(float)

    # Convert the parameters to the appropriate types
    ellipse = db_2_2(db_2(*E[0]), db_2(*E[1]))
    point = db_2(*p_p)
    bbox1 = db_2_2(db_2(*bbox_1[0]), db_2(*bbox_1[1]))
    bbox2 = db_2_2(db_2(*bbox_2[0]), db_2(*bbox_2[1]))

    # Compute the Rz approximation using the C++ library
    res = rz_approx_lib.rz_approx_helper(theta, ellipse, point, bbox1, bbox2, epsilon)
    
    # Return the result as a numpy array
    u = Domega_struct_to_Domega(res.u)
    t = Domega_struct_to_Domega(res.t)

    return np.array([[u, -t.complex_conjugate()], [t, u.complex_conjugate()]])
