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
This module links a C++ implementation of the
:func:`qdecomp.utils.grid_problem.z_rotational_approximation` function to Python for performance
improvements. This new implementation can be called using the :func:`rz_approx_cpp` function. It
uses the same parameters and returns the same type as the original function.
"""

import ctypes
import os
import platform

import numpy as np

from qdecomp.rings import D, Domega
from qdecomp.utils.grid_problem.rz_approx import initialization

__all__ = ["rz_approx_cpp"]


# Import the C++ library
platf = platform.system()
match (platf):
    case "Windows":
        lib_file = os.path.join(os.path.dirname(__file__), "cpp", "lib_rz_approx.dll")
    # case "Linux":
    #     lib_file = os.path.join(os.path.dirname(__file__), "cpp", "lib_rz_approx.so")
    # case "Darwin":
    #     lib_file = os.path.join(os.path.dirname(__file__), "cpp", "lib_rz_approx.dylib")
    case _:
        raise Exception(f"Unsupported platform: {platf}")

rz_approx_lib = ctypes.cdll.LoadLibrary(lib_file)


# Define the Domega data structure
class Domega_struct(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_longlong),
        ("la", ctypes.c_uint),
        ("b", ctypes.c_longlong),
        ("lb", ctypes.c_uint),
        ("c", ctypes.c_longlong),
        ("lc", ctypes.c_uint),
        ("d", ctypes.c_longlong),
        ("ld", ctypes.c_uint),
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
double = ctypes.c_double  # double

# Configure the function input and output types
rz_approx_lib.rz_approx_helper.argtypes = [
    double,  # theta
    ctypes.POINTER(double),  # ellipse
    ctypes.POINTER(double),  # point
    ctypes.POINTER(double),  # bbox1
    ctypes.POINTER(double),  # bbox2
    double,  # epsilon
]
rz_approx_lib.rz_approx_helper.restype = Rz_approx_struct  # 2 Domega_struct


# Helper to call the C++ function
def rz_approx_cpp(theta: float, epsilon: float) -> np.ndarray[Domega]:
    """
    Finds the z-rotational approximation up to an error :math:`\\varepsilon` using a C++ backend.

    This function finds an approximation of a z-rotational inside the Clifford+T group.
    A Python implementation of this function is also available via the
    :func:`qdecomp.utils.grid_problem.z_rotational_approximation` function.

    Args:
        theta (float): Angle :math:`\\theta` of the z-rotational gate
        epsilon (float): Maximum allowable error :math:`\\varepsilon`.

    Returns:
        np.ndarray[Domega]: Approximation :math:`M` of a z-rotational inside the Clifford+T subset.

    Raises:
        ValueError: If :math:`\\theta` is not in the range :math:`[0, 4\\pi]`.
        ValueError: If :math:`\\varepsilon \\geq 0.5`.
        ValueError: If :math:`\\theta` or :math:`\\varepsilon` cannot be converted to floats.
    """
    # Normalize the value of theta
    theta = theta % (4 * np.pi)

    # Verify the value of epsilon
    if epsilon >= 0.5:
        raise ValueError(f"The maximal allowable error is 0.5. Got {epsilon}.")

    # Checks if the angle is trivial
    exponent = round(2 * theta / np.pi)
    if np.isclose(0, theta):
        return np.array(
            [
                [Domega.from_ring(1), Domega.from_ring(0)],
                [Domega.from_ring(0), Domega.from_ring(1)],
            ],
            dtype=object,
        )
    elif np.isclose(2 * theta / np.pi, exponent):
        T = np.array(
            [
                [Domega(-D(1, 0), D(0, 0), D(0, 0), D(0, 0)), Domega.from_ring(0)],
                [Domega.from_ring(0), Domega(D(0, 0), D(0, 0), D(1, 0), D(0, 0))],
            ],
            dtype=object,
        )
        M = T**exponent
        return M

    # Initialize the parameters for the C++ function
    # ellipse, p_p, bbox_1, bbox_2 = initialization(theta, epsilon)
    bbox_1, bbox_2 = initialization(theta, epsilon)

    # ellipse = ellipse.astype(float)
    # p_p = p_p.astype(float)
    bbox_1 = bbox_1.astype(float)
    bbox_2 = bbox_2.astype(float)

    # Convert the parameters to the appropriate types
    c_epsilon = double(epsilon)
    c_ellipse = (double * 4)(1.0, 0.0, 0.0, 1.0)  # Identity ellipse
    c_point = (double * 2)(0.0, 0.0)  # Origin point
    c_bbox1 = (double * 4)(*bbox_1.flatten())
    c_bbox2 = (double * 4)(*bbox_2.flatten())
    c_theta = double(theta)

    # Compute the Rz approximation using the C++ library
    res = rz_approx_lib.rz_approx_helper(c_theta, c_ellipse, c_point, c_bbox1, c_bbox2, c_epsilon)

    # Return the result as a numpy array
    u = Domega_struct_to_Domega(res.u)
    t = Domega_struct_to_Domega(res.t)

    return np.array([[u, -t.complex_conjugate()], [t, u.complex_conjugate()]])
