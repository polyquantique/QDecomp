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
This module provides functionality for the exact synthesis of 2x2 unitary matrices with entries in
:math:`\mathbb{D}[\omega]` into sequences of H and T gates, up to a global phase. It also includes
functions to characterize and generate sequences of H and T gates using the :class:`~qdecomp.rings.rings.Domega` class.

The algorithm is based on :cite:`exact_synth_exact_synthesis`.

This module contains the following functions:

- :func:`exact_synthesis_alg`: Decomposes a unitary 2x2 matrix with elements in :math:`\mathbb{D}[\omega]` into a sequence of H and T gates.
- :func:`exact_synthesis_reduc`: Computes the sde reduction of a matrix with elements in :math:`\mathbb{D}[\omega]` into a sequence of H and T gates and a matrix with sde :math:`\leq 3`.
- :func:`s3_decomposition`: Finds the sequence of H and T gates up to a global phase to synthesize a matrix with elements in :math:`\mathbb{D}[\omega]` by looking in the S3 table.
- :func:`is_unitary_deomega`: Checks if a matrix with :math:`\mathbb{D}[\omega]` elements is unitary.
- :func:`apply_sequence`: Applies a sequence of W, H and T gates to a matrix.
- :func:`domega_matrix_to_tuple`: Converts the first column of a 2x2 array of :math:`\mathbb{D}[\omega]` elements to a tuple format.
- :func:`get_omega_exponent`: Evaluates the phase difference between two complex numbers in terms of powers of :math:`\omega`

**Example:**

.. code-block:: python

    >>> import numpy as np
    >>> from qdecomp.utils.exact_synthesis import exact_synthesis_alg, ZERO_DOMEGA, ONE_DOMEGA
    >>> from qdecomp.rings import Domega

    # Define the X gate with 0 and 1 as Domega objects
    >>> X = np.array([[ZERO_DOMEGA, ONE_DOMEGA], [ONE_DOMEGA, ZERO_DOMEGA]], dtype=Domega)

    # Perform the exact synthesis
    >>> sequence = exact_synthesis_alg(X)
    >>> print(sequence)
    HTTTTH
"""

import json
import os

import numpy as np

from qdecomp.rings import Domega

__all__ = ["exact_synthesis_alg", "optimize_sequence"]

H_11: Domega = Domega((-1, 1), (0, 0), (1, 1), (0, 0))
r""" Domega: First element of H gate (:math:`\frac{1}{\sqrt{2}}`). """

ONE_DOMEGA: Domega = Domega((0, 0), (0, 0), (0, 0), (1, 0))
""" Domega: Domega representation of 1. """

ZERO_DOMEGA: Domega = Domega((0, 0), (0, 0), (0, 0), (0, 0))
""" Domega: Domega representation of 0. """

OMEGA: Domega = Domega((0, 0), (0, 0), (1, 0), (0, 0))
r""" Domega: Domega representation of :math:`\omega = \frac{\sqrt{2}}{2} + i \frac{\sqrt{2}}{2}`. """

T_22_INV: Domega = Domega((-1, 0), (0, 0), (0, 0), (0, 0))
r""" Domega: Last element of T :math:`^\dagger` gate (:math:`-\omega^3`). """

H: np.ndarray = np.array([[H_11, H_11], [H_11, -H_11]], dtype=Domega)
""" np.ndarray[Domega]: Domega representation of H gate. """

T: np.ndarray = np.array([[ONE_DOMEGA, ZERO_DOMEGA], [ZERO_DOMEGA, OMEGA]], dtype=Domega)
""" np.ndarray[Domega]: Domega representation of T gate. """

T_INV: np.ndarray = np.array([[ONE_DOMEGA, ZERO_DOMEGA], [ZERO_DOMEGA, T_22_INV]], dtype=Domega)
r""" np.ndarray[Domega]: Domega representation of T :math:`^\dagger` gate. """

I: np.ndarray = np.array([[ONE_DOMEGA, ZERO_DOMEGA], [ZERO_DOMEGA, ONE_DOMEGA]], dtype=Domega)
""" np.ndarray[Domega]: Domega representation of identity matrix. """

W: np.ndarray = np.array([OMEGA]) * I
""" np.ndarray[Domega]: Domega representation of W gate. """


def exact_synthesis_alg(U: np.ndarray, insert_global_phase: bool = False) -> str:
    """
    Decompose an unitary 2x2 matrix with elements in :math:`\\mathbb{D}[\\omega]` into a sequence of H and T
    gates.

    The algorithm is based on :cite:`exact_synth_exact_synthesis`.

    Args:
        U (np.ndarray): Unitary 2x2 matrix to decompose, with elements in Domega
        insert_global_phase (bool): If True, insert the global phase gates (W) in the final sequence. Default is `False`

    Returns:
        str: Sequence of H and T (W if `insert_global_phase` is `True`) gates to decompose the matrix

    Raises:
        TypeError: If matrix elements are not instances of the class `Domega`
        TypeError: If the matrix is not 2x2
        ValueError: If the matrix is not unitary
    """
    if not np.all([isinstance(element, Domega) for element in U.flatten()]):
        raise TypeError(f"The matrix elements must be of class Domega. Got {type(U[0][0])}.")

    elif U.shape != (2, 2):
        raise TypeError(f"The matrix must be of shape 2x2. Got shape {U.shape}.")

    elif not is_unitary_domega(U):
        raise ValueError("The matrix must be unitary. Got U=\n", U)

    u3_sequence, u3 = exact_synthesis_reduc(U)
    s3_sequence = s3_decomposition(u3, insert_global_phase=insert_global_phase)
    final_sequence = u3_sequence + s3_sequence

    return final_sequence


def exact_synthesis_reduc(U: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Compute the exact synthesis reduction of an unitary 2x2 matrix with elements in :math:`\\mathbb{D}[\\omega]` into the
    sequence that reduces the unitary and a matrix with a sde :math:`\\leq 3`.
    This part of the algorithm is based on Algorithm 1 from :cite:`exact_synth_exact_synthesis`.

    Args:
        U (np.ndarray): Unitary 2x2 matrix with elements in Domega that needs to be reduced

    Returns:
        tuple[str, np.ndarray]: Tuple containing the sequence of H and T gates to reduce the matrix and the remaining matrix with sde :math:`\\leq 3`

    Raises:
        TypeError: If the matrix elements are not instances of the class Domega
        TypeError: If the matrix is not 2x2
        ValueError: If the matrix is not unitary
    """

    if not np.all([isinstance(element, Domega) for element in U.flatten()]):
        raise TypeError(f"The matrix elements must be of class Domega. Got {type(U[0][0])}.")

    elif U.shape != (2, 2):
        raise TypeError(f"The matrix must be of shape 2x2. Got shape {U.shape}.")

    elif not is_unitary_domega(U):
        raise ValueError("The matrix must be unitary. Got U=\n", U)

    sequence: str = ""
    norm_z = U[0, 0] * U[0, 0].complex_conjugate()
    s = norm_z.sde()

    # Reduce sde until sde <= 3
    while s > 3:
        # Look for the k that reduces the sde by 1
        for k in [0, 1, 2, 3]:
            U_prime = H @ np.linalg.matrix_power(T_INV, k) @ U
            norm_z_prime = U_prime[0, 0] * U_prime[0, 0].complex_conjugate()

            # Add found k to sequence and update U and s
            if norm_z_prime.sde() == s - 1:
                sequence += k * "T" + "H"
                s = norm_z_prime.sde()
                U = U_prime
                break

    return sequence, U


def s3_decomposition(U: np.ndarray, insert_global_phase: bool = False) -> str:
    """
    Find the sequence of W, H and T gates to synthesize a matrix with elements in :math:`\\mathbb{D}[\\omega]` by
    looking in the S3 table.

    Args:
        U (np.ndarray): Matrix to synthesize
        print_global_phase (bool): If True, prints the global phase gates in the final sequence. Default is `False`

    Returns:
        str: Sequence of W, H and T gates to synthesize the matrix

    Raises:
        TypeError: If the matrix elements are not instances of the class Domega
        TypeError: If the matrix is not 2x2
        ValueError: If the matrix is not unitary
        ValueError: If the matrix has sde > 3
    """
    if not np.all([isinstance(element, Domega) for element in U.flatten()]):
        raise TypeError("The matrix elements must be of class Domega")

    elif U.shape != (2, 2):
        raise TypeError("The matrix must be of shape 2x2")

    elif not is_unitary_domega(U):
        raise ValueError("The matrix must be unitary")

    elif (U[0, 0] * U[0, 0].complex_conjugate()).sde() > 3:
        raise ValueError(
            "The matrix must have a sde < 4, got sde = ",
            (U[0, 0] * U[0, 0].complex_conjugate()).sde(),
        )

    # Load the s3_table into a dictionary if not already loaded
    if not hasattr(s3_decomposition, "s3_cache"):
        with open(os.path.join(os.path.dirname(__file__), "s3_table.json"), "r") as f:
            s3_dict = json.load(f)
            s3_dict = {
                k: tuple(tuple(tuple(inner) for inner in outer) for outer in v)
                for k, v in s3_dict.items()
            }
            s3_decomposition.s3_cache = s3_dict

    else:
        s3_dict = s3_decomposition.s3_cache

    # Look for first column of powers of omega times U in s3_table
    for i in range(8):
        U_t = np.array([OMEGA**i]) * U
        for key, value in s3_dict.items():
            # If found, build the last column by multiplying by T^k'
            if domega_matrix_to_tuple(U_t) == value:
                # Compute phase differences to find k'
                U_w = apply_sequence(key + "W" * (8 - i))

                # If first element is zero, take second element instead
                if U_w[0, 0] == ZERO_DOMEGA:
                    k = get_omega_exponent(U[0, 1], U[1, 0].complex_conjugate())
                    k_pp = get_omega_exponent(U_w[0, 1], U_w[1, 0].complex_conjugate())

                # Else, take first element
                else:
                    k = get_omega_exponent(U[1, 1], U[0, 0].complex_conjugate())
                    k_pp = get_omega_exponent(U_w[1, 1], U_w[0, 0].complex_conjugate())

                # Compute k': real T exponent
                k_prime = (k - k_pp) % 8
                key += "T" * k_prime
                if insert_global_phase:
                    key += "W" * ((8 - i) % 8)
                return key


def is_unitary_domega(matrix: np.ndarray) -> bool:
    """
    Check if a matrix with :math:`\\mathbb{D}[\\omega]` elements is unitary

    Args:
        matrix (np.ndarray): Matrix to check

    Returns:
        bool: True if matrix is unitary, False otherwise
    """
    conj_transpose = np.array(
        [[element.complex_conjugate() for element in row] for row in matrix.T]
    )
    product = np.dot(matrix, conj_transpose)
    return (product == I).all()


def apply_sequence(sequence: str, U: np.ndarray = I) -> np.ndarray:
    """Apply a sequence of W, H and T gates to a matrix. If the `matrix` is not provided, the identity matrix is used. Only W, H and T gates are supported.

        sequence (str): Sequence of W, H and T gates
        U (np.ndarray): Matrix to apply the sequence to (default is identity matrix)

    Returns:
        np.ndarray: Matrix after applying the sequence of gates

    Raises:
        ValueError: If a character in the sequence is not W, H or T

    """
    for char in sequence[::-1]:
        if char == "H":
            U = H @ U

        elif char == "T":
            U = T @ U

        elif char == "W":
            U = np.array([OMEGA]) * U

        else:
            raise ValueError("Invalid character in sequence")

    return U


def domega_matrix_to_tuple(array: np.ndarray) -> tuple:
    """
    Convert the first column of a 2x2 array of :math:`\\mathbb{D}[\\omega]` elements to a tuple of tuples of (num, denom)
    representing parameters needed to initialize a `Domega` object.

    Args:
        array (np.ndarray): 2x2 array of :math:`\\mathbb{D}[\\omega]` elements

    Returns:
        tuple: First column in tuple of tuples of (num, denom) where num and denom are entries in :math:`\\mathbb{D}`

    Raises:
        TypeError: If the matrix elements are not instances of the class :math:`\\mathbb{D}[\\omega]`
        TypeError: If the matrix is not 2x2

    **Example:**

    .. code-block:: python

        >>> import numpy as np
        >>> from qdecomp.utils.exact_synthesis import domega_matrix_to_tuple, ZERO_DOMEGA, ONE_DOMEGA
        >>> from qdecomp.rings import Domega

        # Define the X gate with 0 and 1 as Domega objects
        >>> X = np.array([[ZERO_DOMEGA, ONE_DOMEGA], [ONE_DOMEGA, ZERO_DOMEGA]], dtype=Domega)

        # Convert X to tuple format
        >>> x_tuple = domega_matrix_to_tuple(X)
        >>> print(x_tuple) # First column of X in tuple format
        ((0, 0), (0, 0), (0, 0), (0, 0)), ((0, 0), (0, 0), (0, 0), (1, 0))
    """
    if not np.all([isinstance(element, Domega) for element in array.flatten()]):
        raise TypeError("Matrix elements must be of class Domega")

    elif array.shape != (2, 2):
        raise TypeError("Matrix must be of shape 2x2")

    return tuple(
        tuple((Domega[i].num, Domega[i].denom) for i in range(4)) for Domega in array[:, 0]
    )


def get_omega_exponent(z_1: Domega, z_2: Domega) -> int:
    """Evaluate the phase difference between two complex numbers z_2 and z_1 in terms of powers of :math:`\\omega`

    Args:
        z_1 (Domega): First :math:`\\mathbb{D}[\\omega]` element
        z_2 (Domega): Second :math:`\\mathbb{D}[\\omega]` element

    Returns:
        int: Number of powers of :math:`\\omega` needed to transform z_2 into z_1
    """
    z_1_angle = np.arctan2(z_1.imag(), z_1.real())
    z_2_angle = np.arctan2(z_2.imag(), z_2.real())

    angle = z_1_angle - z_2_angle
    omega_exponent = int(np.round(angle / (np.pi / 4))) % 8

    return omega_exponent


def optimize_sequence(sequence: str) -> str:
    """
    Performs a basic optimization of a sequence of gates by removing redundant gates and combining consecutive gates.

    Args:
        sequence (str): The input sequence of gates as a string.

    Returns:
        str: The optimized sequence of gates.

    Raises:
        TypeError: If the input sequence is not a string.
    """

    if not isinstance(sequence, str):
        raise TypeError(f"Input sequence must be a string. Got {type(sequence)}.")

    optimized_sequence = sequence  # Copy the sequence
    last_length = -1
    while len(optimized_sequence) != last_length:
        last_length = len(optimized_sequence)
        optimized_sequence = optimized_sequence.replace("TT", "S")
        optimized_sequence = optimized_sequence.replace("SS", "Z")
        optimized_sequence = optimized_sequence.replace("ZZ", "")
        optimized_sequence = optimized_sequence.replace("HH", "")
        optimized_sequence = optimized_sequence.replace("ZTZ", "T")
        optimized_sequence = optimized_sequence.replace("ZSZ", "S")
        optimized_sequence = optimized_sequence.replace("STS", "ZT")
        optimized_sequence = optimized_sequence.replace("TST", "Z")

    return optimized_sequence
