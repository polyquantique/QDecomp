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
This module contains utility functions to check if a matrix is special, orthogonal, unitary or Hermitian.

These functions are used to check the Lie group of a matrix and to select the appropriate decomposition to perform.

The module contains the following functions:

    :func:`is_special`: Check if a matrix is special.

    :func:`is_orthogonal`: Check if a matrix is orthogonal.

    :func:`is_unitary`: Check if a matrix is unitary.

    :func:`is_hermitian`: Check if a matrix is Hermitian.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["is_special", "is_orthogonal", "is_unitary", "is_hermitian"]


def is_special(matrix: NDArray[np.floating]) -> bool:
    """
    Check if a matrix is special.

    Return True if the determinant of the matrix is 1 and False otherwise.

    Args:
        matrix (NDArray[float]): Matrix to check.

    Returns:
        bool: True if the matrix is special, False otherwise.
    """
    return np.isclose(np.linalg.det(matrix), 1)


def is_orthogonal(matrix: NDArray[np.floating]) -> bool:
    """
    Check if a matrix is orthogonal.

    Return True if the matrix inverse is equal to its transpose and False otherwise.

    Args:
        matrix (NDArray[float]): Matrix to check.

    Returns:
        bool: True if the matrix is orthogonal, False otherwise.
    """
    return (
        np.allclose(matrix @ matrix.T, np.identity(matrix.shape[0]))
        and matrix.shape[0] == matrix.shape[1]
    )


def is_unitary(matrix: NDArray[np.floating]) -> bool:
    """
    Check if a matrix is unitary.

    Return True if the matrix inverse is equal to its conjugate transpose and False otherwise.

    Args:
        matrix (NDArray[float]): Matrix to check.

    Returns:
        bool: True if the matrix is unitary, False otherwise.
    """
    return (
        np.allclose(matrix @ matrix.T.conj(), np.identity(matrix.shape[0]))
        and matrix.shape[0] == matrix.shape[1]
    )


def is_hermitian(matrix: NDArray[np.floating]) -> bool:
    """
    Check if a matrix is Hermitian.

    Return True if the matrix is equal to its conjugate transpose and False otherwise.

    Args:
        matrix (NDArray[float]): Matrix to check.

    Returns:
        bool: True if the matrix is Hermitian, False otherwise.
    """
    return np.allclose(matrix, matrix.T.conj())
