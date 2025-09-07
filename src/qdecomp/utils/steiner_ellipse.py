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
This module allows to find the smallest ellipse englobing three points using the Steiner algorithm :cite:`steiner_steiner_ellipse`.
The module also contains useful functions allowing to find the bounding box (BBOX) of an ellipse and
determine whether points are inside an ellipse using its matrix definition.
"""

from typing import Union

import mpmath as mp
import numpy as np

__all__ = [
    "assert_steiner_ellipse",
    "steiner_ellipse_def",
    "is_inside_ellipse",
    "ellipse_bbox",
]


def assert_steiner_ellipse(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> None:
    """
    Check if the three given points can be used to define a Steiner ellipse. The three points must
    be distinct and non-collinear to define a valid ellipse. If the points are not valid, a
    ``ValueError`` is raised.

    Args:
        p1 (list[float]): First point
        p2 (list[float]): Second point
        p3 (list[float]): Third point

    Raises:
        ValueError: If at least two of the points are the same
        ValueError: If the three points are collinear
    """
    # Ensure all three points are distinct
    if (p1 == p2).all() or (p1 == p3).all() or (p2 == p3).all():
        raise ValueError("The three points must be distinct.")

    # Ensure the points are not collinear
    delta1 = p2 - p1
    delta2 = p3 - p2

    if (delta1[0] != 0) and (delta2[0] != 0):
        # Avoid division by 0 for slope calculations
        slope1 = delta1[1] / delta1[0]
        slope2 = delta2[1] / delta2[0]
        if slope1 == slope2:
            raise ValueError("The three points must not be collinear.")

    else:
        # Handle vertical lines to ensure they are not collinear
        if (delta1[0] == 0) and (delta2[0] == 0):
            raise ValueError("The three points must not be collinear.")


def steiner_ellipse_def(
    p1: list[float], p2: list[float], p3: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the smallest ellipse that passes through the three given points using the Steiner
    method. The ellipse is represented by the equation :math:`(u − p)^\dagger D (u − p) \leq 1`, where :math:`p` is the
    center of the ellipse and :math:`D` is a matrix that defines its shape and orientation. :math:`p1`, :math:`p2`, :math:`p3` can
    be any iterable containing real numbers.

    Args:
        p1 (list[float]): First point
        p2 (list[float]): Second point
        p3 (list[float]): Third point

    Returns:
        tuple[np.ndarray, np.ndarray]: :math:`(D, p)`: the matrix defining the shape and orientation of the ellipse, and the center of the ellipse
    """
    # Determine whether to use high precision or not
    high_precision = isinstance(p1[0], mp.mpf)

    # Convert the points to numpy arrays
    p1_, p2_, p3_ = np.array(p1), np.array(p2), np.array(p3)

    # Check that the ellipse can be defined by the three points
    assert_steiner_ellipse(p1_, p2_, p3_)

    # Calculate the center of the ellipse
    p = (p1_ + p2_ + p3_) / 3  # Center of the ellipse

    # Compute useful vectors for the Steiner method
    f1 = p1_ - p
    f2 = (p3_ - p2_) / np.sqrt(3)

    # Define a parametric function for tracing the contour of the ellipse
    if high_precision:
        contour = lambda t: p + f1 * mp.cos(t) + f2 * mp.sin(t)
    else:
        contour = lambda t: p + f1 * np.cos(t) + f2 * np.sin(t)

    # Calculate t0 according to the Steiner method
    if high_precision:
        t0 = mp.atan(2 * f1 @ f2 / (f1 @ f1 - f2 @ f2)) / 2
    else:
        t0 = np.arctan(2 * f1 @ f2 / (f1 @ f1 - f2 @ f2)) / 2

    # Compute the two main axes of the ellipse
    axis1 = contour(t0) - contour(t0 + np.pi)
    axis2 = contour(t0 + np.pi / 2) - contour(t0 - np.pi / 2)

    # Temporary D matrix (defines the size of the axes)
    # An axis-aligned ellipse is defined by a diagonal matrix whose diagonal values are the inverse
    # of the squares of the half-lengths of the axes.
    D_ = np.diag([(2 / np.linalg.norm(axis1)) ** 2, (2 / np.linalg.norm(axis2)) ** 2])

    # Calculate the rotation matrix based on the orientation of the ellipse
    if high_precision:
        theta = mp.atan2(*axis2)
        rotation_matrix = np.array(
            [
                [mp.cos(theta), -mp.sin(theta)],
                [mp.sin(theta), mp.cos(theta)],
            ]
        )
    else:
        theta = np.arctan2(*axis2)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

    # Calculate the final D matrix that defines the oriented ellipse
    D = rotation_matrix.T @ D_ @ rotation_matrix

    # Return the D matrix and the center p
    return D, p


NestedList = Union[list[float], list["NestedList"]]


def is_inside_ellipse(u: NestedList, D: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Check if a point u (or an array of points) is inside the ellipse defined by matrix D and center
    p. The function works for both single points and arrays of points, where the last dimension of u
    must be the same as the number of dimensions of the ellipse.

    Args:
        u (NestedList): The point(s) to be tested, an array of shape (..., n_dim)
        D (np.ndarray): Matrix defining the ellipse's shape and orientation
        p (np.ndarray): Center of the ellipse

    Returns:
        np.ndarray: A boolean array indicating if each point is inside the ellipse

    Raises:
        IndexError: If the dimensions of the D and p arguments are incompatible
        IndexError: If the last dimension of the points to test is different from the number of
            dimensions of the ellipse
    """
    u_ = np.array(u)

    # Test that the dimensions of the arguments are compatible
    n_dim = len(p)

    if D.shape != (n_dim, n_dim):
        raise IndexError(
            f"The matrix definition (shape {D.shape}) and center (shape {p.shape}) must have "
            + "compatible dimensions."
        )

    if u_.shape[-1] != n_dim:
        raise IndexError(
            f"The last dimension of the points to test (shape {u_.shape[-1]}) must be the same "
            + "than the number of dimensions of the ellipse (shape {len(p)})."
        )

    # Determine which points are inside the ellipse
    vector = u_ - p
    is_inside = np.einsum("...i,ij,...j->...", vector, D, vector) <= 1

    return is_inside


def ellipse_bbox(D: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Find the axis-aligned bounding box (BBOX) of an ellipse. Refer to the comment made by Rodrigo de
    Azevedo on November 30th, 2020 in :cite:`steiner_ellipse_bbox`.

    Args:
        D (np.ndarray): Matrix defining the ellipse's shape and orientation
        p (np.ndarray): Center of the ellipse

    Returns:
        np.ndarray: A numpy array of shape (n_dim, 2) representing the bounding box. The first index
        corresponds to each spatial dimension (e.g., x, y, ...), and the second index contains the
        minimum and maximum bounds along that dimension.
    """
    # Determine whether to use high precision or not
    high_precision = isinstance(p[0], mp.mpf)

    if high_precision:
        D_inv = mp.matrix(D) ** -1
        D_inv = np.array(D_inv.tolist())
    else:
        D_inv = np.linalg.inv(D)

    diag = np.diagonal(D_inv)  # Vector with the diagonal values of D_inv

    n_dim = len(p)  # Number of dimensions
    bbox = np.outer(np.sqrt(diag), np.array([-1, 1])) + np.outer(p, np.ones(n_dim))  # BBOX

    return bbox
