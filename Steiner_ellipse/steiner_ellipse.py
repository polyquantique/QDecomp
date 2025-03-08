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
This module allow to find the smallest ellipse englobing three points using the Steiner algorithm.
The module also contains useful functions allowing to plot an ellipse, find its bounding box (BBOX)
and determine wether points are inside an ellipse using its matrix definition.

See this page for more information:
https://en.wikipedia.org/wiki/Steiner_ellipse
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def steiner_ellipse_def(p1, p2, p3):
    """
    Calculates the smallest ellipse that passes through the three given points using the Steiner
    method. The ellipse is represented by the equation (u − p)† D (u − p) <= 1, where p is the
    center of the ellipse and D is a matrix that defines its shape and orientation. p1, p2, p3 can
    be any iterable containing real numbers.

    :param p1: First point (2D coordinates)
    :param p2: Second point (2D coordinates)
    :param p3: Third point (2D coordinates)
    :return: D matrix (defines the shape and orientation of the ellipse), p (center of the ellipse)
    """
    # Convert the points to numpy arrays
    p1_, p2_, p3_ = np.array(p1), np.array(p2), np.array(p3)

    # Check that the ellipse can be defined by the three points
    # ---------------------------------------------------------

    # Ensure all three points are distinct
    if (p1_ == p2_).all() or (p1_ == p3_).all() or (p2_ == p3_).all():
        raise ValueError("The three points must be distinct.")

    # Ensure the points are not collinear
    delta1 = p2_ - p1_
    delta2 = p3_ - p2_

    collin_error = False  # Flag to raise an error
    if (delta1[0] != 0) and (delta2[0] != 0):
        # Avoid division by 0 for slope calculations
        slope1 = delta1[1] / delta1[0]
        slope2 = delta2[1] / delta2[0]
        if slope1 == slope2:
            collin_error = True

    else:
        # Handle vertical lines to ensure they are not collinear
        if (delta1[0] == 0) and (delta2[0] == 0):
            collin_error = True

    if collin_error:
        raise ValueError("The three points must not be collinear.")

    # -------------------------------------------------------------

    # Calculate the center of the ellipse
    p = (p1_ + p2_ + p3_) / 3  # Center of the ellipse

    # Compute useful vectors for the Steiner method
    f1 = p1_ - p
    f2 = (p3_ - p2_) / np.sqrt(3)

    # Define a parametric function for tracing the contour of the ellipse
    contour = lambda t: p + f1 * np.cos(t) + f2 * np.sin(t)

    # Calculate t0 according to the Steiner method
    t0 = np.arctan(2 * f1 @ f2 / (f1 @ f1 - f2 @ f2)) / 2

    # Compute the two main axes of the ellipse
    axis1 = contour(t0) - contour(t0 + np.pi)
    axis2 = contour(t0 + np.pi / 2) - contour(t0 - np.pi / 2)

    # Temporary D matrix (defines the size of the axes)
    # An axis-aligned ellipse is defined by a diagonal matrix whose diagonal values are the inverse
    # of the squares of the half-lengths of the axes.
    D_ = np.diag([(2 / np.linalg.norm(axis1)) ** 2, (2 / np.linalg.norm(axis2)) ** 2])

    # Calculate the rotation matrix based on the orientation of the ellipse
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


def is_inside_ellipse(u, D, p):
    """
    Check if a point u (or an array of points) is inside the ellipse defined by matrix D and center
    p. The function works for both single points and arrays of points, where the last dimension of u
    must be the same as the number of dimensions of the ellipse.

    :param u: The point(s) to be tested, an array of shape (..., n_dim)
    :param D: Matrix defining the ellipse's shape and orientation
    :param p: Center of the ellipse
    :return: A boolean array indicating if each point is inside the ellipse
    """
    u_ = np.array(u)

    # Test that the dimensions of the arguments are compatible
    n_dim = len(p)
    if D.shape != (n_dim, n_dim):
        raise IndexError(
            f"The matrix definition (shape {D.shape}) and center (shape {p.shape}) \
must have compatible dimensions."
        )

    if u_.shape[-1] != n_dim:
        raise IndexError(
            f"The last dimension of the points to test (shape {u_.shape[-1]}) must be \
the same than the number of dimensions of the ellipse (shape {len(p)})."
        )

    # Determine which points are inside the ellipse
    vector = u_ - p
    is_inside = np.einsum("...i,ij,...j->...", vector, D, vector) <= 1

    return is_inside


def ellipse_bbox(D, p):
    """
    Find the axis-aligned bounding box (BBOX) of an ellipse.

    See this link for the algorithm (comment from Rodrigo de Azevedo on November 30th, 2020):
    https://math.stackexchange.com/questions/3926884/smallest-axis-aligned-bounding-box-of-hyper-ellipsoid

    :param D: Matrix defining the ellipse's shape and orientation
    :param p: Center of the ellipse

    :return: A numpy array of shape (n_dim, 2) representing the bounding box. The first index
    corresponds to each spatial dimension (e.g., x, y, ...), and the second index contains the
    minimum and maximum bounds along that dimension.
    """
    D_inv = np.linalg.inv(D)  # Inverse of D
    diag = np.diagonal(D_inv)  # Vector with the diagonal values of D_inv

    n_dim = len(p)  # Number of dimensions
    bbox = np.outer(np.sqrt(diag), np.array([-1, 1])) + np.outer(p, np.ones(n_dim))  # BBOX

    return bbox


def plot_ellipse(D, p, points=None):
    """
    Plot the ellipse defined by matrix D and center p. The function also plots the BBOX of the
    ellipse and its center. Moreover, the (optional) points to plot are cyan if they lie inside the
    ellipse of magenta if they are not.

    :param D: Matrix defining the ellipse's shape and orientation
    :param p: Center of the ellipse
    :param points: Points to plot
    """
    # Find the BBOX of the ellipse
    bbox = ellipse_bbox(D, p)  # BBOX of the ellipse
    edges = bbox[:, 1] - bbox[:, 0]  # Length of the edges of the BBOX

    # Generate a grid for visualizing the ellipse's interior region
    density = 200
    x = np.linspace(*(bbox[0] + edges[0] * np.array([-0.2, 0.2])), density)
    y = np.linspace(*(bbox[1] + edges[1] * np.array([-0.2, 0.2])), density)
    x_mesh, y_mesh = np.meshgrid(x, y)
    point_mesh = np.stack([x_mesh, y_mesh], axis=-1)

    # Determine which points of the meshgrid lie inside the ellipse
    in_ellipse = is_inside_ellipse(point_mesh, D, p)

    # Plot the meshgrid
    plt.pcolormesh(x_mesh, y_mesh, in_ellipse, cmap="Wistia")  # Ellipse region
    plt.scatter(*p, marker="*", color="b", label="Center")  # Center of the ellipse

    # Plot the points (if given)
    if points is not None:  # Convert points into a numpy array
        points_ = np.array(points)
        color = np.full(points_.shape[:-1], fill_value="m")
        color[is_inside_ellipse(points_, D, p)] = "c"
        plt.scatter(points_[:, 0], points_[:, 1], marker="x", c=color, label="Points")

    # Plot the BBOX of the ellipse
    rect = Rectangle(
        p - edges / 2, *edges, linewidth=1, edgecolor="g", facecolor="none", label="BBOX"
    )
    plt.gca().add_patch(rect)

    # Appearance
    plt.title("Plot of the ellipse")
    plt.legend()
    plt.axis("equal")
    plt.show()

