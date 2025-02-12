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
This module allows to plot an ellipse defined by its matrix D and center p (refer to the file
"cliffordplust/grid_problem/steiner_ellipse.py").
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cliffordplust.grid_problem import steiner_ellipse as se


def plot_ellipse(
    ax: plt.Axes, D: np.ndarray, p: np.ndarray, points: np.ndarray | None = None
) -> None:
    """
    Plot the ellipse defined by matrix D and center p. The function also plots the BBOX of the
    ellipse and its center. Moreover, the (optional) points to plot are cyan if they lie inside the
    ellipse of magenta if they are not.

    Args:
        ax (plt.Axes): Axes to plot the ellipse
        D (np.ndarray): Matrix defining the ellipse's shape and orientation
        p (np.ndarray): Center of the ellipse
        points (np.ndarray): Points to plot
    """
    # Find the BBOX of the ellipse
    bbox = se.ellipse_bbox(D, p)  # BBOX of the ellipse
    edges = bbox[:, 1] - bbox[:, 0]  # Length of the edges of the BBOX

    # Generate a grid for visualizing the ellipse's interior region
    density = 200
    x = np.linspace(*(bbox[0] + edges[0] * np.array([-0.2, 0.2])), density)
    y = np.linspace(*(bbox[1] + edges[1] * np.array([-0.2, 0.2])), density)
    x_mesh, y_mesh = np.meshgrid(x, y)
    point_mesh = np.stack([x_mesh, y_mesh], axis=-1)

    # Determine which points of the meshgrid lie inside the ellipse
    in_ellipse = se.is_inside_ellipse(point_mesh, D, p)

    # Plot the meshgrid
    ax.pcolormesh(x_mesh, y_mesh, in_ellipse, cmap="Wistia")  # Ellipse region
    ax.scatter(*p, marker="*", color="b", label="Center")  # Center of the ellipse

    # Plot the points (if given)
    if points is not None:  # Convert points into a numpy array
        points_ = np.array(points)
        color = np.full(points_.shape[:-1], fill_value="m")
        color[se.is_inside_ellipse(points_, D, p)] = "c"
        ax.scatter(points_[:, 0], points_[:, 1], marker="x", c=color, label="Points")

    # Plot the BBOX of the ellipse
    rect = Rectangle(
        p - edges / 2, *edges, linewidth=1, edgecolor="g", facecolor="none", label="BBOX"
    )
    ax.add_patch(rect)

    # Appearance
    ax.set_title("Plot of the ellipse")
    ax.legend()
    ax.axis("equal")
