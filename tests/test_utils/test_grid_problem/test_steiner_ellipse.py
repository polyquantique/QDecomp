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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qdecomp.grid_problem import steiner_ellipse as se
from qdecomp.plot import plot_steiner_ellipse as pse

test_points_collin = [
    [(0, 1), (0, 2), (0, -5)],
    [(1, 0), (2, 0), (-5, 0)],
    [(2, 1), (3, 2), (-5, -6)],
]


@pytest.mark.parametrize("p1, p2, p3", test_points_collin)
def test_collinearity_error(p1, p2, p3):
    """
    Check if a ValueError is raised when the given points are collinear.
    """
    with pytest.raises(ValueError, match="The three points must not be collinear."):
        se.assert_steiner_ellipse(np.asarray(p1), np.asarray(p2), np.asarray(p3))


test_points_same_pts = [
    [(0, 0), (2, 3), (2, 3)],
    [(2, 3), (1, 1), (2, 3)],
    [(-1, 6), (-1, 6), (2, 3)],
    [(2, 4), (2, 4), (2, 4)],
]


@pytest.mark.parametrize("p1, p2, p3", test_points_same_pts)
def test_same_pts_error(p1, p2, p3):
    """
    Check if a ValueError is raised when at least two of the given points are the same.
    """
    with pytest.raises(ValueError, match="The three points must be distinct."):
        se.assert_steiner_ellipse(np.asarray(p1), np.asarray(p2), np.asarray(p3))


test_points_def = [
    [(0, 0), (0, 1), (0.5, 1 / np.sqrt(2))],
    [(0, 0), (1, 2), (3, 4)],
    [(0, -1), (3, 3), (-1, -3)],
]


@pytest.mark.parametrize("p1, p2, p3", test_points_def)
def test_ellipse_def(p1, p2, p3):
    """
    Check that the D matrix and p vector are a good definition of the ellipse.
    """
    D, p = se.steiner_ellipse_def(p1, p2, p3)
    for p_i in (p1, p2, p3):
        vec = p_i - p
        assert (vec @ D @ vec) == pytest.approx(1)

    assert np.allclose(p, np.mean((p1, p2, p3), axis=0))


@pytest.mark.parametrize("dim", ([], [5], [4, 3]))
def test_is_inside_ellipse(dim):
    """
    Test that the "is_inside_ellipse()" function can distinguish points within the ellipse and those
    outside it. Also test if the dimension of the output is the same as the input.
    """
    points = np.random.uniform(low=-1.5, high=1.5, size=(*dim, 2))

    # Ellipse (circle) of radius 1 centered at (0, 0)
    D = np.array([[1, 0], [0, 1]])
    p = np.array([0, 0])

    res_fun = se.is_inside_ellipse(points, D, p)
    res_calculated = np.sum(points**2, axis=-1) <= 1

    assert np.allclose(res_fun, res_calculated)
    assert res_calculated.shape == tuple(dim)


def test_inside_ellipse_D_shape_error():
    """
    Check if an IndexError is raised when the shapes of D and p are incompatible.
    """
    points = np.random.rand(3, 2)
    D = np.eye(3)
    p = np.ones(2)

    match_msg = (
        r"The matrix definition \(shape .*\) and center \(shape .*\) must have compatible "
        + r"dimensions."
    )

    with pytest.raises(IndexError, match=match_msg):
        se.is_inside_ellipse(points, D, p)


def test_inside_ellipse_points_shape_error():
    """
    Check if an IndexError is raised when the shapes of D and p are incompatible.
    """
    points = np.random.rand(3, 3)
    D = np.eye(2)
    p = np.ones(2)

    match_msg = (
        r"The last dimension of the points to test \(shape .*\) must be the same than "
        + r"the number of dimensions of the ellipse \(shape .*\)."
    )
    with pytest.raises(IndexError, match=match_msg):
        se.is_inside_ellipse(points, D, p)


def test_plot_ellipse():
    """
    Test the plot_ellipse() function.
    """
    p1, p2, p3 = (0, 0), (1, 0), (0, 1)
    points_to_plot = np.random.rand(3, 2)
    D, p = se.steiner_ellipse_def(p1, p2, p3)

    _, ax = plt.subplots()
    pse.plot_ellipse(ax, D, p, points_to_plot)
    assert True  # The code has run so far


@pytest.mark.parametrize("center", [(0, 0), (1, 0), (-2, -3)])
def test_ellipse_bbox(center):
    """
    Test the ellipse_bbox() function.
    """
    # Center of the ellipse
    p = np.array(center)

    # Ellipse with a main axis of half-length 2 aligned with axis x and a second main axis of
    # half-length 1 aligned with axis y
    D = np.array([[1 / 4, 0], [0, 1]])

    bbox = se.ellipse_bbox(D, p)

    # Values [:, 1] are the max for each dimension and values [:, 0] are the min
    assert (bbox[:, 0] < bbox[:, 1]).all()

    bbox_expected = np.array([[-2, 2], [-1, 1]]) + p.reshape(-1, 1)

    assert np.allclose(bbox, bbox_expected)
