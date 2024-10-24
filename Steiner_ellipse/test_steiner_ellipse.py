import matplotlib.pyplot as plt
import numpy as np
import pytest

import steiner_ellipse as se

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
        se.steiner_ellipse_def(p1, p2, p3)


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
        se.steiner_ellipse_def(p1, p2, p3)


test_points_def = [
    [(0, 0), (0, 1), (0.5, 1 / np.sqrt(2))],
    [(0, 0), (1, 2), (3, 4)],
    [(0, -1), (3, 3), (-1, -3)],
]


@pytest.mark.parametrize("p1, p2, p3", test_points_def)
def test_ellipse_def(p1, p2, p3):
    D, p = se.steiner_ellipse_def(p1, p2, p3)
    for pi in (p1, p2, p3):
        vec = pi - p
        print(vec @ D @ vec)
        assert (vec @ D @ vec) == pytest.approx(1)

    assert np.allclose(p, np.mean((p1, p2, p3), axis=0))


@pytest.mark.parametrize("verbosity", (0, 1, 3, 5))
def test_ellipse_verbosity(verbosity):
    plt.switch_backend("Agg")  # To test a function that creates a plot.

    p1, p2, p3 = (0, 0), (1, 0), (0, 1)
    D, p = se.steiner_ellipse_def(p1, p2, p3, verbosity=verbosity)
    assert (D.shape == (2, 2)) and (p.shape == (2,))


@pytest.mark.parametrize("dim", ([], [5], [4, 3]))
def test_is_inside_ellipse(dim):
    points = np.random.uniform(low=-1.5, high=1.5, size=(*dim, 2))

    # Ellipse (circle) of radius 1 centered at (0, 0)
    D = np.array([[1, 0], [0, 1]])
    p = np.array([0, 0])

    res_fun = se.is_inside_ellipse(points, D, p)
    res_calculated = np.sum(points**2, axis=-1) <= 1

    assert np.allclose(res_fun, res_calculated)
