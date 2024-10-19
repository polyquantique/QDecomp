import numpy as np
import matplotlib.pyplot as plt


def steiner_ellipse(p1, p2, p3, verbosity=0):
    """
    An ellipse centered in p can be defined with
    (u − p)† D (u − p) <= 1
    where p is its center.

    This function calculates the smallest ellipse that passes through the 3 points given in 
    argument. The ellipse definition is returned as the D matrix and the p vector. The Steiner 
    algorithm is the one implemented by this function.

    :param p1: first point
    :param p2: second point
    :param p3: third point
    :param verbosity: verbosity level
    :return: D matrix, p point
    """
    # Conversion of the points into numpy vectors
    p1_, p2_, p3_ = np.array(p1), np.array(p2), np.array(p3)

    # * * * * * #
    # Verification that the ellipse can be drawn

    # 3 different points
    assert (p1_ != p2_).any() and (p2_ != p3_).any(), "The three points must be distinct."

    # Points are not colinear
    delta1 = p2_ - p1_
    delta2 = p3_ - p2_

    if (delta1[0] != 0) and (delta2[0] != 0):
        # Avoid division by 0
        slope1 = delta1[1] / delta1[0]
        slope2 = delta2[1] / delta2[0]
        assert slope1 != slope2, "The three points must not be colinear."

    else:
        assert (delta1[0] != 0) or (delta1[1] != 0), "The three points must not be colinear."
    
    # * * * * * #
    # Center of the ellipse
    p = (p1_ + p2_ + p3_) / 3  # Center of the ellipse
    
    # Useful vectors
    f1 = p1_ - p
    f2 = (p3_ - p2_) / np.sqrt(3)

    # Function that travels around the ellipse for 0 <= t <= 2pi
    contour = lambda t: p + f1 * np.cos(t) + f2 * np.sin(t)

    # Definition of t0 (see Steiner method)
    t0 = np.arctan(2 * f1 @ f2 / (f1@f1 - f2@f2)) / 2

    # Ellipse axes
    axis1 = contour(t0) - contour(t0 + np.pi)
    axis2 = contour(t0 + np.pi/2) - contour(t0 - np.pi/2)

    # Temporary D matrix
    # An upright ellipse is defined by a diagonal matrix for which each diagonal value is the square
    # of the inverse of the half length of one axis.
    D_ = np.diag([(2 / np.linalg.norm(axis1))**2, (2 / np.linalg.norm(axis2))**2])

    # Rotation matrix
    theta = np.arctan2(*axis2)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

    # D matrix
    D = rotation_matrix.T @ D_ @ rotation_matrix

    # Debug printing
    if verbosity >= 5:
        # Initial points
        points = np.array([p1_, p2_, p3_])

        # Ellipse contour
        t_range = np.linspace(0, 2*np.pi, 100)
        ellipse = np.array([contour(t_) for t_ in t_range]).T
        
        # Ellipse filling
        density = 100
        x = np.linspace(-2, 2, density)
        y = np.linspace(-2, 2, density)
        x_mesh, y_mesh = np.meshgrid(x, y)
        point_mesh = np.stack([x_mesh, y_mesh], axis=-1)

        in_ellipse = is_inside_ellipse(point_mesh, D, p)

        # Results plotting
        plt.plot(ellipse[0], ellipse[1], c="g", label="Contour of the ellipse")
        plt.pcolormesh(x_mesh, y_mesh, in_ellipse, cmap="Wistia", label="Ellipse region")
        plt.scatter(points[:, 0], points[:, 1], marker="x", color="r", label="Initial points")
        plt.scatter(*p, marker="x", color="b", label="Center")

        plt.title("Debugging of the steiner_ellipse() function")
        plt.legend()
        plt.axis("equal")
        plt.show()
    
    # Returning the results
    return D, p


def is_inside_ellipse(u, D, p):
    """
    Determines if a point u = (x, y) is inside the ellipse defined by D and p. u can also be an 
    array of points for which the last dimension is of size 2. The function will return an array of
    of the same shape than u, but without the last dimension.

    :param u: point to be tested
    :param D: matrix definition of the ellipse
    :param p: center of the ellipse

    :return: True if the point is inside the ellipse
    """
    vector = u - p
    is_inside = np.einsum("...i,ij,...j->...", vector, D, vector) <= 1

    return is_inside


if __name__ == "__main__":
    # Points defining the ellipse
    p1 = np.array([1, 0])
    p2 = np.array([0, 1])
    p3 = np.array([1, 1])

    # Find the ellipse
    D, p = steiner_ellipse(p1, p2, p3, verbosity=5)

    # Testing the 'is_inside_ellipse()' function ...
    # ... with many points
    points_to_test = np.array([
        [1, 1],
        [1, 0],
        [0, 0],
    ])

    result = is_inside_ellipse(points_to_test, D, p)
    print(result)

    # ... with one point
    point_to_test = (1, 0)
    result = is_inside_ellipse(point_to_test, D, p)
    print(result)
