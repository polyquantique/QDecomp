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
        points = np.array([p1_, p2_, p3_])
        t_range = np.linspace(0, 2*np.pi, 100)
        ellipse = np.array([contour(t_) for t_ in t_range]).T
        
        density = 50
        x = np.linspace(-2, 2, density)
        y = np.linspace(-2, 2, density)
        x_mesh, y_mesh = np.meshgrid(x, y)
        in_ellipse = np.full_like(x_mesh, False)

        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                vect = np.array([x_-p[0], y_-p[1]])
                is_inside = (vect.T @ D @ vect) <= 1
                in_ellipse[j, i] = is_inside
        
        plt.plot(ellipse[0], ellipse[1], label="Contour of the ellipse")
        plt.scatter(x_mesh, y_mesh, marker=".", c=in_ellipse.astype(int), label="Ellipse region")
        plt.scatter(points[:, 0], points[:, 1], marker="x", color="r", label="Initial points")
        plt.title("Debugging of the steiner_ellipse() function")
        plt.legend()
        plt.show()
    
    # Returning the results
    return D, p

if __name__ == "__main__":
    p1 = np.array([1, 0])
    p2 = np.array([0, 1])
    p3 = np.array([1, 1])

    D, p = steiner_ellipse(p1, p2, p3, verbosity=5)
