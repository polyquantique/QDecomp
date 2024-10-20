import numpy as np
import matplotlib.pyplot as plt


def steiner_ellipse(p1, p2, p3, verbosity=0):
    """
    Calculates the smallest ellipse that passes through the three given points using the Steiner 
    method. The ellipse is represented by the equation (u − p)† D (u − p) <= 1, where p is the 
    center of the ellipse and D is a matrix that defines its shape and orientation.

    :param p1: First point (2D coordinates)
    :param p2: Second point (2D coordinates) 
    :param p3: Third point (2D coordinates)
    :param verbosity: Level of verbosity
    :return: D matrix (defines the shape and orientation of the ellipse), p (center of the ellipse)
    """
    # Convert the points to numpy arrays
    p1_, p2_, p3_ = np.array(p1), np.array(p2), np.array(p3)

    # * * * * * #
    # Check that the ellipse can be defined by the three points

    # Ensure all three points are distinct
    assert (p1_ != p2_).any() and (p2_ != p3_).any(), "The three points must be distinct."

    # Ensure the points are not collinear
    delta1 = p2_ - p1_
    delta2 = p3_ - p2_

    collin_msg = "The three points must not be collinear."
    if (delta1[0] != 0) and (delta2[0] != 0):
        # Avoid division by 0 for slope calculations
        slope1 = delta1[1] / delta1[0]
        slope2 = delta2[1] / delta2[0]
        assert slope1 != slope2, collin_msg

    else:
        # Handle vertical lines to ensure they are not collinear
        assert (delta1[0] != 0) or (delta2[0] != 0), collin_msg
    
    # * * * * * #
    # Calculate the center of the ellipse
    p = (p1_ + p2_ + p3_) / 3  # Center of the ellipse
    
    # Compute useful vectors for the Steiner method
    f1 = p1_ - p
    f2 = (p3_ - p2_) / np.sqrt(3)

    # Define a parametric function for tracing the contour of the ellipse
    contour = lambda t: p + f1 * np.cos(t) + f2 * np.sin(t)

    # Calculate t0 according to the Steiner method
    t0 = np.arctan(2 * f1 @ f2 / (f1@f1 - f2@f2)) / 2

    # Compute the two main axes of the ellipse
    axis1 = contour(t0) - contour(t0 + np.pi)
    axis2 = contour(t0 + np.pi/2) - contour(t0 - np.pi/2)

    # Temporary D matrix (defines the size of the axes)
    # An axis-aligned ellipse is defined by a diagonal matrix whose diagonal values are the inverse
    # of the squares of the half-lengths of the axes.
    D_ = np.diag([(2 / np.linalg.norm(axis1))**2, (2 / np.linalg.norm(axis2))**2])

    # Calculate the rotation matrix based on the orientation of the ellipse
    theta = np.arctan2(*axis2)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

    # Calculate the final D matrix that defines the oriented ellipse
    D = rotation_matrix.T @ D_ @ rotation_matrix

    # If verbosity is set, display a debug plot of the ellipse and its components
    if verbosity >= 5:
        # Collect initial points
        points = np.array([p1_, p2_, p3_])

        # Generate ellipse contour points
        t_range = np.linspace(0, 2*np.pi, 100)
        ellipse = np.array([contour(t_) for t_ in t_range]).T
        
        # Generate a grid for visualizing the ellipse's interior region
        density = 100
        x = np.linspace(-2, 2, density)
        y = np.linspace(-2, 2, density)
        x_mesh, y_mesh = np.meshgrid(x, y)
        point_mesh = np.stack([x_mesh, y_mesh], axis=-1)

        # Determine which points lie inside the ellipse
        in_ellipse = is_inside_ellipse(point_mesh, D, p)

        # Plot the results
        plt.plot(ellipse[0], ellipse[1], c="g", label="Contour of the ellipse")
        plt.pcolormesh(x_mesh, y_mesh, in_ellipse, cmap="Wistia", label="Ellipse region")
        plt.scatter(points[:, 0], points[:, 1], marker="x", color="r", label="Initial points")
        plt.scatter(*p, marker="x", color="b", label="Center")

        plt.title("Debugging of the steiner_ellipse() function")
        plt.legend()
        plt.axis("equal")
        plt.show()
    
    # Return the D matrix and the center p
    return D, p


def is_inside_ellipse(u, D, p):
    """
    Check if a point u (or an array of points) is inside the ellipse defined by matrix D and center
    p. The function works for both single points and arrays of points, where the last dimension of u
    must be 2.

    :param u: The point(s) to be tested, a 2D array of shape (..., 2)
    :param D: Matrix defining the ellipse's shape and orientation
    :param p: Center of the ellipse
    :return: A boolean array indicating if each point is inside the ellipse
    """
    vector = u - p
    is_inside = np.einsum("...i,ij,...j->...", vector, D, vector) <= 1

    return is_inside


if __name__ == "__main__":
    # Define three points to compute the ellipse
    p1 = np.array([1, 0])
    p2 = np.array([0, 1])
    p3 = np.array([1, 1])

    # Find the ellipse that passes through the three points
    D, p = steiner_ellipse(p1, p2, p3, verbosity=5)

    # Test the 'is_inside_ellipse()' function with multiple points
    points_to_test = np.array([
        [1, 1],
        [1, 0],
        [0, 0],
    ])

    # Check if the points are inside the ellipse
    result = is_inside_ellipse(points_to_test, D, p)
    print(result)

    # Test with a single point
    point_to_test = (1, 0)
    result = is_inside_ellipse(point_to_test, D, p)
    print(result)
