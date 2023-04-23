import numpy as np

def is_collinear(p0, p1, p2):
    """
    Determines if three points in a two-dimensional space are collinear.
    Helper function for Delaunay triangulation.
    
    Parameters
    ----------
    p0 : tuple
        A tuple containing the (x, y) coordinates of the first point.
    p1 : tuple
        A tuple containing the (x, y) coordinates of the second point.
    p2 : tuple
        A tuple containing the (x, y) coordinates of the third point.
        
    Returns
    -------
    bool
        True if the three points are collinear, False otherwise.
    """
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < 1e-12

def find_collinear_points(points):
    """
    Finds all sets of three collinear points in a two-dimensional array of
    points. Helper function for Delaunay triangulation.
    
    Parameters
    ----------
    points : ndarray
        A numpy array containing the (x, y) coordinates of each point.
        
    Returns
    -------
    list
        A list of tuples representing the indices of all sets of three collinear 
        points found in the input array.
    """
    collinear_indices = []
    n = points.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear(points[i], points[j], points[k]):
                    collinear_indices.append((i,j,k))
    return collinear_indices
