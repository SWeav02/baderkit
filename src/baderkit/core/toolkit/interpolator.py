# -*- coding: utf-8 -*-

import numpy as np

from baderkit.core.utilities.interpolation import interpolate_points

###############################################################################
# Wrapper class for interpolation
###############################################################################


class Interpolator:
    """
    A helper class for interpolating values from a regular periodic grid. Points
    are assumed to be in fractional coordinates.
    
    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D array representing values on the periodic grid. If the `cubic` method
        is used, this should first be filtered using scipy's spline_filter.
    method : str
        The method to use for interpolation. Current options are nearest, linear,
        or cubic.
    """
    
    def __init__(self, data, method="cubic"):
        self.data = np.asarray(data)
        self.method = method

    def __call__(self, points):
        # get points as a numpy array
        points = np.asarray(points, dtype=np.float64)
        # if 1D, convert to 2D
        if points.ndim == 1:
            points = points[None, :]

        return interpolate_points(
            points,
            self.method,
            self.data,
        )