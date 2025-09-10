# -*- coding: utf-8 -*-
"""
Numba-based 3D RegularGridInterpolator for periodic fractional coordinates.
Supports: nearest, linear, cubic, quintic
"""

import numpy as np
from numba import njit, prange

###############################################################################
# Nearest point interpolation
###############################################################################

@njit(inline='always', cache=True, fastmath=True)
def interp_nearest(fx, fy, fz, data):
    nx, ny, nz = data.shape
    # convert fractional coordinates to voxel coords
    ix = int(round(fx*nx)) % nx
    iy = int(round(fy*ny)) % ny
    iz = int(round(fz*nz)) % nz

    return data[ix, iy, iz]

###############################################################################
# Linear interpolation
###############################################################################
@njit(inline='always', cache=True, fastmath=True)
def interp_linear(fx, fy, fz, data):
    nx, ny, nz = data.shape
    
    # get exact voxel coords and wrap
    ex = fx*nx % nx
    ey = fy*ny % ny
    ez = fz*nz % nz
    
    # get rounded down voxel coords
    ix = int(ex // 1.0)
    iy = int(ey // 1.0)
    iz = int(ez // 1.0)
    
    # get offset from rounded voxel coord
    tx = ex - ix
    ty = ey - iy
    tz = ez - iz
    
    # get data in 2x2x2 cube surrounding point
    v000 = data[ix, iy, iz]
    v100 = data[(ix+1) % nx, iy, iz]
    v010 = data[ix, (iy+1) % ny, iz]
    v001 = data[ix, iy, (iz+1) % nz]
    v110 = data[(ix+1) % nx, (iy+1) % ny, iz]
    v101 = data[(ix+1) % nx, iy, (iz+1) % nz]
    v011 = data[ix, (iy+1) % ny, (iz+1) % nz]
    v111 = data[(ix+1) % nx, (iy+1) % ny, (iz+1) % nz]
    
    # interpolate value from linear approximation
    return (
        (1-tx)*(1-ty)*(1-tz)*v000 +
        tx*(1-ty)*(1-tz)*v100 +
        (1-tx)*ty*(1-tz)*v010 +
        (1-tx)*(1-ty)*tz*v001 +
        tx*ty*(1-tz)*v110 +
        tx*(1-ty)*tz*v101 +
        (1-tx)*ty*tz*v011 +
        tx*ty*tz*v111
    )

###############################################################################
# Spline interpolation
###############################################################################
@njit(inline='always', cache=True, fastmath=True)
def cubic_hermite_weights(t):
    """Return 4 cubic Hermite (Catmull-Rom) weights for fractional part t."""
    w_m1 = -0.5*t**3 +     t**2 - 0.5*t
    w_0  =  1.5*t**3 - 2.5*t**2 + 1.0
    w_p1 = -1.5*t**3 + 2.0*t**2 + 0.5*t
    w_p2 =  0.5*t**3 - 0.5*t**2
    return np.array([w_m1, w_0, w_p1, w_p2])

# @njit(inline='always')
# def quintic_bspline_weights(t):
#     """
#     Return 6 quintic B-spline weights for the stencil offsets [-2,-1,0,1,2,3].
#     Valid for t in [0,1). Coefficients are the degree-5 polynomial pieces.
#     """
#     t2 = t * t
#     t3 = t2 * t
#     t4 = t3 * t
#     t5 = t4 * t

#     # coefficients (highest-order first) collapsed into Horner-like evaluation
#     # w0 -> offset -2, w1 -> offset -1, ..., w5 -> offset +3
#     w0 = (-1.0/120.0)*t5 + (1.0/24.0)*t4 + (-1.0/12.0)*t3 + (1.0/12.0)*t2 + (-1.0/24.0)*t + (1.0/120.0)
#     w1 = ( 1.0/24.0)*t5 + (-1.0/6.0)*t4 + (1.0/6.0)*t3 + (1.0/6.0)*t2 + (-5.0/12.0)*t + (13.0/60.0)
#     w2 = (-1.0/12.0)*t5 + (1.0/4.0)*t4 + 0.0*t3 + (-1.0/2.0)*t2 + 0.0*t + (11.0/20.0)
#     w3 = ( 1.0/12.0)*t5 + (-1.0/6.0)*t4 + (-1.0/6.0)*t3 + (1.0/6.0)*t2 + (5.0/12.0)*t + (13.0/60.0)
#     w4 = (-1.0/24.0)*t5 + (1.0/24.0)*t4 + (1.0/12.0)*t3 + (1.0/12.0)*t2 + (1.0/24.0)*t + (1.0/120.0)
#     w5 = ( 1.0/120.0)*t5  # remaining lower-order coefficients are zero

#     return np.array([w0, w1, w2, w3, w4, w5])

@njit(cache=True, fastmath=True, inline='always')
def interp_spline(fx, fy, fz, data):
    """
    3D Hermite cubic interpolation with periodic boundary conditions.
    """
    nx, ny, nz = data.shape

    # scale fractional coords to grid (periodic)
    ex = fx * nx
    ey = fy * ny
    ez = fz * nz
    
    # round down
    ix = int(ex // 1.0)
    iy = int(ey // 1.0)
    iz = int(ez // 1.0)
    
    # get remainder
    dx = ex - ix
    dy = ey - iy
    dz = ez - iz
    
    # get cubic weights
    wx = cubic_hermite_weights(dx)
    wy = cubic_hermite_weights(dy)
    wz = cubic_hermite_weights(dz)
    offset = 1
    size = 4

    # if order == 3:
    #     wx = cubic_hermite_weights(dx)
    #     wy = cubic_hermite_weights(dy)
    #     wz = cubic_hermite_weights(dz)
    #     offset = 1
    #     size = 4
    # elif order == 5:
    #     wx = quintic_bspline_weights(dx)
    #     wy = quintic_bspline_weights(dy)
    #     wz = quintic_bspline_weights(dz)
    #     offset = 2
    #     size = 6
    # else:
    #     raise ValueError("Order must be 3 (cubic) or 5 (quintic)")

    val = 0.0
    for i in range(size):
        xi = (ix - offset + i) % nx
        for j in range(size):
            yj = (iy - offset + j) % ny
            for k in range(size):
                zk = (iz - offset + k) % nz
                val += wx[i] * wy[j] * wz[k] * data[xi, yj, zk]
    return val

###############################################################################
# Methods to interpolate points depending on requested method
###############################################################################

@njit(cache=True)
def interpolate_point(
    point,
    method,
    data,
        ):
    fx, fy, fz = point
    if method == "nearest":
        value = interp_nearest(fx, fy, fz, data)
    elif method == "linear":
        value = interp_linear(fx, fy, fz, data)
    elif method == "cubic":
        value = interp_spline(fx, fy, fz, data)
    # elif method == "quintic":
    #     value = interp_spline(fx, fy, fz, data, order=5)
    
    return value

@njit(parallel=True, cache=True)
def interpolate_points(
    points,
    method,
    data,
        ):
    out = np.empty(len(points))
    if method == "nearest":
        for i in prange(len(points)):
            fx, fy, fz = points[i]
            out[i] = interp_nearest(fx, fy, fz, data)
    elif method == "linear":
        for i in prange(len(points)):
            fx, fy, fz = points[i]
            out[i] = interp_linear(fx, fy, fz, data)
    elif method == "cubic":
        for i in prange(len(points)):
            fx, fy, fz = points[i]
            out[i] = interp_spline(fx, fy, fz, data)
    # elif method == "quintic":
    #     for i in prange(len(points)):
    #         fx, fy, fz = points[i]
    #         out[i] = interp_spline(fx, fy, fz, data, order=5)
    
    return out

###############################################################################
# Wrapper class for interpolation
###############################################################################

class Interpolator:
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

###############################################################################
# Method for finding offgrid maxima
###############################################################################

@njit(parallel=True)
def get_offgrid_maxima(
    ongrid_maxima_frac,
    data,
    neighbor_transforms,
        ):
    nx, ny, nz = data.shape
    frac_trans = neighbor_transforms.copy()
    # normalize neighbor_transforms and convert to frac coords
    for transform_idx in range(len(frac_trans)):
        transform = frac_trans[transform_idx]
        transform /= np.linalg.norm(transform)
        transform[0] /= nx
        transform[1] /= ny
        transform[2] /= nz

    frac_mult = 1
    # get the initial values
    current_values = interpolate_points(ongrid_maxima_frac, "cubic", data)
    current_coords = ongrid_maxima_frac.copy()
    loop_count = 0
    while loop_count < 15:
        # increase frac multiplier
        frac_mult *= 2
        # get smaller transforms than last loop
        current_trans = frac_trans / frac_mult
        # loop over each coord
        for coord_idx in prange(len(current_coords)):
            # get frac coords
            i, j, k = current_coords[coord_idx]
            # loop over transforms and check if they improve our value
            for si, sj, sk in current_trans:
                ti = i + si
                tj = j + sj
                tk = k + sk
                value = interp_spline(ti, tj, tk, data)
                if value > current_values[coord_idx]:
                    current_values[coord_idx] = value
                    current_coords[coord_idx] = (ti, tj, tk)
        loop_count += 1
    # wrap current corods
    current_coords %= 1.0
    return current_coords, current_values
