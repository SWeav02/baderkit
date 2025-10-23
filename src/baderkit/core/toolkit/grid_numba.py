# -*- coding: utf-8 -*-
"""
Numba-based 3D RegularGridInterpolator for periodic fractional coordinates.
Supports: nearest, linear, cubic, quintic
"""
import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(cache=True, inline="always")
def wrap_point(
    i: np.int64, j: np.int64, k: np.int64, nx: np.int64, ny: np.int64, nz: np.int64
) -> tuple[np.int64, np.int64, np.int64]:
    """
    Wraps a 3D point (i, j, k) into the periodic bounds defined by the grid dimensions (nx, ny, nz).

    If any of the input coordinates are outside the bounds [0, nx), [0, ny), or [0, nz),
    they are wrapped around using periodic boundary conditions.

    Parameters
    ----------
    i : np.int64
        x-index of the point.
    j : np.int64
        y-index of the point.
    k : np.int64
        z-index of the point.
    nx : np.int64
        Number of grid points along x-direction.
    ny : np.int64
        Number of grid points along y-direction.
    nz : np.int64
        Number of grid points along z-direction.

    Returns
    -------
    tuple[np.int64, np.int64, np.int64]
        The wrapped (i, j, k) indices within the bounds.
    """
    if i >= nx:
        i -= nx
    elif i < 0:
        i += nx
    if j >= ny:
        j -= ny
    elif j < 0:
        j += ny
    if k >= nz:
        k -= nz
    elif k < 0:
        k += nz
    return i, j, k


###############################################################################
# Sig Fig Management
###############################################################################


# @njit(cache=True)
# def round_sig(value, num_sig_figs):
#     if value == 0:
#         return 0.0
#     return round(value, int(num_sig_figs - np.floor(np.log10(abs(value))) - 1))


###############################################################################
# Nearest point interpolation
###############################################################################


@njit(inline="always", cache=True, fastmath=True)
def interp_nearest(i, j, k, data, is_frac=True):
    nx, ny, nz = data.shape
    if is_frac:
        # convert to voxel coordinates
        i = i * nx
        j = j * ny
        k = k * nz

    # round and wrap
    ix = int(round(i)) % nx
    iy = int(round(j)) % ny
    iz = int(round(k)) % nz

    return data[ix, iy, iz]


###############################################################################
# Linear interpolation
###############################################################################
@njit(inline="always", cache=True, fastmath=True)
def interp_linear(i, j, k, data, is_frac=True):
    nx, ny, nz = data.shape

    if is_frac:
        # convert to voxel coordinates
        i = i * nx
        j = j * ny
        k = k * nz

    # wrap coord
    i, j, k = wrap_point(i, j, k, nx, ny, nz)

    # get rounded down voxel coords
    ri = int(i // 1.0)
    rj = int(j // 1.0)
    rk = int(k // 1.0)

    # get offset from rounded voxel coord
    di = i - ri
    dj = j - rj
    dk = k - rk

    # get data in 2x2x2 cube surrounding point
    v000 = data[ri, rj, rk]
    v100 = data[(ri + 1) % nx, rj, rk]
    v010 = data[ri, (rj + 1) % ny, rk]
    v001 = data[ri, rj, (rk + 1) % nz]
    v110 = data[(ri + 1) % nx, (rj + 1) % ny, rk]
    v101 = data[(ri + 1) % nx, rj, (rk + 1) % nz]
    v011 = data[ri, (rj + 1) % ny, (rk + 1) % nz]
    v111 = data[(ri + 1) % nx, (rj + 1) % ny, (rk + 1) % nz]

    # interpolate value from linear approximation
    return (
        (1 - di) * (1 - dj) * (1 - dk) * v000
        + di * (1 - dj) * (1 - dk) * v100
        + (1 - di) * dj * (1 - dk) * v010
        + (1 - di) * (1 - dj) * dk * v001
        + di * dj * (1 - dk) * v110
        + di * (1 - dj) * dk * v101
        + (1 - di) * dj * dk * v011
        + di * dj * dk * v111
    )


###############################################################################
# Cubic spline interpolation
###############################################################################


@njit(cache=True, inline="always", fastmath=True)
def cubic_bspline_weights(di, dj, dk):
    weights = np.empty((3, 4), dtype=np.float64)
    for d_idx, d in enumerate((di, dj, dk)):
        for i in range(4):
            x = abs((i - 1) - d)
            if x < 1.0:
                w = (4.0 - 6.0 * x * x + 3.0 * x * x * x) / 6.0
            elif x < 2.0:
                t = 2.0 - x
                w = (t * t * t) / 6.0
            else:
                w = 0.0
            weights[d_idx, i] = w
    return weights


@njit(cache=True, fastmath=True)
def interp_spline(i, j, k, data, is_frac=True):
    nx, ny, nz = data.shape

    # convert fractional to voxel coordinates
    if is_frac:
        i = i * nx
        j = j * ny
        k = k * nz

    # round down to get int value
    ri = int(math.floor(i))  # floor works with negative too
    rj = int(math.floor(j))
    rk = int(math.floor(k))

    # get fractional offsets in [0,1)
    di = i - ri
    dj = j - rj
    dk = k - rk

    # calculate weights
    weights = cubic_bspline_weights(di, dj, dk)

    # separable evaluation:
    # first convolve along x for the 4x4 neighborhood in y,z to produce tmp[4,4]
    tmp = np.zeros((4, 4), dtype=np.float64)  # tmp[j_index, k_index]
    for joff in range(4):
        yj = (rj - 1 + joff) % ny
        for koff in range(4):
            zk = (rk - 1 + koff) % nz
            # convolve along x for this (y,z)
            s = 0.0
            for ioff in range(4):
                xi = (ri - 1 + ioff) % nx
                s += weights[0, ioff] * data[xi, yj, zk]
            tmp[joff, koff] = s

    # now convolve tmp by wy along y and wz along z
    val = 0.0
    for joff in range(4):
        for koff in range(4):
            val += weights[1, joff] * weights[2, koff] * tmp[joff, koff]

    return val


###############################################################################
# Methods to interpolate points depending on requested method
###############################################################################


@njit(cache=True)
def interpolate_point(
    point,
    method,
    data,
    is_frac=True,
):
    i, j, k = point
    if method == "nearest":
        value = interp_nearest(i, j, k, data, is_frac)
    elif method == "linear":
        value = interp_linear(i, j, k, data, is_frac)
    elif method == "cubic":
        value = interp_spline(i, j, k, data, is_frac)
    # elif method == "quintic":
    #     value = interp_spline(i, j, k, data, order=5)

    return value


@njit(parallel=True, cache=True)
def interpolate_points(points, method, data, is_frac=True):
    out = np.empty(len(points))
    if method == "nearest":
        for point_idx in prange(len(points)):
            i, j, k = points[point_idx]
            out[point_idx] = interp_nearest(i, j, k, data, is_frac)
    elif method == "linear":
        for point_idx in prange(len(points)):
            i, j, k = points[point_idx]
            out[point_idx] = interp_linear(i, j, k, data, is_frac)
    elif method == "cubic":
        for point_idx in prange(len(points)):
            i, j, k = points[point_idx]
            out[point_idx] = interp_spline(i, j, k, data, is_frac)
    # elif method == "quintic":
    #     for i in prange(len(points)):
    #         i, j, k = points[i]
    #         out[i] = interp_spline(i, j, k, data, order=5)

    return out


@njit(cache=True)
def linear_slice(
    data,
    p1: NDArray[float],
    p2: NDArray[float],
    n: int = 100,
    is_frac=True,
    method="cubic",
):

    x_pts = np.linspace(p1[0], p2[0], num=n)
    y_pts = np.linspace(p1[1], p2[1], num=n)
    z_pts = np.linspace(p1[2], p2[2], num=n)
    coords = np.column_stack((x_pts, y_pts, z_pts))

    return interpolate_points(coords, method, data, is_frac)


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
# Methods for finding offgrid maxima
###############################################################################


@njit(fastmath=True, cache=True)
def refine_frac_max(grid, frac_coords, lattice):
    """
    Numerically stable refinement of a local maximum on a 3D periodic grid.
    Fits a local quadratic and enforces concavity.

    Parameters
    ----------
    grid : 3D ndarray
        Periodic scalar field (e.g., charge density).
    frac_coords : tuple of float
        Fractional coordinates (fx, fy, fz) of the approximate maximum (in [0, 1)).
    lattice : ndarray, shape (3, 3)
        Lattice vectors as rows (real-space basis).

    Returns
    -------
    refined_frac : ndarray, shape (3,)
        Refined fractional coordinates of the true maximum (wrapped to [0, 1)).
    refined_value : float
        Interpolated value at the refined maximum.
    """

    nx, ny, nz = grid.shape
    fx, fy, fz = frac_coords

    # --- Step 1: nearest grid point
    ix = int(round(fx * nx)) % nx
    iy = int(round(fy * ny)) % ny
    iz = int(round(fz * nz)) % nz

    # --- Step 2: extract 3×3×3 neighborhood
    region = np.empty((3, 3, 3), dtype=grid.dtype)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                region[dx + 1, dy + 1, dz + 1] = grid[
                    (ix + dx) % nx, (iy + dy) % ny, (iz + dz) % nz
                ]

    # --- Step 3: design matrix
    A = np.empty((27, 10), dtype=np.float64)
    b = np.empty(27, dtype=np.float64)
    inv_n = np.array([1.0 / nx, 1.0 / ny, 1.0 / nz])
    row = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                # fractional → Cartesian offset
                fxo = dx * inv_n[0]
                fyo = dy * inv_n[1]
                fzo = dz * inv_n[2]
                ox = lattice[0, 0] * fxo + lattice[1, 0] * fyo + lattice[2, 0] * fzo
                oy = lattice[0, 1] * fxo + lattice[1, 1] * fyo + lattice[2, 1] * fzo
                oz = lattice[0, 2] * fxo + lattice[1, 2] * fyo + lattice[2, 2] * fzo

                A[row, 0] = 1.0
                A[row, 1] = ox
                A[row, 2] = oy
                A[row, 3] = oz
                A[row, 4] = ox * ox
                A[row, 5] = oy * oy
                A[row, 6] = oz * oz
                A[row, 7] = ox * oy
                A[row, 8] = ox * oz
                A[row, 9] = oy * oz
                b[row] = region[dx + 1, dy + 1, dz + 1]
                row += 1

    # --- Step 4: regularized least squares
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)
    # Regularization (Tikhonov) improves numerical stability
    for i in range(10):
        ATA[i, i] += 1e-10
    coeffs = np.linalg.solve(ATA, ATb)

    a0 = coeffs[0]
    ax, ay, az = coeffs[1], coeffs[2], coeffs[3]
    axx, ayy, azz = coeffs[4], coeffs[5], coeffs[6]
    axy, axz, ayz = coeffs[7], coeffs[8], coeffs[9]

    # --- Step 5: solve ∇f = 0 → M * offset = -grad
    M = np.empty((3, 3), dtype=np.float64)
    M[0, 0] = 2.0 * axx
    M[1, 1] = 2.0 * ayy
    M[2, 2] = 2.0 * azz
    M[0, 1] = M[1, 0] = axy
    M[0, 2] = M[2, 0] = axz
    M[1, 2] = M[2, 1] = ayz
    grad = np.array([ax, ay, az])

    # Ensure concave-down surface: flip signs if all curvatures are positive
    trace_M = M[0, 0] + M[1, 1] + M[2, 2]
    if trace_M > 0:
        M *= -1.0
        grad *= -1.0

    # --- Step 6: solve for offset
    try:
        offset_cart = np.linalg.solve(M, -grad)
    except:
        offset_cart = np.zeros(3)

    # --- Step 7: clamp offset to within one voxel size
    step_cart = np.sqrt(np.sum((lattice / np.array([[nx, ny, nz]])) ** 2, axis=1))
    max_step = np.max(step_cart)
    norm_offset = np.sqrt(np.sum(offset_cart**2))
    if norm_offset > max_step:
        offset_cart *= max_step / norm_offset

    # --- Step 8: compute refined value
    x, y, z = offset_cart
    refined_value = (
        a0
        + ax * x
        + ay * y
        + az * z
        + axx * x * x
        + ayy * y * y
        + azz * z * z
        + axy * x * y
        + axz * x * z
        + ayz * y * z
    )

    # Ensure it doesn't fall below nearby grid values
    region_max = np.max(region)
    if refined_value < region_max:
        refined_value = region_max
        offset_cart[:] = 0.0  # fallback: stay at grid point

    # --- Step 9: Cartesian → fractional offset
    frac_offset = np.linalg.solve(lattice.T, offset_cart)
    refined_frac = np.empty(3, dtype=np.float64)
    refined_frac[0] = (fx + frac_offset[0]) % 1.0
    refined_frac[1] = (fy + frac_offset[1]) % 1.0
    refined_frac[2] = (fz + frac_offset[2]) % 1.0

    return refined_frac, refined_value


@njit(parallel=True, cache=True)
def refine_maxima(
    maxima_coords,
    data,
    lattice,
):
    new_coords = np.empty_like(maxima_coords, dtype=np.float64)
    new_values = np.empty(len(maxima_coords), dtype=np.float64)
    for coord_idx in prange(len(maxima_coords)):
        coord = maxima_coords[coord_idx]
        new_coord, new_value = refine_frac_max(data, coord, lattice)
        new_coords[coord_idx] = new_coord
        new_values[coord_idx] = new_value
    # round and wrap coords
    new_coords = np.round(new_coords, 6)
    new_coords %= 1
    return new_coords, new_values


# Method that refines maxima using interpolation. I had a lot of issues with
# ringing/overshooting
# @njit(parallel=True, fastmath=True, cache=True)
# def refine_maxima(
#     maxima_coords,
#     data,
#     neighbor_transforms,
#     tol=1e-8,
#     is_frac=True,
# ):
#     nx, ny, nz = data.shape
#     # copy initial maxima to avoid overwriting them
#     maxima_coords = maxima_coords.copy()
#     # copy transforms to avoid altering in place
#     neighbor_transforms = neighbor_transforms.copy().astype(np.float64)

#     # normalize in each direction to one
#     for transform_idx, transform in enumerate(neighbor_transforms):
#         neighbor_transforms[transform_idx] = transform / np.linalg.norm(transform)

#     # if fractional, convert each coordinate to voxel coords
#     if is_frac:
#         for max_coord in maxima_coords:
#             max_coord[0] *= nx
#             max_coord[1] *= ny
#             max_coord[2] *= nz

#     # get the initial values
#     current_values = interpolate_points(
#         data=data,
#         points=maxima_coords,
#         method="cubic",
#         is_frac=False,
#     )
#     # loop over coords in parallel and optimize positions
#     for coord_idx in prange(len(maxima_coords)):
#         i, j, k = maxima_coords[coord_idx]

#         frac_mult = 1
#         # create initial delta magnitude
#         delta_mag = 1.0
#         loop_count = 0
#         while delta_mag > tol and loop_count < 50:
#             loop_count += 1
#             # increase frac multiplier
#             frac_mult *= 2
#             # get smaller transform than last loop
#             current_trans = neighbor_transforms / frac_mult
#             # get current best position
#             i, j, k = maxima_coords[coord_idx]
#             # loop over transforms and check if they improve our value
#             for si, sj, sk in current_trans:
#                 ti = i + si
#                 tj = j + sj
#                 tk = k + sk
#                 value = interp_spline(
#                     ti,
#                     tj,
#                     tk,
#                     data=data,
#                     is_frac=False,
#                     )
#                 # if value is improved, update the best position/value
#                 if value > current_values[coord_idx]:
#                     current_values[coord_idx] = value
#                     maxima_coords[coord_idx] = (ti, tj, tk)
#                     # calculate magnitude of delta in fractional coordinates
#                     fsi = si / nx
#                     fsj = sj / ny
#                     fsk = sk / nz
#                     delta_mag = (fsi * fsi + fsj * fsj + fsk * fsk) ** 0.5
#     dec = -int(math.log10(tol))
#     if is_frac:
#         # convert to frac, round, and wrap
#         for max_idx, (i, j, k) in enumerate(maxima_coords):
#             i = round(i / nx, dec) % 1.0
#             j = round(j / ny, dec) % 1.0
#             k = round(k / nz, dec) % 1.0
#             maxima_coords[max_idx] = (i, j, k)
#     else:
#         # round and wrap
#         for max_idx, (i, j, k) in enumerate(maxima_coords):
#             i = round(i, dec) % nx
#             j = round(j, dec) % ny
#             k = round(k, dec) % nz
#             maxima_coords[max_idx] = (i, j, k)

#     return maxima_coords, current_values
