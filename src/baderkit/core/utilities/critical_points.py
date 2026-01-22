# -*- coding: utf-8 -*-

from .interpolation import interp_spline

@njit(cache=True, inline="always")
def get_gradient(
    data: NDArray[np.float64],
    voxel_coord: NDArray[np.int64],
    dir2lat: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], np.bool_]:
    """
    Peforms a neargrid step from the provided voxel coordinate.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    voxel_coord : NDArray[np.int64]
        The point to make the step from.
    car2lat : NDArray[np.float64]
        A matrix that converts a coordinate in cartesian space to fractional
        space.

    Returns
    -------
    charge_grad_frac : NDArray[np.float64]
        The gradient in direct space at this voxel coord

    """
    nx, ny, nz = data.shape
    i, j, k = voxel_coord
    # calculate the gradient at this point in voxel coords
    charge000 = data[i, j, k]
    charge001 = data[i, j, (k + 1) % nz]
    charge010 = data[i, (j + 1) % ny, k]
    charge100 = data[(i + 1) % nx, j, k]
    charge00_1 = data[i, j, (k - 1) % nz]
    charge0_10 = data[i, (j - 1) % ny, k]
    charge_100 = data[(i - 1) % nx, j, k]

    gi = (charge100 - charge_100) / 2.0
    gj = (charge010 - charge0_10) / 2.0
    gk = (charge001 - charge00_1) / 2.0

    if charge100 <= charge000 and charge_100 <= charge000:
        gi = 0.0
    if charge010 <= charge000 and charge0_10 <= charge000:
        gj = 0.0
    if charge001 <= charge000 and charge00_1 <= charge000:
        gk = 0.0

    # convert to direct
    # NOTE: Doing this rather than the original car2lat with two np.dot operations
    # saves about half the time.
    r0 = dir2lat[0, 0] * gi + dir2lat[0, 1] * gj + dir2lat[0, 2] * gk
    r1 = dir2lat[1, 0] * gi + dir2lat[1, 1] * gj + dir2lat[1, 2] * gk
    r2 = dir2lat[2, 0] * gi + dir2lat[2, 1] * gj + dir2lat[2, 2] * gk
    return r0, r1, r2