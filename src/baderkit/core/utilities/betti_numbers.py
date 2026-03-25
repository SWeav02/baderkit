# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange

from baderkit.core.utilities.basic import coords_to_flat, flat_to_coords
from baderkit.core.utilities.union_find import find_root, union

# Full 26-neighborhood (excluding (0,0,0))
NEIGHBOR_TRANSFORMS = np.array(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == dy == dz == 0)
    ],
    dtype=np.int8,
)

NEIGHBOR_TRANSFORMS1 = NEIGHBOR_TRANSFORMS + 1

FACE_TRANSFORMS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
    dtype=np.int8,
)

###############################################################################
# Betti Numbers on Mask
###############################################################################


@njit(cache=True)
def get_betti0(vol):
    nx, ny, nz = vol.shape
    num_vox = nx * ny * nz
    ny_nz = ny * nz

    flat_vol = vol.ravel()
    labels = np.arange(num_vox, dtype=np.uint32)

    no_vol = True

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # skip points not in the volume
                if not vol[i, j, k]:
                    continue

                # note we have some volume and get flat label
                no_vol = False
                flat_idx = coords_to_flat(i, j, k, ny_nz, nz)
                label = labels[flat_idx]

                # check all neighbors
                for di, dj, dk in NEIGHBOR_TRANSFORMS:
                    ni = i + di
                    nj = j + dj
                    nk = k + dk

                    # skip if we're outside our bounds or the neighbor is not
                    # part of the volume
                    if ni < 0 or nj < 0 or nk < 0:
                        continue
                    if ni >= nx or nj >= ny or nk >= nz:
                        continue
                    if not vol[ni, nj, nk]:
                        continue

                    # get neighbors label
                    neigh_flat = coords_to_flat(ni, nj, nk, ny_nz, nz)

                    # only check if the neighbor has a lower index, to avoid
                    # extra union calls
                    if neigh_flat <= flat_idx:
                        continue

                    union(labels, label, labels[neigh_flat])

    if no_vol:
        return 0, labels

    # Compress roots only for occupied voxels
    roots = np.empty_like(labels, dtype=np.uint32)

    for idx, is_vol in enumerate(flat_vol):
        if not is_vol:
            roots[idx] = num_vox
        else:
            roots[idx] = find_root(labels, labels[idx])

    # Count unique connected components, excluding empty space
    return len(np.unique(roots)) - 1, roots


@njit(cache=True)
def get_betti2(vol):
    nx, ny, nz = vol.shape
    num_vol = nx * ny * nz
    ny_nz = ny * nz

    flat_vol = vol.ravel()
    labels = np.arange(num_vol, dtype=np.uint32)

    no_vol = True

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # skip points in the volume
                if vol[i, j, k]:
                    # note we have some volume and get flat label
                    no_vol = False
                    continue

                flat_idx = coords_to_flat(i, j, k, ny_nz, nz)
                label = labels[flat_idx]

                # check all neighbors
                for di, dj, dk in FACE_TRANSFORMS:
                    ni = i + di
                    nj = j + dj
                    nk = k + dk

                    # skip if we're outside our bounds or the neighbor is part
                    # of the volume
                    if ni < 0 or nj < 0 or nk < 0:
                        continue
                    if ni >= nx or nj >= ny or nk >= nz:
                        continue
                    if vol[ni, nj, nk]:
                        continue

                    # get neighbors label
                    neigh_flat = coords_to_flat(ni, nj, nk, ny_nz, nz)

                    # only check if the neighbor has a lower index, to avoid
                    # extra union calls
                    if neigh_flat <= flat_idx:
                        continue

                    union(labels, label, labels[neigh_flat])

    if no_vol:
        return 0, np.zeros_like(labels, dtype=np.uint32)

    # Compress roots only for occupied volels
    roots = np.empty_like(labels, dtype=np.uint32)

    for idx, is_vol in enumerate(flat_vol):
        if is_vol:
            roots[idx] = num_vol
        else:
            roots[idx] = find_root(labels, labels[idx])

    # we want to return the number of unique roots that are not part of the
    # volume or the edges. By construction the edges are fully connected, and
    # the volume has 1 label, so we subtract 2.
    return len(np.unique(roots)) - 2, roots


@njit(cache=True)
def get_euler_characteristic(vol):
    """
    Compute Euler characteristic of a 3D boolean array
    using a cubical complex (26-connected foreground).
    """
    nx, ny, nz = vol.shape

    N0 = 0  # vertices
    N1 = 0  # edges
    N2 = 0  # faces
    N3 = 0  # cubes

    # --- Count cubes ---
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if vol[i, j, k]:
                    N3 += 1

    # --- Count vertices ---
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                found = False
                for di in (-1, 0):
                    for dj in (-1, 0):
                        for dk in (-1, 0):
                            ii = i + di
                            jj = j + dj
                            kk = k + dk
                            if (
                                0 <= ii < nx
                                and 0 <= jj < ny
                                and 0 <= kk < nz
                                and vol[ii, jj, kk]
                            ):
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    N0 += 1

    # --- Count edges ---
    # x-edges
    for i in range(nx):
        for j in range(ny + 1):
            for k in range(nz + 1):
                found = False
                for dj in (-1, 0):
                    for dk in (-1, 0):
                        jj = j + dj
                        kk = k + dk
                        if 0 <= jj < ny and 0 <= kk < nz and vol[i, jj, kk]:
                            found = True
                            break
                    if found:
                        break
                if found:
                    N1 += 1

    # y-edges
    for i in range(nx + 1):
        for j in range(ny):
            for k in range(nz + 1):
                found = False
                for di in (-1, 0):
                    for dk in (-1, 0):
                        ii = i + di
                        kk = k + dk
                        if 0 <= ii < nx and 0 <= kk < nz and vol[ii, j, kk]:
                            found = True
                            break
                    if found:
                        break
                if found:
                    N1 += 1

    # z-edges
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz):
                found = False
                for di in (-1, 0):
                    for dj in (-1, 0):
                        ii = i + di
                        jj = j + dj
                        if 0 <= ii < nx and 0 <= jj < ny and vol[ii, jj, k]:
                            found = True
                            break
                    if found:
                        break
                if found:
                    N1 += 1

    # --- Count faces ---
    # xy-faces
    for i in range(nx):
        for j in range(ny):
            for k in range(nz + 1):
                if (k > 0 and vol[i, j, k - 1]) or (k < nz and vol[i, j, k]):
                    N2 += 1

    # xz-faces
    for i in range(nx):
        for j in range(ny + 1):
            for k in range(nz):
                if (j > 0 and vol[i, j - 1, k]) or (j < ny and vol[i, j, k]):
                    N2 += 1

    # yz-faces
    for i in range(nx + 1):
        for j in range(ny):
            for k in range(nz):
                if (i > 0 and vol[i - 1, j, k]) or (i < nx and vol[i, j, k]):
                    N2 += 1

    return N0 - N1 + N2 - N3


@njit(cache=True)
def get_betti_numbers(vol):
    b0, flat_labels = get_betti0(vol)
    b2 = get_betti2(vol)
    chi = get_euler_characteristic(vol)
    b1 = b0 + b2 - chi
    return b0, b1, b2


@njit(parallel=True, cache=True)
def get_all_betti_numbers(groups, shape):

    betti_nums = np.empty((len(groups), 3), dtype=np.uint16)

    for group_idx in prange(len(groups)):
        group = groups[group_idx]

        if len(group) <= 2:
            # this group can't form a convex hull and must therefore be a
            # point extremum
            betti_nums[group_idx] = (1, 0, 0)
            continue

        # convert to fractional coords
        frac_coords = group / shape

        # adjust all points relative to a single reference point
        ref = frac_coords[0]
        d = frac_coords - ref
        d -= np.round(d)
        frac_coords = ref + d
        # convert back to grid indices
        coords = np.round((frac_coords * shape)).astype(np.int16)

        # create box around volume with 1 voxel of padding
        mins = np.empty(3, dtype=np.int16)
        maxs = np.empty(3, dtype=np.int16)
        for i in range(3):
            mins[i] = np.min(coords[:, i])
            maxs[i] = np.max(coords[:, i])

        bi, bj, bk = maxs - mins + 3
        vol = np.zeros((bi, bj, bk), dtype=np.bool_)

        # create volume vol. get indices starting one voxel from the edge
        idx = (coords - mins) + 1
        for i, j, k in idx:
            vol[i, j, k] = True

        # get betti numbers
        b0, b1, b2 = get_betti_numbers(vol)
        betti_nums[group_idx] = (b0, b1, b2)

    return betti_nums


###############################################################################
# Betti Numbers Scanning Through Values
###############################################################################


@njit(cache=True)
def euler_delta(active, scratch_mask, i, j, k, nx, ny, nz, ny_nz):
    """
    Robust local Euler change for activating voxel (i,j,k).
    Computes Euler on the clipped 3x3x3 window before/after activation
    and returns (chi_after - chi_before).
    """
    # create mask
    mask = scratch_mask
    mask[1, 1, 1] = False

    for (si, sj, sk), (si1, sj1, sk1) in zip(NEIGHBOR_TRANSFORMS, NEIGHBOR_TRANSFORMS1):
        ni = (i + si) % nx
        nj = (j + sj) % ny
        nk = (k + sk) % nz
        flat_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
        if active[flat_idx]:
            mask[si1, sj1, sk1] = True
        else:
            mask[si1, sj1, sk1] = False
    # get chi before and after
    chi_before = get_euler_characteristic(mask)

    # activate center
    mask[1, 1, 1] = True

    chi_after = get_euler_characteristic(mask)

    return chi_after - chi_before


@njit(cache=True)
def get_betti0_and_chi_scanning(flat_data, flat_indices, vol_num, nx, ny, nz):

    ny_nz = ny * nz

    active = np.zeros(vol_num, np.uint8)
    labels = np.arange(vol_num, dtype=np.uint32)

    b0 = 0
    chi = 0

    scratch_chi_mask = np.empty((3, 3, 3), dtype=np.bool_)

    values = np.empty(vol_num, np.float64)
    bettis = np.empty(vol_num, np.int64)
    chis = np.empty(vol_num, np.int64)
    m = 0
    val = np.inf
    for idx in flat_indices:
        prev_val = val

        b0 += 1  # new component
        val = flat_data[idx]

        # get coord
        i, j, k = flat_to_coords(idx, ny_nz, nz)

        # update chi
        chi += euler_delta(active, scratch_chi_mask, i, j, k, nx, ny, nz, ny_nz)

        # note point is active
        active[idx] = 1

        # iterate over neighbors
        for di, dj, dk in NEIGHBOR_TRANSFORMS:
            ni = i + di
            nj = j + dj
            nk = k + dk
            # make sure we're within our bounds
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                # get flat coord
                nidx = coords_to_flat(ni, nj, nk, ny_nz, nz)
                if active[nidx]:
                    # get roots and make union
                    r1 = find_root(labels, idx)
                    r2 = find_root(labels, nidx)
                    if r1 != r2:
                        union(labels, r1, r2)
                        b0 -= 1
        # we only care about points where there is a complete solid.
        # if b0 != 1:
        #     continue
        # note if our betti number changed
        if val != prev_val:
            values[m] = val
            bettis[m] = b0
            chis[m] = chi
            m += 1
        else:
            bettis[m - 1] = b0
            chis[m - 1] = chi

    return bettis[:m], chis[:m], values[:m]


@njit(cache=True)
def get_betti2_scanning(vol, order, flat_vol, flat_data, vol_num, nx, ny, nz):
    ny_nz = ny * nz

    # get the initial betti2 for the entire solid
    b2, labels = get_betti2(vol)

    # set solid points back to their initial labels
    labels[order] = order

    active = ~flat_vol

    values = np.empty(vol_num + 1, np.float64)
    bettis = np.empty(vol_num + 1, np.int64)
    val = flat_data[order[0]]
    values[0] = val
    bettis[0] = b2
    m = 1

    for idx in order:
        prev_val = val
        # not this is an active point
        active[idx] = True
        b2 += 1  # new component
        val = flat_data[idx]

        # get coord
        i, j, k = flat_to_coords(idx, ny_nz, nz)

        # iterate over neighbors
        for di, dj, dk in FACE_TRANSFORMS:
            ni = i + di
            nj = j + dj
            nk = k + dk
            # make sure we're within our bounds
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                # get flat coord
                nidx = coords_to_flat(ni, nj, nk, ny_nz, nz)
                if active[nidx]:
                    # get roots and make union
                    r1 = find_root(labels, idx)
                    r2 = find_root(labels, nidx)
                    if r1 != r2:
                        union(labels, r1, r2)
                        b2 -= 1
        # note if our betti number changed
        if val != prev_val:
            values[m] = flat_data[idx]
            bettis[m] = b2
            m += 1
        else:
            bettis[m - 1] = b2

    return bettis[:m], values[:m]


@njit(cache=True)
def get_betti_numbers_scanning(vol, flat_vol, flat_data, flat_indices, use_minima):
    nx, ny, nz = vol.shape
    vol_num = nx * ny * nz

    if use_minima:
        flat_indices_rev = np.flip(flat_indices)
    else:
        flat_indices_rev = flat_indices
        flat_indices = np.flip(flat_indices_rev)

    # get b0s and chis
    b0s, chis, b0_vals = get_betti0_and_chi_scanning(
        flat_data, flat_indices, vol_num, nx, ny, nz
    )

    # get b2s
    b2s, b2_vals = get_betti2_scanning(
        vol, flat_indices_rev, flat_vol, flat_data, vol_num, nx, ny, nz
    )

    # get unique groups of betti numbers
    all_vals = np.unique(np.concatenate((b0_vals, b2_vals)))
    if use_minima:
        all_vals = np.flip(all_vals)
        b2s = np.flip(b2s)
        b2_vals = np.flip(b2_vals)
    else:
        b0_vals = np.flip(b0_vals)
        chis = np.flip(chis)
        b0s = np.flip(b0s)

    b0_idx = np.searchsorted(b0_vals, all_vals, side="left")
    b2_idx = np.searchsorted(b2_vals, all_vals, side="left")

    # clamp for safety
    b0_idx = np.clip(b0_idx, 0, len(b0s) - 1)
    b2_idx = np.clip(b2_idx, 0, len(b2s) - 1)

    b0 = b0s[b0_idx]
    b2 = b2s[b2_idx]
    chi = chis[b0_idx]
    b1 = b0 + b2 - chi

    all_bettis = np.column_stack((b0, b1, b2)).astype(np.int64)

    betti_type = 0
    best_val = all_vals[0]
    for val, (b0, b1, b2) in zip(all_vals, all_bettis):
        if b0 == 1 and b1 == 0 and b2 == 1:
            betti_type = 2
            best_val = val
            break
        elif b0 == 1 and b1 == 1 and b2 == 0 and betti_type == 0:
            betti_type = 1
            best_val = val

    if betti_type == 2:
        return 1, 0, 1, best_val
    elif betti_type == 1:
        return 1, 1, 0, best_val
    else:
        return 1, 0, 0, best_val


@njit(parallel=True, cache=True)
def get_all_betti_numbers_scanning(groups, group_vals, data, use_minima):
    shape = np.array(data.shape, dtype=np.int64)
    betti_nums = np.empty((len(groups), 3), dtype=np.uint16)
    best_vals = group_vals.copy()

    for group_idx in prange(len(groups)):
        group = groups[group_idx]

        if len(group) <= 2:
            # this group can't form a convex hull and must therefore be a
            # point extremum
            betti_nums[group_idx] = (1, 0, 0)
            continue

        # convert to fractional coords
        frac_coords = group / shape

        # adjust all points relative to a single reference point
        ref = frac_coords[0]
        d = frac_coords - ref
        d -= np.round(d)
        frac_coords = ref + d
        # convert back to grid indices
        coords = np.round((frac_coords * shape)).astype(np.int16)

        # create box around volume with 1 voxel of padding
        mins = np.empty(3, dtype=np.int16)
        maxs = np.empty(3, dtype=np.int16)
        for i in range(3):
            mins[i] = np.min(coords[:, i])
            maxs[i] = np.max(coords[:, i])

        bi, bj, bk = maxs - mins + 3
        vol = np.zeros((bi, bj, bk), dtype=np.bool_)

        # create volume vol. get indices starting one voxel from the edge
        idx = (coords - mins) + 1
        vol_data = np.zeros((bi, bj, bk), dtype=np.float64)
        for i, ((i, j, k), coord) in enumerate(zip(idx, coords)):
            vol[i, j, k] = True
            ci, cj, ck = coord % shape
            vol_data[i, j, k] = data[ci, cj, ck]

        flat_vol = vol.ravel()

        # sort data
        flat_indices = np.where(flat_vol)[0]
        flat_data = vol_data.ravel()
        ordered = np.argsort(flat_data[flat_indices])
        flat_indices = flat_indices[ordered]

        # get betti numbers
        b0, b1, b2, val = get_betti_numbers_scanning(
            vol,
            flat_vol,
            flat_data,
            flat_indices,
            use_minima,
        )
        betti_nums[group_idx] = (b0, b1, b2)
        best_vals[group_idx] = val

    return betti_nums, best_vals