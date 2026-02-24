# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np

from baderkit.core.utilities.union_find import union, find_root
from baderkit.core.utilities.basic import coords_to_flat

# Full 26-neighborhood (excluding (0,0,0))
NEIGHBOR_TRANSFORMS = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1)
     for dy in (-1, 0, 1)
     for dz in (-1, 0, 1)
     if not (dx == dy == dz == 0)],
    dtype=np.int8
)

@njit(cache=True)
def get_betty0(vol):
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

FACE_TRANSFORMS = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [-1,0,0],
    [0,-1,0],
    [0,0,-1],
    ], dtype=np.int8)

@njit(cache=True)
def get_betty2(vol):
    nx, ny, nz = vol.shape
    num_vox = nx * ny * nz
    ny_nz = ny * nz

    flat_vol = vol.ravel()
    labels = np.arange(num_vox, dtype=np.uint32)

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
        return 0

    # Compress roots only for occupied voxels
    roots = np.empty_like(labels, dtype=np.uint32)

    for idx, is_vol in enumerate(flat_vol):
        if is_vol:
            roots[idx] = num_vox
        else:
            roots[idx] = find_root(labels, labels[idx])

    # we want to return the number of unique roots that are not part of the
    # volume or the edges. By construction the edges are fully connected, and
    # the volume has 1 label, so we subtract 2.
    return len(np.unique(roots)) - 2

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
                                0 <= ii < nx and
                                0 <= jj < ny and
                                0 <= kk < nz and
                                vol[ii, jj, kk]
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
                        if (
                            0 <= jj < ny and
                            0 <= kk < nz and
                            vol[i, jj, kk]
                        ):
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
                        if (
                            0 <= ii < nx and
                            0 <= kk < nz and
                            vol[ii, j, kk]
                        ):
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
                        if (
                            0 <= ii < nx and
                            0 <= jj < ny and
                            vol[ii, jj, k]
                        ):
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
                if (
                    (k > 0 and vol[i, j, k - 1]) or
                    (k < nz and vol[i, j, k])
                ):
                    N2 += 1

    # xz-faces
    for i in range(nx):
        for j in range(ny + 1):
            for k in range(nz):
                if (
                    (j > 0 and vol[i, j - 1, k]) or
                    (j < ny and vol[i, j, k])
                ):
                    N2 += 1

    # yz-faces
    for i in range(nx + 1):
        for j in range(ny):
            for k in range(nz):
                if (
                    (i > 0 and vol[i - 1, j, k]) or
                    (i < nx and vol[i, j, k])
                ):
                    N2 += 1

    return N0 - N1 + N2 - N3

@njit(cache=True)
def get_betti_numbers(vol):
    b0, flat_labels = get_betty0(vol)
    if b0 > 1:
        nx, ny, nz = vol.shape
        num_vox = nx * ny * nz
        # take the largest portion and run again
        a = np.sort(flat_labels)
        n = len(a)
        
        unique = [a[0]]
        counts = [1]
        
        for i in range(1, n):
            # skip background
            if a[i] == num_vox:
                continue
            if a[i] == a[i - 1]:
                counts[-1] += 1
            else:
                unique.append(a[i])
                counts.append(1)
        counts = np.array(counts, dtype=np.int64)
        max_label = unique[np.argmax(counts)]
        vol = (flat_labels==max_label).reshape(vol.shape)
        b0 = 1
        
    b2 = get_betty2(vol)
    chi = get_euler_characteristic(vol)
    b1 = b0 + b2 - chi
    return b0, b1, b2

@njit(parallel=True, cache=True)
def get_all_betti_numbers(groups, shape):
    
    betti_nums = np.empty((len(groups),3), dtype=np.uint16)
    
    for group_idx in prange(len(groups)):
        group = groups[group_idx]

        if len(group) <= 2:
            # this group can't form a convex hull and must therefore be a
            # point extremum
            betti_nums[group_idx] = (1,0,0)
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
            mins[i] = np.min(coords[:,i])
            maxs[i] = np.max(coords[:,i])
    
        bi, bj, bk = maxs - mins + 3
        vol = np.zeros((bi, bj, bk), dtype=np.bool_)
        
        # create volume vol. get indices starting one voxel from the edge
        idx = (coords - mins)+1
        for i,j,k in idx:
            vol[i,j,k] = True
            
        # get betti numbers
        b0, b1, b2 = get_betti_numbers(vol)
        betti_nums[group_idx] = (b0, b1, b2)


        
    return betti_nums

# TODO:
    # 1. get actual values in each group
    # 2. sort
    # 3. add connections from high to low (or low to high for minima)
    # tracking betti numbers along the way. Return a group with b0=1.
    # Give preference to certain shapes: hole > ring > point.