# -*- coding: utf-8 -*-
from numba import njit, prange
from baderkit.core.utilities.union_find import union, find_root
from baderkit.core.utilities.basic import coords_to_flat

import numpy as np

UPPER_TRANSFORMS = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,1,0],
    [1,0,1],
    [0,1,1],
    [1,1,1]
    ], dtype=np.uint8)

# @njit
def get_betty0(vol):
    nx, ny, nz = vol.shape
    num_vox = nx*ny*nz
    ny_nz = ny*nz
    flat_vol = vol.ravel()
    labels = np.arange(num_vox, dtype=np.uint32)
    
    no_vol = True
    for i in range(nx-1):
        for j in range(ny-1):
            for k in range(nz-1):
                # skip points not in the volume
                if not vol[i,j,k]:
                    continue
                no_vol = False
                # get flat label of this coord
                flat_idx = coords_to_flat(i, j, k, ny_nz, nz)
                label = labels[flat_idx]
                # check all neighbors above this point
                for si, sj, sk in UPPER_TRANSFORMS:
                    ni = i+si
                    nj = j+sj
                    nk = k+sk
                    # skip if neighbor not in volume
                    if not vol[ni,nj,nk]:
                        continue
                    # get neighbor label
                    flat_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
                    # make unions
                    neigh_label = labels[flat_idx]
                    union(labels, label, neigh_label)
    # if we have no volume, we immediately return 0
    if no_vol:
        return 0
    
    # otherwise we get the number of unique labels
    label_roots = np.empty_like(labels, dtype=np.uint32)
    for idx, (label, is_vol) in enumerate(zip(labels, flat_vol)):
        if not is_vol:
            label_roots[idx] = num_vox
            continue

        label_roots[idx] = find_root(labels, label)
        
    # we return the number of unique roots, not including the empty regions. By
    # construction we always have at least some empty regions, and they all have
    # the same label, so we subtract 1
    return len(np.unique(label_roots)) - 1

FACE_TRANFORMS = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    ], dtype=np.uint8)

# @njit
def get_betty2(vol):
    nx, ny, nz = vol.shape
    num_vox = nx*ny*nz
    ny_nz = ny*nz
    flat_vol = vol.ravel()
    labels = np.arange(num_vox, dtype=np.uint32)

    no_vol = True
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # skip points in the volume
                if vol[i,j,k]:
                    no_vol = False
                    continue

                # get flat label of this coord
                flat_idx = coords_to_flat(i, j, k, ny_nz, nz)
                label = labels[flat_idx]
                # check all faces above this point
                for si, sj, sk in FACE_TRANFORMS:
                    ni = i+si
                    nj = j+sj
                    nk = k+sk
                    # skip if we're outside the allowed range
                    if ni>=nx or nj>=ny or nk>=nz:
                        continue
                    # skip if neighbor in volume
                    if vol[ni,nj,nk]:
                        continue
                    # get neighbor label
                    flat_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
                    # make unions
                    neigh_label = labels[flat_idx]
                    union(labels, label, neigh_label)
    # if we have no volume, we cannot have any other betti numbers and we continue
    if no_vol:
        return 0
    
    # otherwise we get the number of unique labels
    label_roots = np.empty_like(labels, dtype=np.uint32)
    for idx, (label, is_vol) in enumerate(zip(labels, flat_vol)):
        # label volume with label outside allowed
        if is_vol:
            label_roots[idx] = num_vox
            continue

        label_roots[idx] = find_root(labels, label)

    # we want to return the number of unique roots that are not part of the
    # volume or the edges. By construction the edges are fully connected, and
    # the volume has 1 label, so we subtract 2.
    return len(np.unique(label_roots)) - 2

@njit
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

# @njit
def get_betti_numbers(vol):
    b0 = get_betty0(vol)
    b2 = get_betty2(vol)
    chi = get_euler_characteristic(vol)
    b1 = b0 + b2 - chi
    return b0, b1, b2

# @njit(parallel=True)
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
    
        box_shape = (maxs - mins + 3)
        vol = np.zeros(box_shape, dtype=bool)
        
        # create volume vol. get indices starting one voxel from the edge
        idx = (coords.astype(int) - mins)+1
        for i,j,k in idx:
            vol[i,j,k] = True
            
        # get betti numbers
        b0, b1, b2 = get_betti_numbers(vol)
        betti_nums[group_idx] = (b0, b1, b2)

        
    return betti_nums
