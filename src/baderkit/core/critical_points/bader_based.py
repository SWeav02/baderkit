# -*- coding: utf-8 -*-

from baderkit.core import Bader
from baderkit.core.utilities.basic import wrap_point
import numpy as np
from numpy.typing import NDArray

PARITIES = np.array([
    [0,0,0], # 0D
    [0,0,1], # 1D
    [0,1,0], # 1D
    [0,1,1], # 2D
    [1,0,0], # 1D
    [1,0,1], # 2D
    [1,1,0], # 2D
    [1,1,1], # 3D
    ], dtype=np.uint8)

PARITY_DIMS = np.array([0,1,1,2,1,2,2,3], dtype=np.uint8)
PARITY_TO_INT = np.zeros((2,2,2), dtype=np.int8)
for idx, (i,j,k ) in enumerate(PARITIES):
    PARITY_TO_INT[i,j,k] = idx

CUBE_VERTICES = [
    [-1,-1,-1],
    [1,-1,-1],
    [1,1,-1],
    [-1,1,-1],
    [-1,-1,1],
    [1,-1,1],
    [1,1,1],
    [-1,1,1]
    ]
POLY_VERTICES = [
    [-1,-1],
    [1,-1],
    [1,1],
    [-1,1],
    ]
EDGE_VERTICES = [
    [-1],
    [1],
    ]
VERTEX_VERTICES = [[]]
VERTICES = [CUBE_VERTICES, POLY_VERTICES, EDGE_VERTICES, VERTEX_VERTICES]

CUBE_TO_POLY = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [3, 2, 6, 7],
    [0, 4, 7, 3],
    [1, 5, 6, 2],
    ],dtype=np.uint8)
POLY_TO_EDGE = np.array([
    [0,1],
    [1,2],
    [2,3],
    [3,0],
    ],dtype=np.uint8)
EDGE_TO_VERTEX = np.array([
    [0],
    [1],
    ],dtype=np.uint8)
empty = np.array([[]], dtype=np.uint8)
# queried by number of vertices
FACET_MAPS = (
    empty,
    empty, 
    EDGE_TO_VERTEX, 
    empty, 
    POLY_TO_EDGE, 
    empty, 
    empty, 
    empty, 
    CUBE_TO_POLY,
    )

def parity_to_cofacets(parity: NDArray[int]) -> NDArray[int]:
    """

    Parameters
    ----------
    parity : NDArray[int]
        The parity of a cell.

    Returns
    -------
    NDArray[int]
        A Nx3 array where each row is the relative transform from a cell with
        the input parity to each of its cofacets.

    """
    # get where the parity is 1
    zero = np.where(parity==0)[0]
    # get transformations
    transforms = []
    for i in zero:
        transform = np.zeros(3, dtype=np.int8)
        for j in range(3):
            if j == i:
                transform[j] = 1
                transforms.append(transform.copy())
                transform[j] = -1
                transforms.append(transform.copy())
    # for 3D parity, we add a blank list to ensure array has the same typing
    if len(transforms) == 0:
        return np.array([[]], dtype=np.int8)
    return np.array(transforms, dtype=np.int8)

def get_parity_cofacets() -> list[NDArray]:
    """

    Returns
    -------
    transforms : list[NDArray]
        The relative transforms for each parity that points to a corresponding
        cell's cofacets.

    """
    transforms = []
    for parity in PARITIES[:-1]:
        transforms.append(parity_to_cofacets(parity))
    return tuple(transforms)

PARITY_COFACETS = get_parity_cofacets()

def parity_to_vertices(parity: NDArray[int]) -> NDArray[int]:
    """

    Parameters
    ----------
    parity : NDArray[int]
        The parity of a cell.

    Returns
    -------
    NDArray[int]
        A Nx3 array where each row is the relative transform from a cell with
        the input parity to each of its vertices.

    """
    zero = np.where(parity==0)[0]
    vertex_type = len(zero)
    base_transforms = VERTICES[vertex_type]
    transforms = []
    for trans in base_transforms:
        new_trans = trans.copy()
        for idx in zero:
            new_trans.insert(idx,0)
        transforms.append(new_trans)
    return np.array(transforms)

def get_parity_cofacet_vertices() -> tuple:    
    """

    Returns
    -------
    tuple
        A tuple with four entries, each indexed by parity:
            transforms
                The relative transforms to all vertices required to construct 
                the cofacets of a cell with the indexed parity.
            cofacets
                Lists of arrays representing the indices relative to the transforms
                that construct each cofacet of a cell with the indexed parity.
            cell_indices
                The indices relative to the transforms that construct a cell
                with the indexed parity
            corresponding_cells
                The facet index for each cofacet that corresponds to the cell
                with the given parity

    """
    transforms = []
    cofacets = []
    cell_indices = []
    corresponding_cells = []
    for parity in PARITIES:
        parity_trans = []
        parity_indices = []
        parity_cell_indices = []
        parity_corresponding_cell = []
        # get transform to this cell
        cell_trans = parity_to_vertices(parity)
        for trans in cell_trans:
            parity_cell_indices.append(len(parity_trans))
            parity_trans.append(trans)
        
        # get transform to cofacet
        cofacet_trans = parity_to_cofacets(parity)
        # if we have no cofacets, this is a 3-cell and we just continue
        if cofacet_trans.shape[1] == 0:
            continue
        
        for trans in cofacet_trans:
            cofacet_indices = []
            # get parity of this cofacet
            new_point = parity + trans

            # get vertices of this cofacet
            new_trans = parity_to_vertices(new_point % 2)
            for trans1 in new_trans:
                # add transformations to total list for this parity and get
                # index in transformation list
                cofacet_indices.append(len(parity_trans))
                parity_trans.append(trans + trans1)
            # add the indices of this cofacet to the list for this parity
            parity_indices.append(cofacet_indices)
        # get unique transformations for this parity
        parity_trans, indices, inverse = np.unique(parity_trans, axis=0, return_index=True, return_inverse=True)

        # remap the cofacet indices to point to proper transforms
        parity_indices = [inverse[idx] for idx in parity_indices]
        parity_cell_indices = inverse[parity_cell_indices]
        transforms.append(np.array(parity_trans, dtype=np.int8))
        cofacets.append(np.array(parity_indices, dtype=np.int8))
        cell_indices.append(np.array(parity_cell_indices, dtype=np.int8))
        # determine which facet index of each cofacet corresponds to the central
        # cell
        facet_map = FACET_MAPS[len(parity_indices[0])]

        for cofacet in parity_indices:
            facet_indices = cofacet[facet_map]
            for facet_idx, facet in enumerate(facet_indices):
                if np.all(np.isin(facet, parity_cell_indices)):
                    parity_corresponding_cell.append(facet_idx)
                    break
        corresponding_cells.append(np.array(parity_corresponding_cell, dtype=np.int8))
            
    return tuple(transforms), tuple(cofacets), tuple(cell_indices), tuple(corresponding_cells)

(
 COFACET_TRANSFORMS, 
 COFACET_VERTICES, 
 COFACET_CELL_VERTICES,  
 COFACET_CELL_INDICES
 ) = get_parity_cofacet_vertices()

def get_cofacet_vertex_values(x, y, z, nx, ny, nz, data, parity_idx):
    parity_trans = COFACET_TRANSFORMS[parity_idx]
    values = np.empty(len(parity_trans), dtype=np.float64)
    for trans_idx, (si, sj, sk) in enumerate(parity_trans):
        ni = int(((x + si) / 2) % nx)
        nj = int(((y + sj) / 2) % ny)
        nk = int(((z + sk) / 2) % nz)
        # print(ni,nj,nk)
        values[trans_idx] = data[ni,nj,nk]
    return values


def get_higher(values1, values2):
    """
    returns:
      0 if values1 > values2
      1 if values2 > values1
    """
    m1 = values1.max()
    m2 = values2.max()

    if m1 > m2:
        return 0
    if m2 > m1:
        return 1

    fmap = FACET_MAPS[len(values1)]
    if fmap.shape[1] == 0:
        # vertex case → SoS
        return 0

    # recurse into highest facets
    f1 = highest_facet(values1)[0]
    f2 = highest_facet(values2)[0]

    return get_higher(f1, f2)

def highest_facet(values):
    """
    values: 1D numpy array of vertex values
    returns:
      facet_values: 1D numpy array
      facet_index: index into facet_maps[len(values)]
    """
    if len(values) == 0:
        return None, None
    fmap = FACET_MAPS[len(values)]

    # Step 1: compute max value per facet
    facet_vals = []
    max_val = -np.inf
    for inds in fmap:
        v = values[inds].max()
        facet_vals.append(v)
        if v > max_val:
            max_val = v

    # Step 2: collect tied facets
    candidates = [
        i for i, v in enumerate(facet_vals) if v == max_val
    ]

    # Step 3: unique winner
    if len(candidates) == 1:
        idx = candidates[0]
        return values[fmap[idx]], idx

    # Step 4: recursive tie-breaking
    best_idx = candidates[0]
    best_vals = values[fmap[best_idx]]

    for idx in candidates[1:]:
        cur_vals = values[fmap[idx]]
        better = get_higher(cur_vals, best_vals)
        if better == 0:
            best_idx = idx
            best_vals = cur_vals

    return best_vals, best_idx

# def find_saddles(labels, data):
#     """
#     Find index-1 saddles from minimum basin labels.
#     Returns a list of edges that contain saddles.
#     """
#     nx, ny, nz = labels.shape

#     directions = [
#         (1,0,0),
#         (0,1,0),
#         (0,0,1),
#     ]
#     critical_points = []
    
#     for i in range(nx):
#         for j in range(ny):
#             for k in range(nz):
#                 m0 = labels[i, j, k]
                
#                 # iterate over discrete morse edges
#                 for dx, dy, dz in directions:
#                     ni = (i + dx) % nx
#                     nj = (j + dy) % ny
#                     nk = (k + dz) % nz
#                     m1 = labels[ni, nj, nk]
                    
#                     # if our labels differ, we may have a saddle point. We need
#                     # to confirm that this edge does not form a gradient pair
#                     # with any of its cofacets (faces of our parallelpipeds)
#                     if m0 != m1:
#                         # get position in double grid
#                         ni = (2*i + dx) % (2*nx)
#                         nj = (2*j + dy) % (2*ny)
#                         nk = (2*k + dz) % (2*nz)
#                         parity = PARITY_TO_INT[dx, dy, dz]
#                         # get cofacets
#                         values = get_cofacet_vertex_values(ni, nj, nk, nx, ny, nz, data, parity)
#                         # get cofacets that have this cell as their highest facet
#                         cofacet_vertices = COFACET_VERTICES[parity]
#                         valid_cofacets = []
#                         for cofacet_idx, cofacet in enumerate(cofacet_vertices):
#                             _, best = highest_facet(values[cofacet])
#                             if best == COFACET_CELL_INDICES[parity][cofacet_idx]:
#                                 valid_cofacets.append(cofacet_idx)
                        
#                         if len(valid_cofacets) == 0:
#                             # if we have no valid cofacets, we mark this as a
#                             # critical point
#                             critical_points.append((i,j,k))
#                             break
#     return critical_points

# @njit(parallel=True, cache=True)
def find_saddles(
    edge_mask,
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    use_minima: bool = False,
):

    nx, ny, nz = data.shape
    # create 3D array to store maxima
    maxima = np.zeros_like(data, dtype=np.bool_)
    # loop over each voxel in parallel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if not edge_mask[i,j,k]:
                    continue
                # get this voxels value
                value = data[i, j, k]
                is_max = True
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    if not edge_mask[ii,jj,kk]:
                        continue
                    
                    if not use_minima:
                        if data[ii, jj, kk] > value:
                            is_max = False
                            break
                    else:
                        if data[ii, jj, kk] < value:
                            is_max = False
                            break
                if is_max:
                    maxima[i, j, k] = True
    return maxima

bader_up = Bader.from_vasp("CHGCAR")

# get basins
labels = bader_up.basin_labels

# get reverse
reverse_grid = bader_up.reference_grid.copy()
reverse_grid.total *= -1
reverse_grid.total += bader_up.reference_grid.total.max() + bader_up.vacuum_tol
reverse_grid.total[bader_up.vacuum_mask] = 0

bader_down = Bader(charge_grid=reverse_grid)
labels_down = bader_down.basin_labels

mask_up = bader_up.basin_edges
mask_down = bader_down.basin_edges
mask_total = mask_up & mask_down

neighbor_transforms, _ = bader_up.reference_grid.point_neighbor_transforms

test = find_saddles(
    edge_mask=mask_total,
    data=bader_up.reference_grid.total,
    neighbor_transforms=neighbor_transforms,
    use_minima=False,
    )
test1 = find_saddles(
    edge_mask=mask_total,
    data=bader_up.reference_grid.total,
    neighbor_transforms=neighbor_transforms,
    use_minima=True,
    )