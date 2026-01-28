# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 19:28:25 2026

@author: sammw
"""
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

# Needed:
# 1. consistent ordering of cell vertices such that proper tie mechanisms can
# be made regardless of the source of the vertices. This means extracting the
# faces from a cell represented by an array of vertices should results in
# new sets of vertices that are also properly ordered.
# 2. mapping from parity to local transforms that reproduce all vertices of
# the corresponding cell, in proper order.
# 3. mapping from parity to local transforms that reproduce all vertices of
# the cofacets of the corresponding cell. This should be accompanied by mapping
# arrays that represent which transforms are required to produce each cofacet
# in proper order. This reduces the number of local transformations that must
# be applied.
# 4. given two appropriately sorted sets of vertex values, compare and
# return which is higher or if it is a tie.

###############################################################################
# Bit to Integer maps
###############################################################################
# We store multiple pieces of data in a single byte which requires conversion
# to an integer

def get_trans_maps() -> tuple:
    """

    Returns
    -------
    tuple
        A tuple with two parts:
            trans_to_int
                An array which maps the 6 unit vector transforms to a single
                integer representation
            int_to_trans
                An array which maps the integer representation of a 6 unit vector
                to the 3 entry transformation

    """
    trans_to_int = np.zeros((3,3,3), dtype=np.int8)
    int_to_trans = np.zeros((6,3), dtype=np.int8)
    for i in range(6):
        j = i // 2
        value = (-1) ** (i & 1) # -1 for even, 1 for odd
        int_to_trans[i, j] = value
        x, y, z = [0 if k != j else value for k in (0, 1, 2)]
        trans_to_int[x,y,z] = i
    return trans_to_int, int_to_trans
TRANS_TO_INT, INT_TO_TRANS = get_trans_maps()

def get_inverse_trans_ints():
    inverse_trans = np.zeros(len(INT_TO_TRANS), dtype=np.int8)
    for i in range(len(inverse_trans)):
        trans = INT_TO_TRANS[i]
        x,y,z = -trans
        inverse_idx = TRANS_TO_INT[x,y,z]
        inverse_trans[i] = inverse_idx
    return inverse_trans
INVERSE_TRANS_IDX = get_inverse_trans_ints()

def get_pointer_maps() -> tuple:
    """
    bit 7 (value 128): blank
    bit 6 (value 64): crit (0/1)
    bits 3-5 (8,16,32): second highest facet transform (0..7)
    bits 0-2 (1,2,4): paired facet/cofacet transform (0..7)
    
    Returns
    -------
    tuple
        A tuple with two parts:
            pointers_to_int
                An array which maps the traversal flag, critical flag, second highest
                facet transformation index, and paired cell transformation index
                to a single np.uint8
            int_to_pointers
                An array which maps a np.uint8 to a traversal flag, critical flag, 
                second highest facet transformation index, and paired cell transformation
                index
                
    """
    pointers_to_int = np.zeros((2, 8, 8), dtype=np.uint8)
    int_to_pointers = np.zeros((255, 3), dtype=np.uint8)
    
    count = 0
    for crit in range(2):
        for face in range(8):
            for cofacet in range(8):
                pointers_to_int[crit, face, cofacet] = count
                int_to_pointers[count] = [crit, face, cofacet]
                count += 1

    return pointers_to_int, int_to_pointers

POINTERS_TO_INT, INT_TO_POINTERS = get_pointer_maps()
    
###############################################################################
# Parities
###############################################################################
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

# Conversion from parity to its index
PARITY_TO_INT = np.zeros((2,2,2), dtype=np.int8)
for idx, (i,j,k ) in enumerate(PARITIES):
    PARITY_TO_INT[i,j,k] = idx

###############################################################################
# Index Maps
###############################################################################
# Given an array of vertex transformations for a cell, indexing with the following
# array of the correct cell type will return the vertices of each of its faces
# Similarly, an array of values at the vertex transforms can be indexed

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

###############################################################################
# Vertices
###############################################################################
# Generate transformations to vertices for each parity type.
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

def get_parity_vertices() -> list[NDArray]:
    """

    Returns
    -------
    transforms : list[NDArray]
        The relative transforms for each parity that generate a corresponding
        cells vertices.

    """
    transforms = []
    for parity in PARITIES:
        transforms.append(parity_to_vertices(parity))
    return tuple(transforms)

PARITY_VERTICES = get_parity_vertices()

###############################################################################
# Facets
###############################################################################
# Generate transformations to facets for each parity type.

CUBE_FACETS = [
    [0,0,-1],
    [0,0,1],
    [0,-1,0],
    [0,1,0],
    [-1,0,0],
    [1,0,0]
    ]
POLY_FACETS = [
    [0,-1],
    [1,0],
    [0,1],
    [-1,0],
    ]
EDGE_FACETS = [
    [-1],
    [1]
    ]
VERTEX_FACETS = []
FACETS = [CUBE_FACETS, POLY_FACETS, EDGE_FACETS, VERTEX_FACETS]

def parity_to_facets(parity: NDArray[int]) -> NDArray[int]:
    """

    Parameters
    ----------
    parity : NDArray[int]
        The parity of a cell.

    Returns
    -------
    NDArray[int]
        A Nx3 array where each row is the relative transform from a cell with
        the input parity to each of its facets.

    """
    zero = np.where(parity==0)[0]
    vertex_type = len(zero)
    base_transforms = FACETS[vertex_type]
    transforms = []
    for trans in base_transforms:
        new_trans = trans.copy()
        for idx in zero:
            new_trans.insert(idx,0)
        transforms.append(new_trans)
    return np.array(transforms)

def get_parity_facets() -> list[NDArray]:
    """

    Returns
    -------
    transforms : list[NDArray]
        The relative transforms for each parity that points to a corresponding
        cell's facets.

    """
    transforms = []
    for parity in PARITIES:
        transforms.append(parity_to_facets(parity))
    return tuple(transforms)
PARITY_FACETS = get_parity_facets()

PARITY_FACETS_INT = tuple([np.array([TRANS_TO_INT[x,y,z] for x,y,z in parity], dtype=np.int8) for parity in PARITY_FACETS])


###############################################################################
# Cofacets
###############################################################################
# Generate transforms to cofacets for each parity
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
PARITY_COFACETS_INT = tuple([np.array([TRANS_TO_INT[x,y,z] for x,y,z in parity], dtype=np.int8) for parity in PARITY_COFACETS])


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

###############################################################################
# Helper Functions
###############################################################################

@njit
def get_cell_vertex_values(x, y, z, nx, ny, nz, data, parity_idx):
    parity_trans = PARITY_VERTICES[parity_idx]
    values = np.empty(len(parity_trans), dtype=np.float64)
    for trans_idx, (si, sj, sk) in enumerate(parity_trans):
        ni = int(((x + si) / 2) % nx)
        nj = int(((y + sj) / 2) % ny)
        nk = int(((z + sk) / 2) % nz)
        # print(ni,nj,nk)
        values[trans_idx] = data[ni,nj,nk]
    return values

@njit
def get_cofacet_vertex_values(x, y, z, nx, ny, nz, data, parity_idx, importance_mask):
    parity_trans = COFACET_TRANSFORMS[parity_idx]
    values = np.empty(len(parity_trans), dtype=np.float64)
    for trans_idx, (si, sj, sk) in enumerate(parity_trans):
        if not importance_mask[trans_idx]:
            continue
        ni = int(((x + si) / 2) % nx)
        nj = int(((y + sj) / 2) % ny)
        nk = int(((z + sk) / 2) % nz)
        # print(ni,nj,nk)
        values[trans_idx] = data[ni,nj,nk]
    return values

@njit
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

@njit
def second_highest_facet(values):
    """
    values: 1D numpy array of vertex values for a cell
    returns:
      facet_values: 1D numpy array of the second-highest facet
      facet_index: index into facet_maps[len(values)]
    """
    fmap = FACET_MAPS[len(values)]

    # First, find the highest facet
    _, highest_idx = highest_facet(values)

    # Collect remaining facet indices
    remaining = [i for i in range(len(fmap)) if i != highest_idx]

    # Initialize best among remaining
    best_idx = remaining[0]
    best_vals = values[fmap[best_idx]]

    # Compare remaining facets structurally
    for idx in remaining[1:]:
        cur_vals = values[fmap[idx]]
        better = get_higher(cur_vals, best_vals)
        if better == 0:
            best_idx = idx
            best_vals = cur_vals

    return best_vals, best_idx

@njit
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



###############################################################################
# Pairing Algorithms
###############################################################################
    
@njit(parallel=True)
def get_second_highest_facets(
        data,
        ):
    # get shape
    nx, ny, nz = data.shape
    
    # create double grid to store cell pointers in
    cell_pointers = np.empty((nx*2, ny*2, nz*2), dtype=np.uint8)
    fi, fj, fk = cell_pointers.shape
    
    # loop over cells and get best facets
    for i in prange(fi):
        for j in range(fj):
            for k in range(fk):
                # get parity index
                parity_idx = PARITY_TO_INT[i&1, j&1, k&1]
                dim = PARITY_DIMS[parity_idx]
                # We will only need these values for 2-cell and 3-cell cofacets
                # so we skip 0-cells and 1-cells
                if dim <= 1:
                    best_trans = 6
                else:
                    # get values at vertices
                    vertex_values = get_cell_vertex_values(i, j, k, nx, ny, nz, data, parity_idx)
                    # get second highest facet
                    _, facet_idx = second_highest_facet(vertex_values)

                    # get index of transform to highest facet
                    best_trans = PARITY_FACETS_INT[parity_idx][facet_idx]
                # get byte and initialize this cell to not have a pair.
                # The index 6 represents no pair assignment

                cell_pointers[i,j,k] = POINTERS_TO_INT[0,best_trans,6]

    return cell_pointers
                
@njit(parallel=True)
def pair_cells_alg1(
        data,
        cell_pointers,
        ):
    # get shapes
    nx, ny, nz = data.shape
    fi, fj, fk = cell_pointers.shape
    
    scratch_mask = np.ones(12, dtype=np.bool_)
    # iterate over each point
    for i in prange(fi):
        for j in range(fj):
            for k in range(fk):
                # get parity index
                parity_idx = PARITY_TO_INT[i&1, j&1, k&1]
                dim = PARITY_DIMS[parity_idx]
                # if we have a 3-cell we can't have cofacets and continue
                if dim == 3:
                    continue
                # get the current pointers
                _,best_facet,pairing_int = INT_TO_POINTERS[cell_pointers[i,j,k]]
                # if we already have a pairing, we can skip
                if pairing_int != 6:
                    continue
                
                # get the values at all vertices involved with cofacets
                vertex_values = get_cofacet_vertex_values(i, j, k, nx, ny, nz, data, parity_idx,
                                                          scratch_mask
                                                          )

                # get cofacets that have this cell as their highest facet
                cofacet_vertices = COFACET_VERTICES[parity_idx]
                valid_cofacets = []
                for cofacet_idx, cofacet in enumerate(cofacet_vertices):
                    _, best = highest_facet(vertex_values[cofacet])
                    if best == COFACET_CELL_INDICES[parity_idx][cofacet_idx]:
                        valid_cofacets.append(cofacet_idx)
                
                if len(valid_cofacets) == 0:
                    # this point has no assignment from this algorithm and
                    # we do nothing
                    continue
                elif len(valid_cofacets) == 1:
                    lowest_cofacet = valid_cofacets[0]
                else:
                    # get the lowest valid cofacet
                    lowest_cofacet = valid_cofacets[0]
                    lowest_values = vertex_values[cofacet_vertices[lowest_cofacet]]
                    for cofacet_idx in valid_cofacets[1:]:
                        values = vertex_values[cofacet_vertices[cofacet_idx]]
                        higher = get_higher(lowest_values, values)
                        if not higher:
                            lowest_cofacet = cofacet_idx
                            lowest_values = values
                
                # get the transformation to this cofacet
                si,sj,sk = PARITY_COFACETS[parity_idx][lowest_cofacet]
                ni = (i+si)%fi
                nj = (j+sj)%fj
                nk = (k+sk)%fk
                
                # pair cell to cofacet
                best_cofacet = TRANS_TO_INT[si,sj,sk]
                cell_pointers[i,j,k] = POINTERS_TO_INT[0,best_facet, best_cofacet]
                
                # pair cofacet to cell
                best_cofacet = INVERSE_TRANS_IDX[best_cofacet]
                _,best_facet,_ = INT_TO_POINTERS[cell_pointers[ni,nj,nk]]
                cell_pointers[ni,nj,nk] = POINTERS_TO_INT[0,best_facet, best_cofacet]
                

@njit(parallel=True)
def pair_cells_alg2(
        data,
        cell_pointers,
        ):
    # get shapes
    nx, ny, nz = data.shape
    fi, fj, fk = cell_pointers.shape

    # iterate over each point
    for i in prange(fi):
        for j in range(fj):
            for k in range(fk):
                # get the current pointers
                _,best_facet,pairing_int = INT_TO_POINTERS[cell_pointers[i,j,k]]

                # if we have a pairing, continue
                if pairing_int != 6:
                    continue
                # get parity index
                parity_idx = PARITY_TO_INT[i&1, j&1, k&1]
                dim = PARITY_DIMS[parity_idx]
                # 0-cells will not be paired if they failed to pair in the
                # previous step so we skip these
                # 3-cells cannot have cofacets so we skip these as well
                if dim == 0 or dim == 3:
                    continue
                # get transformations to cofacets
                cofacet_trans = PARITY_COFACETS[parity_idx]
                cofacet_trans_ints = PARITY_COFACETS_INT[parity_idx]
                # get the valid cofacets
                valid_cofacets = []
                for cofacet_idx, (trans_idx, (si, sj, sk)) in enumerate(zip(cofacet_trans_ints, cofacet_trans)):
                    ni = (i+si)%fi
                    nj = (j+sj)%fj
                    nk = (k+sk)%fk
                    _, second_facet, pair = INT_TO_POINTERS[cell_pointers[ni,nj,nk]]

                    # skip cofacets that have an assignment or don't have this
                    # cell as their second highest facet
                    if pair != 6 or second_facet != INVERSE_TRANS_IDX[trans_idx]:
                        continue
                    # otherwise, we add this as a possible pairing
                    valid_cofacets.append(cofacet_idx)
                
                # if we did not find any valid cofacets, this is a critical point
                # and we can continue
                if len(valid_cofacets) == 0:
                    continue

                # now we get the lowest valid cofacet
                elif len(valid_cofacets) == 1:
                    # if there is only one option, that's what we choose
                    lowest_cofacet = valid_cofacets[0]
                else:
                    # If there is more than one option, we select the lowest.
                    # We only get values at required transforms to avoid extra
                    # operations
                    cofacet_vertices = COFACET_VERTICES[parity_idx]
                    # get important transform indices
                    importance_mask = np.zeros(len(COFACET_TRANSFORMS[parity_idx]), dtype=np.bool_)
                    for cofacet_idx in valid_cofacets:
                        importance_mask[cofacet_vertices[cofacet_idx]] = True
                    vertex_values = get_cofacet_vertex_values(i, j, k, nx, ny, nz, data, parity_idx,importance_mask)
                    
                    # get the lowest valid cofacet
                    lowest_cofacet = valid_cofacets[0]
                    lowest_values = vertex_values[cofacet_vertices[lowest_cofacet]]
                    for cofacet_idx in valid_cofacets[1:]:
                        values = vertex_values[cofacet_vertices[cofacet_idx]]
                        higher = get_higher(lowest_values, values)
                        if not higher:
                            lowest_cofacet = cofacet_idx
                            lowest_values = values
                            
                # get the transformation to this cofacet
                si,sj,sk = PARITY_COFACETS[parity_idx][lowest_cofacet]
                ni = (i+si)%fi
                nj = (j+sj)%fj
                nk = (k+sk)%fk
                
                # pair cell to cofacet
                best_cofacet = TRANS_TO_INT[si,sj,sk]
                _,best_facet,_ = INT_TO_POINTERS[cell_pointers[i,j,k]]
                cell_pointers[i,j,k] = POINTERS_TO_INT[0,best_facet, best_cofacet]
                
                # pair cofacet to cell
                best_cofacet = INVERSE_TRANS_IDX[best_cofacet]
                _,best_facet,_ = INT_TO_POINTERS[cell_pointers[ni,nj,nk]]
                cell_pointers[ni,nj,nk] = POINTERS_TO_INT[0,best_facet, best_cofacet]

@njit(parallel=True)
def get_critical_points(cell_pointers):
    # loop over cells and mark those that don't have a pair as critical.
    fi, fj, fk = cell_pointers.shape

    # iterate over each point
    for i in prange(fi):
        for j in range(fj):
            for k in range(fk):
                pointer = cell_pointers[i,j,k]
                _, second_facet, pair_int = INT_TO_POINTERS[pointer]
                if pair_int != 6:
                    # this is not a critical point and we can continue
                    continue
                # update pointer with critical flag
                cell_pointers[i,j,k] = POINTERS_TO_INT[1,6,6]
    
    # get the coordinates of the critical points
    critical_coords = np.argwhere(cell_pointers==118)
    # get the types of critical points
    critical_types = np.sum(critical_coords % 2, axis=1)
    return critical_coords, critical_types

def get_descending_manifolds(
        cell_pointers,
        critical_coords,
        critical_types
        ):
    fx, fy, fz = np.array(cell_pointers.shape)
    
    # create an array to store labels
    num_crits = len(critical_coords)
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if np.iinfo(dtype).max > num_crits+1:
            break
    descending_manifolds = np.full_like(cell_pointers, num_crits, dtype=dtype)
    # add crit point labels
    descending_manifolds[critical_coords[:,0],critical_coords[:,1],critical_coords[:,2]] = np.arange(num_crits)
    
    # get indices of maxima
    maxima_indices = np.where(critical_types==3)[0]
    
    # label using breadth first search
    for max_idx in prange(len(maxima_indices)):
        crit_idx = maxima_indices[max_idx]
        # get the coordinates and dimensionality of this critical point
        ci, cj, ck = critical_coords[crit_idx]
        dim = critical_types[crit_idx]
        
        # create queues for this critical point
        queue = []
        queue1 = []
        
        # We want to traverse gradient paths to label the descending manifolds
        # To do this, we alternate between d and d-1 cells. We only continue
        # along a path for a d-1 cell if it is paired with a d cell. For each
        # d-cell we add each face that is not paired to it. We label anything
        # that is not critical along the way.
        
        # get the parity of this maximum
        parity = parity_to_int[ci&1, cj&1, ck&1]
        transforms = parity_transforms[parity]
        for si, sj, sk in transforms:
            fi = (ci + si) % fx
            fj = (cj + sj) % fy
            fk = (ck + sk) % fz
            queue.append((fi, fj, fk))
        
        queue_num = 0
        while True:
            if queue_num & 1 == 0:
                current_queue = queue
                queue1 = []
                next_queue = queue1
            else:
                current_queue = queue1
                queue = []
                next_queue = queue
            if len(current_queue) == 0:
                break
            queue_num += 1
            # now we iterate over the queue and add to it as needed
            for i, j, k in current_queue:
                # D-1
                
                # if this point already has an assignment, we can skip immediately
                # as it is either already traversed or a critical point
                if descending_manifolds[i,j,k] != num_crits:
                    continue
                
                # otherwise, we get the coords of the paired point
                crit, facet_int, pair_int = int_to_pointers[cell_pointers[i,j,k]]
                si, sj, sk = int_to_trans[pair_int]
                pi = (i+si)%fx
                pj = (j+sj)%fy
                pk = (k+sk)%fz
                # D-cell
                
                # if this d-cell has already been traversed, or is a critical
                # point, we continue
                if descending_manifolds[pi,pj,pk] != num_crits:
                    continue
                
                # if this d-cell is not the right dimension, we continue
                parity = parity_to_int[pi&1, pj&1, pk&1]
                if parity_dims[parity] != dim:
                    continue
                
                # if we're still here, we want to add both of these cells to our
                # manifold
                descending_manifolds[i, j, k] = crit_idx
                descending_manifolds[pi, pj, pk] = crit_idx

                
                # Now we get all untraversed facets of the current cell and add
                # them to the appropriate list
                for si, sj, sk in transforms:
                    fi = (ci + si) % fx
                    fj = (cj + sj) % fy
                    fk = (ck + sk) % fz
                    # skip already traversed points. This includes crits by
                    # default
                    if descending_manifolds[fi, fj, fk] != num_crits:
                        continue
                    next_queue.append((fi, fj, fk))
    return descending_manifolds