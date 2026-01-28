from numba import njit, prange
import numpy as np

def get_facet_transforms(parity):
    facet_transforms = []
    opposites = []
    for i, p in enumerate(parity):
        trans = np.zeros(3, dtype=np.int8)
        if p == 1:
            trans[i] = -1
            facet_transforms.append(trans.copy())
            opposites.append(1)
            
            trans[i] = 1
            facet_transforms.append(trans.copy())
            opposites.append(-1)
    return facet_transforms, opposites

def get_cofacet_transforms(parity):
    cofacet_transforms = []
    for i, p in enumerate(parity):
        trans = np.zeros(3, dtype=np.int8)
        if p == 0:
            trans[i] = -1
            cofacet_transforms.append(trans.copy())
            trans[i] = 1
            cofacet_transforms.append(trans.copy())
    # add 0,0,0 for 3d parity
    if not cofacet_transforms:
        cofacet_transforms.append(np.zeros(3, dtype=np.int8))
    return cofacet_transforms

def get_trans_maps():
    trans_to_int = np.zeros((3,3,3), dtype=np.int8)
    int_to_trans = np.zeros((6,3), dtype=np.int8)
    for i in range(6):
        j = i // 2
        value = (-1) ** (i & 1) # -1 for even, 1 for odd
        int_to_trans[i, j] = value
        x, y, z = [0 if k != j else value for k in (0, 1, 2)]
        trans_to_int[x,y,z] = i
    return trans_to_int, int_to_trans

def get_parity_maps():
    parity_to_int = np.empty((3,3,3), dtype=np.int8)
    int_to_parity = np.empty((8,3), dtype=np.int8)
    parities = [
        [0,0,0], # 0D
        [0,0,1], # 1D
        [0,1,0], # 1D
        [0,1,1], # 2D
        [1,0,0], # 1D
        [1,0,1], # 2D
        [1,1,0], # 2D
        [1,1,1], # 3D
        ]
    parity_dims = np.array([0,1,1,2,1,2,2,3], dtype=np.int8)
    for parity_idx, (x,y,z) in enumerate(parities):
        parity_to_int[x,y,z] = parity_idx
        int_to_parity[parity_idx] = [x,y,z]
    return parity_to_int, int_to_parity, parity_dims

def get_pointer_maps():
    """
    Assumes:
        bit 7 (value 128): traversal flag (0/1)
        bit 6 (value 64): crit (0/1)
        bits 3-5 (8,16,32): face (0..7)
        bits 0-2 (1,2,4): cofacet (0..7)
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

def get_cofacet_map():
    
    parities = np.array([
        [0,0,0],  # 0D
        [0,0,1],  # 1D
        [0,1,0],  # 1D
        [0,1,1],  # 2D
        [1,0,0],  # 1D
        [1,0,1],  # 2D
        [1,1,0],  # 2D
        [1,1,1],  # 3D
    ])
    parity_cofacet_trans = []
    for parity in parities:
        facet_transforms = get_cofacet_transforms(parity)
        parity_cofacet_trans.append(facet_transforms)
    return parity_cofacet_trans

def get_facet_vertices():
    """
    Generate all facet vertices for each cell parity in 3D.

    Returns:
        parity_facets: list of np.arrays
            Each entry contains the vertices of facets for the corresponding parity.
        parity_indices: list of lists
            Each entry contains indices mapping vertices to facets.
    """

    # Define all possible parities and their dimensions
    parities = np.array([
        [0,0,0],  # 0D
        [0,0,1],  # 1D
        [0,1,0],  # 1D
        [0,1,1],  # 2D
        [1,0,0],  # 1D
        [1,0,1],  # 2D
        [1,1,0],  # 2D
        [1,1,1],  # 3D
    ])
    parity_dims = np.array([0,1,1,2,1,2,2,3], dtype=np.int8)

    parity_transforms = []
    parity_facets = []
    parity_indices = []

    for parity, dim in zip(parities, parity_dims):

        # 0D cells are trivial
        if dim == 0:
            parity_transforms.append([np.array([0,0,0], dtype=int)])
            parity_facets.append([np.array([0,0,0], dtype=int)])
            parity_indices.append([ [0] ])
            continue

        # Get initial facet transforms
        facet_transforms, _ = get_facet_transforms(parity)
        parity_transforms.append(facet_transforms)

        facet_vertices = []
        facet_idx_list = []
        vertex_counter = 0

        # Process each facet
        for facet_transform in facet_transforms:
            # Start with the transform for this facet
            current_transforms = [facet_transform]
            
            # get parity
            current_parities = [(facet_transform + parity) % 2]

            # Iteratively reduce dimension until we reach vertices
            for _ in range(dim - 1):
                new_transforms = []
                new_parities = []
                for current_trans, current_par in zip(current_transforms, current_parities):
                    # Get facet transforms of this intermediate cell
                    subfacet_transforms, _ = get_facet_transforms(current_par)
                    for subfacet_trans in subfacet_transforms:
                        new_transform = current_trans + subfacet_trans
                        new_transforms.append(new_transform)
                        new_parities.append(((new_transform%2) + parity)%2)
                current_transforms = new_transforms
                current_parities = new_parities

            # At this point, current_transforms are all vertices. We first get
            # only the unique ones in case of double-ups (e.g. in 111 case)
            current_transforms = np.unique(current_transforms, axis=0)
            indices = []
            for v in current_transforms:
                facet_vertices.append(v)
                indices.append(vertex_counter)
                vertex_counter += 1

            facet_idx_list.append(indices)
            
        # get unique vertices
        facet_vertices, indices, inverse = np.unique(facet_vertices, axis=0, return_index=True, return_inverse=True)
        # reduce indices
        facet_idx_list = [inverse[new_indices] for new_indices in facet_idx_list]

        parity_facets.append(facet_vertices)
        parity_indices.append(facet_idx_list)

    return parity_facets, parity_indices, parity_transforms

def get_cofacet_vertices():
    # Define all possible parities and their dimensions
    parities = np.array([
        [0,0,0],  # 0D
        [0,0,1],  # 1D
        [0,1,0],  # 1D
        [0,1,1],  # 2D
        [1,0,0],  # 1D
        [1,0,1],  # 2D
        [1,1,0],  # 2D
        [1,1,1],  # 3D
    ])
    parity_facets, parity_indices, parity_transforms = get_facet_vertices()
    parity_to_int, int_to_parity, parity_dims = get_parity_maps()
    parity_dims = np.array([0,1,1,2,1,2,2,3], dtype=np.int8)
    
    cofacet_transforms = []
    cofacet_vertices = [] # all vertices required by any cofacet of this parity
    cofacet_facets = [] # list of lists of lists where each sublist represents the
    # facets of a cofacet and each sub-sub-list represents the vertex indices
    # for that facet
    cofacet_indices = [] # list of lists where each sublist is the vertex indices
    # making up a cofacet

    for parity, dim in zip(parities, parity_dims):

        # 3D cells are trivial
        if dim == 3:
            continue
        
        # first we get the transforms to the cofacets
        transforms = get_cofacet_transforms(parity)
        cofacet_transforms.append(transforms)
        current_parity_vertices = []
        current_parity_facet_indices = []
        current_parity_cofacet_indices = []
        count = 0
        
        # for each cofacet we get its facets and the transforms required to
        # construct them
        for cofacet_trans in transforms:
            current_facet_indices = []
            current_cofacet_indices = []
            
            # get each facet of this cofacet
            trans_parity = (parity+cofacet_trans) % 2
            facet_transforms, _ = get_facet_transforms(trans_parity)
            
            # find which transform points to the current cell
            for facet_idx, (facet_trans) in enumerate(facet_transforms):
                facet_center = facet_trans + cofacet_trans
                if np.max(np.abs(facet_center)) == 0:
                    # this is the current cell
                    break
            # move corresponding facet transform to the beginning
            center_facet = facet_transforms.pop(facet_idx)
            facet_transforms.insert(0,center_facet)
            
            
            # for each facet, get the vertices required to construct it
            for facet_trans in facet_transforms:
                facet_indices = []
                # transform to this facets center
                facet_center = facet_trans + cofacet_trans
                # get facet parity
                facet_parity = (facet_center + parity) % 2
                pi, pj, pk = facet_parity
                facet_parity_idx = parity_to_int[pi, pj, pk]
                vertex_transformations = parity_facets[facet_parity_idx]
                for vertex_trans in vertex_transformations:
                    full_trans = facet_center + vertex_trans
                    current_parity_vertices.append(full_trans)
                    current_cofacet_indices.append(count)
                    facet_indices.append(count)
                    count += 1
                # add this facets indices
                current_facet_indices.append(facet_indices)
            # append this transforms vertices
            current_parity_facet_indices.append(current_facet_indices)
            current_parity_cofacet_indices.append(current_cofacet_indices)
        # get only unique transforms
        current_parity_vertices, indices, inverse  = np.unique(current_parity_vertices, axis=0, return_index=True, return_inverse=True)
        # update indices to point to correct indices
        sorted_parity_indices = []
        for cofacet in current_parity_facet_indices:
            sorted_cofacet = []
            for facet in cofacet:
                sorted_facet = [inverse[idx] for idx in facet]
                sorted_cofacet.append(sorted_facet)
            sorted_parity_indices.append(sorted_cofacet)
        current_parity_cofacet_indices = [inverse[idx] for idx in current_parity_cofacet_indices]
        
        # append cofacet info for this parity
        cofacet_vertices.append(current_parity_vertices)
        cofacet_facets.append(sorted_parity_indices)
        cofacet_indices.append(current_parity_cofacet_indices)
    return cofacet_transforms, cofacet_vertices, cofacet_facets, cofacet_indices
        

def get_highest_facets(
        data,
        ):
    # get shape
    nx, ny, nz = data.shape
    # generate maps
    # 1) maps from parity idx to the corresponding 3bit parity
    parity_to_int, int_to_parity, parity_dims = get_parity_maps()
    # 2) maps from 6 transformations to integer index equivalent
    trans_to_int, int_to_trans = get_trans_maps()
    # 3) maps from pointer information (crit, highest facet, lowest cofacet 
    # with this cell as highest)) to single uint8
    pointers_to_int, int_to_pointers = get_pointer_maps()
    # 4) maps from parity index to its vertices
    parity_facets, parity_indices, parity_transforms = get_facet_vertices()
    
    
    # create double grid to store cell pointers in
    cell_pointers = np.empty((nx*2, ny*2, nz*2), dtype=np.uint8)
    fi, fj, fk = cell_pointers.shape
    
    # iterate over each point
    for i in range(fi):
        for j in range(fj):
            for k in range(fk):
                # get parity of this point
                pi = i & 1
                pj = j & 1
                pk = k & 1
                parity = parity_to_int[pi, pj, pk]
                dim = parity_dims[parity]
                # if the dim is 0, there is no highest facet. We set the value
                # above the maximum
                if dim == 0:
                    cell_pointers[i,j,k] = 54 # equivalent to 0,6,6 pointer
                    continue
                
                # get transformations
                transforms = parity_facets[parity]
                facet_indices = parity_indices[parity]
                facet_trans = parity_transforms[parity]
                # get values at each transform
                values = np.empty(len(transforms), dtype=np.float32)
                for trans_idx, (si, sj, sk) in enumerate(transforms):
                    ni = int(((i+si)/2) % nx)
                    nj = int(((j+sj)/2) % ny)
                    nk = int(((k+sk)/2) % nz)
                    values[trans_idx] = data[ni, nj, nk]
                # get value of each face as the maximum value
                best_facets = []
                best_value = -np.inf
                
                for facet_idx, indices in enumerate(facet_indices):
                    facet_vals = values[indices]
                    facet_max = facet_vals.max()
                
                    if facet_max > best_value:
                        best_value = facet_max
                        best_facets = [facet_idx]
                    elif facet_max == best_value:
                        best_facets.append(facet_idx)
                
                if len(best_facets) > 1:
                    # We need to get the facet with the highest value. As a simple
                    # version, we can sort each facet from high to low and take
                    # the best. 
                    # NOTE: I don't think this exactly matches the proper definition
                    # as it doesn't take into account the order of subfaces. For
                    # example, two polys may have the same highest value. The
                    # second highest value may be disjoint from this value meaning
                    # a properly sorted value wouldn't be a standard sort
                    # For now I'm keeping this as it should still be deterministic
                    # and is simple and fast
                    
                    # get an initial guess as the first facet
                    best_facet = best_facets[0]
                    best_values = np.flip(np.sort(values[facet_indices[best_facet]]))
                    
                    for facet_idx in best_facets[1:]:
                        sorted_values = np.flip(np.sort(values[facet_indices[facet_idx]]))
                        for idx, val in enumerate(sorted_values):
                            if val > best_values[idx]:
                                best_facet = facet_idx
                                best_values = sorted_values
                                break
                            elif val < best_values[idx]:
                                break
                else:
                    best_facet = best_facets[0]

                
                # get the transform to the highest facet
                x, y, z = facet_trans[best_facet]
                trans_int = trans_to_int[x,y,z]
                pointer_int = pointers_to_int[0,trans_int,6]
                cell_pointers[i,j,k] = pointer_int
                
    return cell_pointers

def get_cell_pairs_alg1(
    data,
    cell_pointers,
        ):
    # get shapes
    nx, ny, nz = data.shape
    fi, fj, fk = cell_pointers.shape
    # generate maps
    # 1) maps from parity idx to the corresponding 3bit parity
    parity_to_int, int_to_parity, parity_dims = get_parity_maps()
    # 2) maps from 6 transformations to integer index equivalent
    trans_to_int, int_to_trans = get_trans_maps()
    # 3) maps from pointer information (crit, highest facet, lowest cofacet 
    # with this cell as highest)) to single uint8
    pointers_to_int, int_to_pointers = get_pointer_maps()
    # 4) maps from parity index to its vertices
    parity_facets, parity_indices, parity_transforms = get_facet_vertices()
    # 5) maps form parity index to cofacet transformations. Does not map to 
    # vertices
    parity_cofacet_trans = get_cofacet_map()
    
    # iterate over each point
    for i in range(fi):
        for j in range(fj):
            for k in range(fk):
                # get parity
                pi = i & 1
                pj = j & 1
                pk = k & 1
                parity = parity_to_int[pi, pj, pk]
                dim = parity_dims[parity]
                # if this is a 3d cell, it doesn't have cofacets and we can continue
                if dim == 3:
                    continue
                # get the current tranform to the best facet
                pointer = cell_pointers[i,j,k]
                _, facet_int, cofacet_int = int_to_pointers[pointer]
                # if we already have a cofacet pointer, this cell is paired and
                # we can continue
                if cofacet_int != 6:
                    continue
                # get cofacet transforms
                cofacet_transforms = parity_cofacet_trans[parity]
                # For each cofacet, check if it has this point as its highest
                # facet
                valid_cofacets = []
                cofacet_coords = []
                cofacet_trans_ints = []
                for trans_idx, (si, sj, sk) in enumerate(cofacet_transforms):
                    ni = (i+si)%fi
                    nj = (j+sj)%fj
                    nk = (k+sk)%fk
                    cofacet_pointer = cell_pointers[ni,nj,nk]
                    _, trans_int, _ = int_to_pointers[cofacet_pointer]
                    # get cofacet's highest face
                    ci, cj, ck = int_to_trans[trans_int]
                    if ci == -si and cj == -sj and ck == -sk:
                        valid_cofacets.append(trans_idx)
                        cofacet_coords.append((ni,nj,nk))
                        cofacet_trans_ints.append(trans_int)
                # if we have multiple possible cofacets, we need to tie break
                # by selecting the cofacet with the lowest disjoint facet. Due
                # to our parallelpiped setup, this is always the cell at twice
                # the transform
                if len(valid_cofacets) > 1:
                    sorted_values = []
                    for trans_idx in valid_cofacets:
                        # get disjoint cofacet's indices
                        si, sj, sk = cofacet_transforms[trans_idx]
                        ni = (i+si*2)%fi
                        nj = (j+sj*2)%fj
                        nk = (k+sk*2)%fk
                        # get parity
                        npi = ni & 1
                        npj = nj & 1
                        npk = nk & 1
                        parity = parity_to_int[npi, npj, npk]
                        # get vertex transformations
                        vertex_trans = parity_facets[parity]
                        # get vertex values
                        values = np.empty(len(vertex_trans), dtype=np.float64)
                        for vert_idx, (si, sj, sk) in enumerate(vertex_trans):
                            nni = int(((ni+si)/2) % nx)
                            nnj = int(((nj+sj)/2) % ny)
                            nnk = int(((nk+sk)/2) % nz)
                            values[vert_idx] = data[nni, nnj, nnk]
                        #sort values high to low
                        values = np.flip(np.sort(values))
                        sorted_values.append(values)
                    # get the lowest option
                    best_value = sorted_values[0]
                    best_cofacet = 0
                    for cofacet_idx in range(1, len(sorted_values)):
                        values = sorted_values[cofacet_idx]
                        for val_idx, val in enumerate(values):
                            if val < best_value[val_idx]:
                                best_value = values
                                best_cofacet = cofacet_idx
                                break
                            elif val > best_value[val_idx]:
                                break
                elif len(valid_cofacets) == 0:
                    best_cofacet = -1
                else:
                    best_cofacet = 0
                    
                # if we have no best_cofacet, we don't want to do anything
                if best_cofacet == -1:
                    continue
                # otherwise, we want to update both this cell and its partner
                # to point to each other
                
                # get the transform to the best cofacet
                x, y, z = cofacet_transforms[valid_cofacets[best_cofacet]]
                cofacet_int = trans_to_int[x,y,z]
                # set pointer int
                pointer_int = pointers_to_int[0,facet_int, cofacet_int]
                cell_pointers[i,j,k] = pointer_int
                # Now get the coordinates of the best cofacet
                ci, cj, ck = cofacet_coords[best_cofacet]
                facet_int = cofacet_trans_ints[best_cofacet] # same for both pointers
                pointer_int = pointers_to_int[0,facet_int,facet_int]
                cell_pointers[ci, cj, ck] = pointer_int
    return cell_pointers

def get_cell_pairs_alg2(
        data,
        cell_pointers,
        ):
    # get shapes
    nx, ny, nz = data.shape
    fi, fj, fk = cell_pointers.shape
    # generate maps
    # 1) maps from parity idx to the corresponding 3bit parity
    parity_to_int, int_to_parity, parity_dims = get_parity_maps()
    # 2) maps from 6 transformations to integer index equivalent
    trans_to_int, int_to_trans = get_trans_maps()
    # 3) maps from pointer information (crit, highest facet, lowest cofacet 
    # with this cell as highest)) to single uint8
    pointers_to_int, int_to_pointers = get_pointer_maps()
    # 4) maps from parity index to its vertices
    parity_facets, parity_indices, parity_transforms = get_facet_vertices()
    # 5) maps form parity index to cofacet transformations. Does not map to 
    # vertices
    cofacet_transforms, cofacet_vertices, cofacet_facets, cofacet_indices = get_cofacet_vertices()
    
    # iterate over each point
    for i in range(fi):
        for j in range(fj):
            for k in range(fk):
                # get the current pointers
                pointer = cell_pointers[i,j,k]
                _, facet_int, pairing_int = int_to_pointers[pointer]
                # if we have a pairing, continue
                if pairing_int != 6:
                    continue
                # get parity
                pi = i & 1
                pj = j & 1
                pk = k & 1
                parity = parity_to_int[pi, pj, pk]
                dim = parity_dims[parity]
                # 0-cells will not be paired if they failed to pair in the
                # previous step so we skip these
                # 3-cells cannot have cofacets so we skip these
                if dim == 0 or dim == 3:
                    continue
                # get cofacet transforms
                cofacet_trans = cofacet_transforms[parity]
                cofacet_verts = cofacet_vertices[parity]
                cofacet_fac = cofacet_facets[parity]
                
                # get values at each vertex
                values = np.empty(len(cofacet_verts), dtype=np.float64)
                for trans_idx, (si, sj, sk) in enumerate(cofacet_verts):
                    ni = int(((i+si)/2)%nx)
                    nj = int(((j+sj)/2)%ny)
                    nk = int(((k+sk)/2)%nz)
                    values[trans_idx] = data[ni,nj,nk]
                
                # get the value of this central point. This is always stored as
                # the first facet in each cofacet's list of facets
                center_indices = cofacet_fac[0][0]
                center_values = np.flip(np.sort(values[center_indices]))
                best_center_value = center_values[0]
                
                valid_cofacets = []
                cofacet_coords = []
                cofacet_trans_ints = []
                lowest_cofacet = np.inf
                # for each cofacet, check that it has no pairing and that this
                # cell is its second highest cofacet
                for trans_idx, ((si, sj, sk), facet_indices) in enumerate(zip(cofacet_trans, cofacet_fac)):
                    # check that this cofacet has no pairing
                    ni = ((i+si))%fi
                    nj = ((j+sj))%fj
                    nk = ((k+sk))%fk
                    copointer = cell_pointers[i,j,k]
                    _, cofacet_int, copairing_int = int_to_pointers[copointer]
                    if copairing_int == 6:
                        continue
                    cofacet_coords.append((ni,nj,nk))
                    cofacet_trans_ints.append(cofacet_int)
                    
                    # Now we count how many facets have a higher value than our
                    # central facet. We want exactly 1 for it to be the second
                    # highest
                    num_higher = 0
                    best_value = best_center_value
                    possible_higher = []
                    highest_value = best_center_value
                    for facet_indices in cofacet_fac[trans_idx][1:]:
                        facet_values = values[facet_indices]
                        facet_best = facet_values.max()
                        highest_value = max(highest_value, facet_best)
                        if facet_best > best_value:
                            num_higher += 1
                        elif facet_best == best_value:
                            possible_higher.append(facet_values)
                        if num_higher == 2:
                            break
                    
                    # if we had at least one tie and our num_higher is not 2,
                    # we need to do a more rigorous check
                    if num_higher < 2 and len(possible_higher) > 0:
                        for facet_values in possible_higher:
                            facet_values = np.flip(np.sort(facet_values))
                            for val_idx, val in enumerate(facet_values):
                                if val > center_values[val_idx]:
                                    num_higher += 1
                                    break
                                elif val < center_values[val_idx]:
                                    break
                            if num_higher == 2:
                                break
                    
                    # if we're still here, this is a possible cofacet. If it has
                    # a max value than the current best, we record it
                    if highest_value < lowest_cofacet:
                        valid_cofacets = [trans_idx]
                        lowest_cofacet = highest_value
                    elif highest_value == lowest_cofacet:
                        valid_cofacets.append(trans_idx)
                
                # now we get the best cofacet if any from our list
                if len(valid_cofacets) > 1:
                    # had more than one cofacet that tied for the highest value.
                    # we need to sort the vertex values to obtain the cofacet
                    # with the next best value
                    sorted_values = []
                    for cofacet_idx in valid_cofacets:
                        # get indices for this cofacet
                        indices = cofacet_indices[cofacet_idx]
                        covalues = np.flip(np.sort(values[indices]))
                        sorted_values.append(covalues)
                    # get the lowest option
                    best_value = sorted_values[0]
                    best_cofacet = 0
                    for cofacet_idx in range(1, len(sorted_values)):
                        values = sorted_values[cofacet_idx]
                        for val_idx, val in enumerate(values):
                            if val < best_value[val_idx]:
                                best_value = values
                                best_cofacet = cofacet_idx
                                break
                            elif val > best_value[val_idx]:
                                break
                elif len(valid_cofacets) == 0:
                    best_cofacet = -1
                else:
                    best_cofacet = 0
                    
                # if we have no best_cofacet, this is a critical point and we
                # just continue
                if best_cofacet == -1:
                    continue
                # otherwise, we want to update both this cell and its partner
                # to point to each other
                
                # get the transform to the best cofacet
                x, y, z = cofacet_transforms[valid_cofacets[best_cofacet]]
                cofacet_int = trans_to_int[x,y,z]
                # set pointer int
                pointer_int = pointers_to_int[0,facet_int, cofacet_int]
                cell_pointers[i,j,k] = pointer_int
                # Now get the coordinates of the best cofacet
                ci, cj, ck = cofacet_coords[best_cofacet]
                facet_int = cofacet_trans_ints[best_cofacet] # same for both pointers
                cofacet_int = trans_to_int[-x, -y, -z]
                pointer_int = pointers_to_int[0,facet_int,cofacet_int]
                cell_pointers[ci, cj, ck] = pointer_int
    return cell_pointers

                
def mark_critical_points(cell_pointers):
    # loop over cells and mark those that don't have a pair as critical. Store
    # coordinates and types in lists
    critical_coords = []
    critical_types = []
    fi, fj, fk = cell_pointers.shape
    # generate maps
    pointers_to_int, int_to_pointers = get_pointer_maps()
    parity_to_int, int_to_parity, parity_dims = get_parity_maps()

    # iterate over each point
    for i in range(fi):
        for j in range(fj):
            for k in range(fk):
                pointer = cell_pointers[i,j,k]
                _, facet_int, pair_int = int_to_pointers[pointer]
                if pair_int != 6:
                    # this is not a critical point and we can continue
                    continue
                # update pointer with critical flag
                cell_pointers[i,j,k] = pointers_to_int[1,facet_int,pair_int]
                # get parity
                pi = i & 1
                pj = j & 1
                pk = k & 1
                parity = parity_to_int[pi, pj, pk]
                dim = parity_dims[parity]
                # append coord and type
                critical_coords.append((i,j,k))
                critical_types.append(dim)
    # convert to arrays
    critical_coords = np.array(critical_coords, dtype=np.int64)
    critical_types = np.array(critical_types, dtype=np.int8)
    return critical_coords, critical_types
                
def get_descending_manifolds(
        cell_pointers,
        critical_coords,
        critical_types
        ):
    # NOTE:
        # I think this should actually only be used for maxima and type-1 saddles
        # as Algorithm 3 is designed specifically for saddle point traversal.
        # Algorithm 3 should be implemented separately.
    fx, fy, fz = np.array(cell_pointers.shape)
    # generate maps
    pointers_to_int, int_to_pointers = get_pointer_maps()
    parity_to_int, int_to_parity, parity_dims = get_parity_maps()
    trans_to_int, int_to_trans = get_trans_maps()
    parity_facets, parity_indices, parity_transforms = get_facet_vertices()
    # get indices of the critical points that are not minima
    crit_indices = np.where(critical_types!=0)[0]
    # create an array to store labels
    num_crits = len(critical_coords)
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if np.iinfo(dtype).max > num_crits+1:
            break
    descending_manifolds = np.full_like(cell_pointers, num_crits, dtype=dtype)
    breakpoint()
    # add crit point labels
    descending_manifolds[critical_coords[:,0],critical_coords[:,1],critical_coords[:,2]] = np.arange(num_crits)
    
    # label using breadth first search
    for sub_crit_idx in prange(len(crit_indices)):
        crit_idx = crit_indices[sub_crit_idx]
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
            