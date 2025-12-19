# -*- coding: utf-8 -*-
"""
This file contains code from when I was testing splitting charge/volume exactly
at planes. I ended up abandoning this for now, as we can't get partial voxel
assignments for the electride sites when using badelf anyways, and because handling
voxels that lie outside the plane is rather difficult.
"""

# @njit
# def volume_fraction(
#         d, 
#         normal, 
#         dx,
#         dy,
#         dz, 
#         voxel_volume,
#         ):
#     """
#     Calculates the exact volume of a voxel split by a plane at a set distance.

#     Parameters
#     ----------
#     d : float
#         distance to plane in voxel coordinates
#     normal : float
#         normal vector of the plane
#     dx : float
#         half of the width of the voxel along the x axis
#     dy : float
#         half of the width of the voxel along the y axis
#     dz : float
#         half of the width of the voxel along the z axis
#     voxel_volume : float
#         the volume of the voxel for normalization

#     Returns
#     -------
#     vol : float
#         the fractional volume split by the plane

#     """

#     vol = 0.0

#     for xi1 in (-dx, dx):
#         for xi2 in (-dy, dy):

#             rhs = -(d + normal[0]*xi1 + normal[1]*xi2)

#             if normal[2] == 0.0:
#                 continue

#             xi3_cut = rhs / normal[2]

#             lo = max(-dz, xi3_cut)
#             hi = dz

#             if lo < hi:
#                 vol += hi - lo

#     return vol / voxel_volume

    
# @njit
# def precompute_piecewise_volume_fraction(
#         normal: NDArray, 
#         grid_shape: NDArray,
#         ):
#     """
#     Constructs piecewise polynomials mapping the distance from a voxel center to
#     a plane to the volume lying on each side of the plane.

#     Parameters
#     ----------
#     normal : NDArray
#         The normal vector of the plane in fractional coordinates.
#     grid_shape : NDArray
#         The number of grid points along each axis of the lattice.

#     Returns
#     -------
#     breaks : NDArray
#         The points where polynomial segments start and stop.
#     coeffs : NDArray
#         The polynomial coefficients along each segment.

#     """
#     # calculate voxel volume in fractional space
#     voxel_x = 1.0 / grid_shape[0]
#     voxel_y = 1.0 / grid_shape[1]
#     voxel_z = 1.0 / grid_shape[2]
#     voxel_volume = voxel_x*voxel_y*voxel_z

#     # vertices of voxel in fractional space
#     vertices = np.empty((8,3), dtype=np.float64)
#     x_range = voxel_x / 2.0
#     y_range = voxel_y / 2.0
#     z_range = voxel_z / 2.0
#     vertex_index = 0
#     for x in (-x_range, x_range):
#         for y in (-y_range, y_range):
#             for z in (-z_range, z_range):
#                 vertices[vertex_index] = (x,y,z)
#                 vertex_index += 1

#     # project lattice vertices onto normal vector
#     vertex_d = vertices @ normal
#     vertex_d = np.round(vertex_d, 12) # round for numerical stability

#     # get breakpoints along normal vector (vertex locations)
#     breaks = np.unique(-vertex_d)
#     breaks.sort()

#     # create array to store polynomial coefficients
#     coeffs = np.zeros((len(breaks)-1, 4), dtype=np.float64)
#     # fit fractional volume to cubic polynomial between each breakpoint
#     for i in range(len(breaks)-1):
#         # get start/end values of this segment
#         d0, d1 = breaks[i], breaks[i+1]
#         # get several points along this segment
#         ds = np.array([d0,
#                        (2*d0 + d1)/3,
#                        (d0 + 2*d1)/3,
#                        d1])
#         fs = np.empty(len(ds), dtype=np.float64)
#         # calculate the volume fraction at each d
#         for didx, d in enumerate(ds):
#             fs[didx] = volume_fraction(
#                 d=d, 
#                 normal=normal, 
#                 dx=x_range,
#                 dy=y_range,
#                 dz=z_range,
#                 voxel_volume=voxel_volume,
#                 )

#         # Solve Vandermonde for cubic coefficients
#         V = np.vstack((ds**3, ds**2, ds, np.ones_like(ds))).T
#         a3, a2, a1, a0 = np.linalg.solve(V, fs)

#         coeffs[i] = (a3, a2, a1, a0)

#     return breaks, coeffs

# @property
# def plane_distance_polynomials(self):
#     if self._plane_distance_polynomials is None:
#         # get unique normals
#         _,_,_, plane_vectors, _ = self.partitioning
#         plane_vectors = plane_vectors.round(12)
#         unique_vectors, index, inverse = np.unique(plane_vectors, return_inverse=True, return_index=True, axis=0
#     )
#         # TODO: also remove repeats where the normals are exactly opposite
#         # each other. The actual distance to volume function should be identical.
#         # This would cut the required calculations in half

#         # for each unique vector, fit distance to volume
#         unique_dbreaks = []
#         unique_polynomials = []
#         for vector in unique_vectors:
#             # get break points and polynomials for each plane such that we can calculate
#             # partial volume based exclusively on distance from the plane
#             d_breaks, polynomials = precompute_piecewise_volume_fraction(
#                 normal=vector,
#                 grid_shape=self.reference_grid.shape,
#                 )
#             unique_dbreaks.append(d_breaks)
#             unique_polynomials.append(polynomials)
            
#         # get all dbreaks/polynomials for non-unique planes
#         all_d_breaks = [unique_dbreaks[i] for i in inverse]
#         all_polynomials = [unique_polynomials[i] for i in inverse]

#         self._plane_distance_polynomials = all_d_breaks, all_polynomials
#     return self._plane_distance_polynomials

# @njit
# def fast_volume_fraction(d, breaks, coeffs):
#     """
#     Calculates the volume fraction on each side of a plane based on the distance 
#     to the plane and precomputed break points and coefficients. Units should be
#     in fractional coordinates

#     """
#     # get the number of segments
#     n = len(coeffs) -1

#     # check if d is less than or greater than the possible values.
#     if d <= breaks[0]:
#         return 0.0

#     if d >= breaks[n]:
#         return 1.0

#     # If we didn't find a value, find the segment this d belongs to
#     for i in range(len(coeffs)):
#         if breaks[i] <= d <= breaks[i+1]:
#             a3, a2, a1, a0 = coeffs[i]
#             return ((a3*d + a2)*d + a1)*d + a0

# def get_on_partition_assignments(
#     data: NDArray,
#     labels: NDArray, # previous assignments
#     site_indices: NDArray,
#     site_transforms: NDArray,
#     neigh_indices: NDArray,
#     plane_points: NDArray,
#     plane_vectors: NDArray,
#     vacuum_mask: NDArray,
#     d_breaks: list,
#     polynomials: list,
#     min_plane_dist: float,
#     num_assignments: int, #total possible regions with assigned charge/volume
#     neighbor_transforms: NDArray,
#     neighbor_dists: NDArray,
#     unassigned_mask: NDArray,
#         ):
#     ###########################################################################
#     # Edge assignments
#     ###########################################################################
#     nx, ny, nz = data.shape
    
#     labeled_idx = np.iinfo(unassigned_mask.dtype).max
#     edge_idx = labeled_idx - 1
#     outside_idx = labeled_idx - 2

#     # We want to track the portion of each voxel that is assigned to multiple basins
#     # so that we can calculate the portions for later voxels as well.
#     # First, we find create arrays to track partial fractions. This includes points
#     # that are not edges
#     partial_voxels = np.argwhere(unassigned_mask != labeled_idx)
#     n_partial = len(partial_voxels)
    
#     # create arrays to store the assignments and fraction of the voxels going to them
#     partial_assignments = np.full((n_partial, len(neighbor_transforms)), labeled_idx, dtype=np.uint32)
#     partial_fractions = np.full((n_partial, len(neighbor_transforms)), 0, dtype=np.float64)
#     partial_dists = np.full((n_partial, len(neighbor_transforms)), -1, dtype=np.float64)

#     # create arrays to track the nearest atom/plane for each point
#     nearest_atom = np.empty(n_partial, dtype=np.uint32)
#     nearest_plane = np.empty(n_partial, dtype=np.uint32)
#     plane_dists = np.empty(n_partial, dtype=np.float32)
    
#     # Now we iterate over each partially assigned point and calculate how much
#     # volume goes to each splitting neighbor
#     for partial_idx in prange(n_partial):
#         i, j, k = partial_voxels[partial_idx]
#         # skip points that are not edges
#         if unassigned_mask[i,j,k] < outside_idx:
#             continue
#         # mark neighbors sit outside the partitioning as future edge points
#         for si, sj, sk in neighbor_transforms:
#             ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#             if unassigned_mask[ni,nj,nk] == outside_idx:
#                 unassigned_mask[ni,nj,nk] = edge_idx

#         # get fractional coordinates
#         fi = i / nx
#         fj = j / ny
#         fk = k / nz
        
#         # get this points site and transform (stored in the labels and unassigend_mask
#         # arrays) to determine which planes we need to use
#         label = labels[i,j,k]
#         transform = unassigned_mask[i,j,k]
        
#         # mark the the index associated with this point (overwrites transform)
#         unassigned_mask[i,j,k] = partial_idx
        
#         # Now we need to collect the planes that intersect this point. First we
#         # find the atoms neighboring the assigned site, then we check for combinations
#         # of the found atoms.
#         neighboring_atoms = []
#         pairs = []
#         fractions = []
#         lowest_dist = 1.0e300
#         closest_plane = -1
#         current_site = -1
#         current_plane = -1
#         for (
#                 site_idx, 
#                 site_transform, 
#                 neigh_idx,
#                 plane_point, 
#                 plane_vector,
#                 breaks,
#                 coeffs,
#                 ) in zip(
#                     site_indices, 
#                     site_transforms, 
#                     neigh_indices,
#                     plane_points, 
#                     plane_vectors,
#                     d_breaks,
#                     polynomials,
#                     ):
#             # track site
#             if site_idx != current_site:
#                 current_site = site_idx
#                 current_plane = 0
#             else:
#                 current_plane += 1
            
#             # skip planes that do not belong to our atom
#             if site_idx != label or site_transform != transform:
#                 continue
#             # calculate the portion of the voxel sitting on this side of the plane
#             dist = get_plane_dist(
#                 point=(fi, fj, fk),
#                 plane_vector=plane_vector,
#                 plane_point=plane_point,
#                 )
#             # track the closest plane to this point
#             if abs(dist) < lowest_dist:
#                 lowest_dist = abs(dist)
#                 closest_plane = current_plane
            
#             fraction = fast_volume_fraction(
#                 d=dist, 
#                 breaks=breaks, 
#                 coeffs=coeffs,
#                 )
#             # if the full fraction is contained, this is not an important plane
#             if fraction == 1.0:
#                 continue
#             # we also need to check that this neighbor also has this plane as
#             # part of it's partitioning. If not, part of this voxel likely
#             # sits outside the voronoi cells
#             neigh_found = False
#             for (
#                     neigh_idx1, 
#                     site_idx1,
#                     plane_vector1,
#                     ) in zip(
#                         site_indices, 
#                         neigh_indices,
#                         plane_vectors
#                         ):
#                 # skip if the sites do not match
#                 if site_idx1 != site_idx or neigh_idx1 != neigh_idx:
#                     continue
#                 # skip if the plane vectors are not reverses of each other
#                 if not np.allclose(plane_vector, -plane_vector1):
#                     continue
#                 # otherwise, we've found the equivalent plane and we can stop
#                 neigh_found = True
#                 break
#             # if we didn't find the corresponding neighbor plane, we don't count
#             # that neighbor and continue
#             if not neigh_found:
#                 continue
            
#             # otherwise, this neighbor is important
#             neighboring_atoms.append(neigh_idx)
#             pairs.append((site_idx, neigh_idx))
#             fractions.append(fraction)
        
#         # set closest plane
#         nearest_atom[partial_idx] = label
#         nearest_plane[partial_idx] = closest_plane
#         plane_dists[partial_idx] = lowest_dist
        
#         # Now we need to check between the important neighbors to see if they share
#         # a partitioning surface passing through the voxel
#         for nidx in range(len(neighboring_atoms)-1):
#             neigh = neighboring_atoms[nidx]
#             for (
#                     site_idx, 
#                     site_transform, 
#                     neigh_idx,
#                     plane_point, 
#                     plane_vector,
#                     breaks,
#                     coeffs,
#                     ) in zip(
#                         site_indices, 
#                         site_transforms, 
#                         neigh_indices,
#                         plane_points, 
#                         plane_vectors,
#                         d_breaks,
#                         polynomials,
#                         ):
#                 # skip planes that don't belong to this site
#                 if site_idx != neigh:
#                     continue
#                 # skip planes that don't belong to another neighbor
#                 if not neigh_idx in neighboring_atoms[nidx:]:
#                     continue
#                 # if this plane does belong to another neighbor, calculate the
#                 # fraction belonging to each
#                 dist = get_plane_dist(
#                     point=(fi, fj, fk),
#                     plane_vector=plane_vector,
#                     plane_point=plane_point,
#                     )
#                 fraction = fast_volume_fraction(
#                     d=dist, 
#                     breaks=breaks, 
#                     coeffs=coeffs,
#                     )
#                 # ignore fractions that do not intersect the voxel
#                 if fraction == 1.0 or fraction == 0.0:
#                     continue
#                 pairs.append((neigh, neigh_idx))
#                 fractions.append(fraction)
#                 # TODO: Do I need to check all planes in case a neighbor has its
#                 # own neighbor that the dominant site does not have?
        
#         # If we found no neighbors (e.g. we sit on an edge of a region with no assignment)
#         # we just assign all charge to the current assignment
#         partial_assignments[partial_idx][0] = label
#         partial_fractions[partial_idx][0] = 1.0
        
#         # Now that we can calculate the portion of the voxel going to each competing
#         # neighbor. We can do this but starting with the full voxel and subtracting
#         # off the portions belonging to other sites. The remaining volume belongs
#         # to that neighbor
#         # NOTE: We will not capture volume sitting outside all of the sites. In
#         # these cases, we will assign that volume to the dominant site
#         neighbor_fractions = []
#         for neigh in neighboring_atoms:
#             volume = 1.0
#             # get fractions belonging to other neighbors
#             for pair, fraction in zip(pairs, fractions):
#                 if pair[0] == neigh:
#                     volume -= 1 - fraction
#                 elif pair[1] == neigh:
#                     volume -= fraction
#             neighbor_fractions.append(volume)
#         # Now we assign charge and volume
#         remaining_fraction = 1.0
#         neigh_num = 0
#         for neigh, fraction in zip(neighboring_atoms, neighbor_fractions):
#             remaining_fraction -= fraction
#             # add to our partial assignments
#             partial_assignments[partial_idx][neigh_num] = neigh
#             partial_fractions[partial_idx][neigh_num] = fraction
#             neigh_num += 1
#         # assign largest fraction
#         partial_assignments[partial_idx][neigh_num] = label
#         partial_fractions[partial_idx][neigh_num] = remaining_fraction
    
# def get_badelf_assignments(
#     data: NDArray,
#     labels: NDArray, # previous assignments
#     site_indices: NDArray,
#     site_transforms: NDArray,
#     neigh_indices: NDArray,
#     plane_points: NDArray,
#     plane_vectors: NDArray,
#     vacuum_mask: NDArray,
#     d_breaks: list,
#     polynomials: list,
#     min_plane_dist: float,
#     num_assignments: int, #total possible regions with assigned charge/volume
#     neighbor_transforms: NDArray,
#     neighbor_dists: NDArray,
#         ):
#     # first we create a scrap array that will track information about each point
#     labeled_idx = np.iinfo(np.uint32).max
#     edge_idx = labeled_idx - 1
#     outside_idx = labeled_idx - 2
#     outside_idx1 = labeled_idx - 3
#     unassigned_mask = np.full(data.shape, labeled_idx, dtype=np.uint32)
    
#     labels, charges, volumes, unassigned_mask = get_in_partition_assignments(
#         data=data,
#         labels=labels,
#         site_indices=site_indices,
#         site_transforms=site_transforms,
#         neigh_indices=neigh_indices,
#         plane_points=plane_points,
#         plane_vectors=plane_vectors,
#         vacuum_mask=vacuum_mask,
#         min_plane_dist=min_plane_dist,
#         num_assignments=num_assignments, #total possible regions with assigned charge/volume
#         unassigned_mask=unassigned_mask,
#         )
#     breakpoint()
    

#     ###########################################################################
#     # Outside Partitioning Assignments
#     ###########################################################################
#     # The final step is assigning any points that sit outside the partitioning.
#     # To do so, we will take inspiration from the 'weight' Bader algorithm. Starting
#     # from the points closest to the partitioning surfaces, we will calculate the
#     # portion flowing to adjacent points, and from there calculate the portion
#     # flowing to each basin.
    
#     # First we need to calculate the distance of each remaining voxel to the nearest
#     # partitioning plane. This is actually somewhat complicated, as an unrelated
#     # plane from across the cell may travel close by. Instead what we do is iterate
#     # over the points closest to the edge of each partitioning plane and track which
#     # atom it belongs to. We then check each unlabeled points neighbors, find what
#     # atoms they belong to, and calculate the minimum distance to one of these atoms
#     # planes.
#     not_all_scanned = True
#     while not_all_scanned:
#         not_all_scanned = False
#         for partial_idx in prange(n_partial):
#             i, j, k = partial_voxels[partial_idx]
#             fi = i / nx
#             fj = j / ny
#             fk = k / nz
#             # skip points that are assigned
#             if labels[i,j,k] != -1:
#                 continue
            
#             # note this point has been checked
#             unassigned_mask[i,j,k] = outside_idx1
            
#             # collect possible neighboring sites/planes
#             possible_sites = []
#             possible_planes = []
#             # mark neighbors sit outside the partitioning as future edge points
#             for si, sj, sk in neighbor_transforms:
#                 ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#                 mask_value = unassigned_mask[ni,nj,nk]
#                 if mask_value == outside_idx:
#                     unassigned_mask[ni,nj,nk] = edge_idx
#                     not_all_scanned = True
#                     continue
#                 if mask_value == edge_idx or mask_value == labeled_idx or mask_value == outside_idx1:
#                     continue
#                 # if we're still here, this is a previous edge point.
#                 possible_sites.append(nearest_atom[mask_value])
#                 possible_planes.append(nearest_plane[mask_value])
            
#             # get the closest site/plane
#             lowest_dist = 1.0e300
#             closest_site = -1
#             closest_plane = -1
#             for site, plane_idx in zip(possible_sites, possible_planes):
#                 current_site = -1
#                 current_plane = -1
#                 for (
#                         site_idx, 
#                         site_transform, 
#                         neigh_idx,
#                         plane_point, 
#                         plane_vector,
#                         breaks,
#                         coeffs,
#                         ) in zip(
#                             site_indices, 
#                             site_transforms, 
#                             neigh_indices,
#                             plane_points, 
#                             plane_vectors,
#                             d_breaks,
#                             polynomials,
#                             ):
#                     # update site/plane counters
#                     if current_site != site_idx:
#                         current_site = site_idx
#                         current_plane = 0
#                     else:
#                         current_plane += 1
#                     # skip planes that don't belong to this site
#                     if site_idx != site:
#                         continue
#                     # skip planes with a different index
#                     if current_plane != plane_idx:
#                         continue
#                     # Calculate the distance
#                     dist = get_plane_dist(
#                         point=(fi, fj, fk),
#                         plane_vector=plane_vector,
#                         plane_point=plane_point,
#                         )
#                     if abs(dist) < lowest_dist:
#                         lowest_dist = abs(dist)
#                         closest_site = site_idx
#                         closest_plane = current_plane
#             # assign this distance
#             nearest_atom[partial_idx] = closest_site
#             nearest_plane[partial_idx] = closest_plane
#             plane_dists[partial_idx] = lowest_dist
#     # We now have distances to the closest plane for each unassigned point. We
#     # sort from lowest to highest
#     sorted_indices = np.argsort(plane_dists)
#     # now we assign each point partially to the surrounding voxels
#     site_scratch = np.empty(num_assignments, dtype=np.float64)
#     for partial_idx in sorted_indices:
#         # skip edges (already have labels)
#         i, j, k = partial_voxels[partial_idx]
#         if labels[i,j,k] != -1:
#             continue
#         # reset scratch
#         site_scratch[:] = 0.0
#         # assign this points index
#         unassigned_mask[i,j,k] = partial_idx
#         partial_sites = []
#         # search for surrounding points with assignments
#         for neigh_dist, (si, sj, sk) in zip(neighbor_dists, neighbor_transforms):
#             ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#             # skip unassigned points
#             if labels[ni,nj,nk] == -1:
#                 continue
            
#             neigh_part_idx = unassigned_mask[i,j,k]
#             if neigh_part_idx == labeled_idx:
#                 # this is not a partially assigned point. We assign the full
#                 # volume
#                 site = labels[ni,nj,nk]
#                 if site_scratch[site] == 0.0:
#                     partial_sites.append(site)
#                 site_scratch[site] += 1/neigh_dist
#                 continue
            
#             # get partial assignments
#             for site, frac in zip(partial_assignments[neigh_part_idx], neigh_partial_fracs = partial_fractions[neigh_part_idx]):
#                 if frac == 0.0:
#                     break
#                 if site_scratch[site] == 0.0:
#                     partial_sites.append(site)
#                 site_scratch[site] += frac/neigh_dist
#         # consolidate sites/fracs to a small array
#         partial_sites_array = np.empty(len(partial_sites), dtype=np.int64)
#         partial_fracs_array = np.empty(len(partial_sites), dtype=np.float64)
#         for site_idx, site in enumerate(partial_sites):
#             frac = site_scratch[site]
#             partial_sites_array[site_idx] = site
#             partial_fracs_array[site_idx] = frac
#         # sort highest to lowest and clip to the highest number of transformations
#         sorted_fracs = np.flip(np.sort(partial_fracs_array))[:len(neighbor_transforms)]
#         partial_fracs_array = partial_fracs_array[sorted_fracs]
#         partial_sites_array = partial_sites_array[sorted_fracs]
#         # set assighments
#         partial_assignments[partial_idx] = partial_sites_array
#         partial_fractions[partial_idx] = partial_fracs_array / partial_fracs_array.sum()
        
#     ###########################################################################
#     # Integrate charges and volumes
#     ###########################################################################
#     approx_charges = charges.copy()
#     # Now loop over each split voxel and assign charges, volumes, and labels
#     for partial_idx in sorted_indices:
#         i, j, k = partial_voxels[partial_idx]
#         charge = charges[partial_idx]
#         sites = partial_assignments[partial_idx]
#         fracs = partial_fractions[partial_idx]
        
#         best_label = -1
#         best_frac = 0.0
#         tied_labels = False
#         tol = 1e-6  # for floating point errors

#         for label, frac in zip(sites, fracs):
#             if frac == 0.0:
#                 break
#             charges[label] += charge * frac
#             volumes[label] += frac
    
#             if frac > best_frac + tol:  # greater than with a tolerance
#                 best_label = label
#                 best_frac = frac
#                 tied_labels = False
#             elif frac > best_frac - tol:  # equal to with a tolerance
#                 tied_labels == True
    
#         # Now we want to assign our label. If there wasn't a tie in our labels,
#         # we assign to highest weight
#         if not tied_labels:
#             labels[i,j,k] = best_label
#             approx_charges[best_label] += charge
#         else:
#             # we have a tie. We assign to the basin where the added charge will
#             # most improve the approximate charge
#             best_improvement = -1.0
#             for label, frac in zip(sites, fracs):
#                 if frac == 0.0:
#                     break
#                 if frac < best_frac - tol:
#                     continue
#                 # calculate the difference from the current charge before and
#                 # after adding this point
#                 diff = approx_charges[label] - charges[label]
#                 before = abs(diff)
#                 after = abs(diff + charge)
#                 improvement = (before - after) / charges[label]
#                 if improvement > best_improvement:
#                     best_improvement = improvement
#                     best_label = label
#             labels[i,j,k] = best_label
#             approx_charges[best_label] += charge
    
#     return labels, charges, volumes