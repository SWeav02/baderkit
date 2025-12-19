# -*- coding: utf-8 -*-

from numba import njit, prange
import numpy as np
from numpy.typing import NDArray

from baderkit.core.utilities.voronoi import get_plane_dist


@njit(parallel=True, cache=True)
def get_in_partition_assignments(
        data: NDArray,
        labels: NDArray, # previous assignments
        site_indices: NDArray,
        site_transforms: NDArray,
        plane_points: NDArray,
        plane_vectors: NDArray,
        vacuum_mask: NDArray,
        min_plane_dist: float,
        num_assignments: int, #total possible regions with assigned charge/volume
        lattice_matrix: NDArray, # lattice matrix with row vectors
        ):
    nx, ny, nz = data.shape
    
    # get single grid to cartesian matrix
    grid2cart = np.empty((3,3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            grid2cart[i, j] = lattice_matrix[i, j] / data.shape[i]
    
    # create trackers for charge/volume
    charges = np.zeros(num_assignments, dtype=np.float64)
    volumes = np.zeros(num_assignments, dtype=np.float64)
    
    ###########################################################################
    # Fully inside partitioning assignments
    ###########################################################################
    # Most of our points will sit fully inside a single points partitioning. For
    # these points, we can assign all of their charge/volume immediately. We handle
    # points lying near or outside the planes later
    
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # If this voxel was assigned before this stage, or it is part
                # of the vacuum, we just continue
                if labels[i,j,k] != -1 or vacuum_mask[i,j,k]:
                    continue
                # create a tracker for which atom assignment we have
                current_site = -1
                current_transform = -1
                atom_assigned = False
                # Calculate the distance to each plane
                for plane_idx, (
                        site_idx, 
                        site_transform, 
                        plane_point, 
                        plane_vector
                        ) in enumerate(zip(
                            site_indices, 
                            site_transforms, 
                            plane_points, 
                            plane_vectors)):
                    # if our site index is new, check if we successfully assigned
                    # to our previous atom
                    if site_idx != current_site or current_transform != site_transform:
                        if atom_assigned:
                            # atom_assigned is True, we've found our assignment
                            break
                        else:
                            # otherwise, reset the assignment
                            atom_assigned = True
                            current_site = site_idx
                            current_transform = site_transform
                    # if the site/transform isn't new, check if we already found
                    # a plane where the voxel sits outside the partitioning. If so,
                    # we continue
                    elif not atom_assigned:
                        continue
                        
                    # get cartesian coords (equivalent to point @ matrix)
                    ci = i * grid2cart[0,0] + j * grid2cart[1,0] + k * grid2cart[2,0]
                    cj = i * grid2cart[0,1] + j * grid2cart[1,1] + k * grid2cart[2,1]
                    ck = i * grid2cart[0,2] + j * grid2cart[1,2] + k * grid2cart[2,2]
                    
                    # calculate the distance to the plane
                    dist = get_plane_dist(
                        point=(ci, cj, ck),
                        plane_vector=plane_vector,
                        plane_point=plane_point,
                        )
                    dist = round(dist, 12)
                    
                    # If we are under the plane and not close enough to intersect
                    # it, we can continue. "Under" corresponds to a positive value
                    if dist < min_plane_dist:
                        atom_assigned = False

                if atom_assigned:
                    # If we have an assignment, label it
                    labels[i,j,k] = current_site
                
    # assign charge/volume from whole assignments
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                label = labels[i,j,k]
                if label == -1:
                    continue
                charges[label] += data[i,j,k]
                volumes[label] += 1
    return labels, charges, volumes

@njit(parallel=True, cache=True)
def get_outside_partition_assignments(
        data,
        labels,
        vacuum_mask,
        sphere_transforms,
        transform_dists,
        transform_breaks,
        charges,
        volumes,
        max_label, # use to ignore electrides/features
        ):
    nx, ny, nz = data.shape
    
    # get unassigned points (exclude vacuum)
    unassigned_indices = np.argwhere((labels==-1) & ~vacuum_mask)
    num_unassigned = len(unassigned_indices)
    
    # create array to store site fractions. We only store up to a set number
    max_fracs = 10
    site_assignments = np.empty((num_unassigned, max_fracs), dtype=np.uint32)
    site_fractions = np.zeros((num_unassigned, max_fracs), dtype=np.float64)

    # loop over unassigned points
    for point_idx in prange(num_unassigned):
        i, j, k = unassigned_indices[point_idx]
        # create an array to track assignment counts on each atom
        site_counts = np.zeros(max_label, dtype=np.float64)
        found_sites = []
        total_counts = 0.0
        # loop over transforms, iteratively increasing the sphere radius and checking
        # for assignments
        break_idx = 0
        current_break = transform_breaks[0]
        found = False
        for trans_idx, ((si, sj, sk), dist) in enumerate(zip(sphere_transforms, transform_dists)):
            
            # check if this neighbor has an assignment
            ni = (i+si) % nx
            nj = (j+sj) % ny
            nk = (k+sk) % nz
            label = labels[ni,nj,nk]

            if label != -1 and label < max_label:
                found = True
                if site_counts[label] == 0.0:
                    found_sites.append(label)
                portion = 1/dist
                site_counts[label] += portion
                total_counts += portion
                
            # check if we've found nearby sites and checked one shell further
            if trans_idx == current_break and found:
                # condense site counts
                reduced_counts = np.empty(len(found_sites), dtype=np.float64)
                for idx, site in enumerate(found_sites):
                    reduced_counts[idx] =  site_counts[site]
                # if we have more sites than our max allowed, we cut down the
                # smallest fractions
                if len(reduced_counts) > max_fracs:
                    sorted_counts = np.argsort(reduced_counts)[:max_fracs]
                    # update total counts
                    reduced_counts = reduced_counts[sorted_counts]
                    total_counts = reduced_counts.sum()
                    found_sites = [found_sites[idx] for idx in sorted_counts]
                # update our site/frac array
                for idx, (site, frac) in enumerate(zip(found_sites, reduced_counts)):
                    site_assignments[point_idx, idx] = site
                    site_fractions[point_idx, idx] = frac / total_counts
                break
                
            # if we've hit the end of this sphere's range, check if we have an
            # assignment
            elif trans_idx == current_break:                
                # Increase next stopping point
                break_idx += 1
                if break_idx == len(transform_breaks):
                    break
                current_break = transform_breaks[break_idx]
            
                
    # Now we assign charges/volumes and labels
    tie_indices = []
    tie_charges = []
    tol = 1e-6
    approx_charges = charges.copy()
    for point_idx, ((i,j,k), sites, fracs) in enumerate(zip(unassigned_indices, site_assignments, site_fractions)):
        charge = data[i,j,k]
        
        # track the best fraction
        best_frac = 0.0
        best_site = -1
        tie = False
        
        for site, frac in zip(sites, fracs):
            # stop if we have no fraction left
            if frac == 0.0:
                break
            # check if this is a larger fraction
            if frac > best_frac + tol:
                best_frac = frac
                best_site = site
                tie = False
            elif frac > best_frac - tol:
                tie = True
        
        # if there isn't a tie, we can go ahead and assign the label, charge, and
        # volume. Otherwise, we will assign them later
        if not tie:
            labels[i,j,k] = best_site
            approx_charges[best_site] += charge
            for site, frac in zip(sites, fracs):
                if frac == 0.0:
                    break
                # assign charge/volume
                charges[site] += charge*frac
                volumes[site] += frac
        
        # if there is a tie, we need to note it for later
        if tie:
            tie_indices.append(point_idx)
            tie_charges.append(charge)
            
    # to ensure consistent assignments for ties, we sort them by their charge value
    # and assign highest to lowest
    
    sorted_indices = np.flip(np.argsort(np.array(tie_charges)))
    for sorted_idx in sorted_indices:
        # get charge and corresponding unassigned point index
        charge = tie_charges[sorted_idx]
        point_idx = tie_indices[sorted_idx]
        # assign charges for this point. We need to do this before approximating
        # the best improvement
        for site, frac in zip(sites, fracs):
            if frac == 0.0:
                break
            # assign charge/volume
            charges[site] += charge*frac
            volumes[site] += frac
        
        # get indices and possible sites/fracs
        i, j, k = unassigned_indices[point_idx]
        sites = site_assignments[point_idx]
        fracs = site_fractions[point_idx]
        # Assign to the basin where the added charge will most improve the approximate charge
        best_improvement = -1.0
        best_frac = 0.0
        best_site = -1
        for site, frac in zip(sites, fracs):
            if frac == 0.0:
                break
            
            if frac < best_frac - tol:
                continue
            # calculate the difference from the current charge before and
            # after adding this point
            diff = approx_charges[site] - charges[site]
            before = abs(diff)
            after = abs(diff + charge)
            improvement = (before - after) / charges[site]
            if improvement > best_improvement:
                best_improvement = improvement
                best_site = site
        labels[i,j,k] = best_site
        approx_charges[best_site] += charge
    
    return labels, charges, volumes

@njit(cache=True)
def get_badelf_assignments(
        data: NDArray,
        labels: NDArray, # previous assignments
        site_indices: NDArray,
        site_transforms: NDArray,
        plane_points: NDArray,
        plane_vectors: NDArray,
        vacuum_mask: NDArray,
        min_plane_dist: float,
        num_assignments: int, #total possible regions with assigned charge/volume
        lattice_matrix: NDArray, # lattice matrix with row vectors
        sphere_transforms,
        transform_dists,
        transform_breaks,
        max_label, # use to ignore electrides/features
        ):
    
    # get assignments for voxels fully inside partitioning
    labels, charges, volumes = get_in_partition_assignments(
        data=data,
        labels=labels,
        site_indices=site_indices,
        site_transforms=site_transforms,
        plane_points=plane_points,
        plane_vectors=plane_vectors,
        vacuum_mask=vacuum_mask,
        min_plane_dist=min_plane_dist,
        num_assignments=num_assignments, #total possible regions with assigned charge/volume
        lattice_matrix=lattice_matrix,
        )
    
    # get assignments for voxels on or outside the partitioning
    labels, charges, volumes = get_outside_partition_assignments(
        data,
        labels,
        vacuum_mask,
        sphere_transforms,
        transform_dists,
        transform_breaks,
        charges,
        volumes,
        max_label, # use to ignore electrides/features
        )
    
    return labels, charges, volumes

    