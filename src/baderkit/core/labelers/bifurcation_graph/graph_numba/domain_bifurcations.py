# -*- coding: utf-8 -*-

import numpy as np
from numba import njit
from numpy.typing import NDArray

from baderkit.core.utilities.basic import wrap_point
from baderkit.core.utilities.union_find import compress_roots, find_root, union

from .bifurcation_base import get_connected_groups, get_domain_dimensionality


@njit(cache=True)
def find_domain_connections(
    basin_labels: NDArray[np.int64],
    data: NDArray[np.float64],
    bif_mask: NDArray[np.bool_],
    edge_mask: NDArray[np.bool_],
    num_basins: np.int64,
    neighbor_transforms: NDArray[np.int64],
):
    """
    Finds values where domains potentially form new connections. Success depends
    entirely upon the provided bifurcation/saddle point mask, and will generally
    include some false connections.
    """
    nx, ny, nz = basin_labels.shape

    # BUGFIX
    # create a tracker for which basins we find a connection for. We add this
    # because very small basins (close to the size of a voxel) will not have
    # bifurcations found by our union method. In those cases we need to loop
    # over the edges again to find potential connection points.
    found_mask = np.zeros(num_basins, dtype=np.bool_)

    lower_points = []
    upper_points = []
    conn_values = []
    # Loop over our possible bifurcations and find connections to other basins
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # skip points that are not edge maxima
                if not bif_mask[i, j, k]:
                    continue

                # get the label at this point
                label = basin_labels[i, j, k]

                if label == -1:
                    # shouldn't happen unless bader package isn't working
                    continue
                value = data[i, j, k]

                # collect labels of neighbors
                neigh_labels = []
                neigh_vals = []
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    # skip points in the same basin
                    neigh_label = basin_labels[ii, jj, kk]
                    if neigh_label == label or neigh_label == -1:
                        continue
                    neigh_labels.append(neigh_label)
                    neigh_vals.append(
                        min(value, data[ii, jj, kk])
                    )  # cap at this points value
                # get unique labels
                unique_neigh_labels = []
                for neigh_label in neigh_labels:
                    if not neigh_label in unique_neigh_labels:
                        unique_neigh_labels.append(neigh_label)
                # get highest point for each connected basin
                for unique_label in unique_neigh_labels:
                    best_val = -1.0e300
                    for neigh_label, neigh_val in zip(neigh_labels, neigh_vals):
                        if neigh_label != unique_label:
                            continue
                        if neigh_val > best_val:
                            # cap at this points value if the neighbor is higher
                            best_val = neigh_val
                    # add connection
                    lower_points.append(min(unique_label, label))
                    upper_points.append(max(unique_label, label))
                    conn_values.append(best_val)
                    # make sure both labels are marked
                    found_mask[unique_label] = True
                    found_mask[label] = True

    # get the basins that don't have connections
    missing_basins = np.where(~found_mask)[0]
    if len(missing_basins) > 0:
        # create an array to store neighbor results
        new_connections = np.zeros((len(missing_basins), num_basins), dtype=np.float64)
        # create a reverse index array
        basin_map = np.empty(num_basins, dtype=np.uint16)
        basin_map[missing_basins] = np.arange(len(missing_basins))

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # skip points that are not edge maxima
                    if not edge_mask[i, j, k]:
                        continue
                    # get the label at this point
                    label = basin_labels[i, j, k]
                    missing_idx = basin_map[label]
                    # skip labels we already have a results for
                    if found_mask[label] or label == -1:
                        continue
                    # get value
                    value = data[i, j, k]

                    # iterate over neighbors
                    for si, sj, sk in neighbor_transforms:
                        # wrap points
                        ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                        # skip points in the same basin
                        neigh_label = basin_labels[ii, jj, kk]
                        if neigh_label == label or neigh_label == -1:
                            continue
                        # get the value that these two points connect at
                        conn_val = min(data[ii, jj, kk], value)
                        if conn_val > new_connections[missing_idx, neigh_label]:
                            new_connections[missing_idx, neigh_label] = conn_val
        # get the new connections for each basin and add them to our lists
        for missing_idx, connections in enumerate(new_connections):
            label = missing_basins[missing_idx]
            for neigh_label, conn_val in enumerate(connections):
                if conn_val == 0:
                    continue
                # add connection
                lower_points.append(min(neigh_label, label))
                upper_points.append(max(neigh_label, label))
                conn_values.append(conn_val)

    return (
        np.array(lower_points, dtype=np.int64),
        np.array(upper_points, dtype=np.int64),
        np.array(conn_values, dtype=np.float64),
    )


@njit(cache=True)
def find_domain_bifurcations(
    connection_pairs,
    connection_values,
    basin_maxima_grid,
    basin_maxima_ref_values,
    data,
    neighbor_transforms,
):
    """
    Finds the values at which changes in domain connections occur. Assumes the
    connection values list includes all bifurcation values and some extra.
    """
    nx, ny, nz = data.shape
    ny_nz = ny * nz
    N = nx * ny * nz

    num_basins = len(basin_maxima_grid)

    # get all possible elf values and flip to move from high to low
    possible_values = np.flip(np.unique(connection_values))

    ###########################################################################
    # Dimensionality Setup
    ###########################################################################
    new_solid = np.zeros((nx, ny, nz), dtype=np.bool_)  # initial empty solid
    root_mask = np.zeros(N, dtype=np.bool_)  # nothing is a root yet
    roots = np.empty(0, dtype=np.int64)  # empty 1D array as nothing is a root yet
    cycles = [
        [np.array((-1, -1, -1), dtype=np.float64)]
    ]  # empty as we have no cycles yet
    cycles = cycles[1:]
    parent = np.arange(N, dtype=np.uint32)  # All voxels point to themselves
    offset_x = np.zeros(N, dtype=np.int8)  # no offsets to start
    offset_y = np.zeros(N, dtype=np.int8)
    offset_z = np.zeros(N, dtype=np.int8)
    size = np.ones(N, dtype=np.uint16)  # no size to start

    ###########################################################################
    # Basin Connections Setup
    ###########################################################################
    # create lists to store each bifurcation
    bifurcation_values = []
    bifurcation_domains = []
    bifurcation_domain_indices = []
    bifurcation_domain_dims = []

    # create an array representing which basins are connected to one another
    # at a given value
    basin_connections = np.full(num_basins, -1, dtype=np.int64)
    # and an array pointing each basin group to its corresponding list index
    # at a given value
    domain_group_indices = basin_connections.copy()

    # create an empty set of domain groups representing the groups above the
    # highest value.
    domain_groups = [[-1]]  # -1 is just for typing as numba dislikes the empty list
    domain_ids = [-1]
    domain_groups = domain_groups[1:]
    domain_ids = domain_ids[1:]
    # do the same for dimensionalities of the domains
    new_dimensionalities = [-1]
    new_dimensionalities = new_dimensionalities[1:]
    # create a counter for the total number of unique domains
    unique_domains = 0

    ###########################################################################
    # Begin loop
    ###########################################################################
    # loop over elf values from high to low
    previous_value = 1.0e12  # make unreasonably large
    for val_idx, value in enumerate(possible_values):
        #######################################################################
        # Find groups of connected basins
        #######################################################################
        # get the new connections that exist at this value
        connection_indices = np.where(
            (connection_values >= value) & (connection_values < previous_value)
        )
        current_connections = connection_pairs[connection_indices]
        # reset our connections and groups.
        # NOTE: connections don't need to be reset as they will never disappear
        # as we loop downwards in ELF value
        domain_group_indices[:] = -1

        # Loop over basin connections
        for i, j in current_connections:
            # ensure both basins have a connection
            for basin_idx in (i, j):
                if basin_connections[basin_idx] == -1:
                    basin_connections[basin_idx] = basin_idx

            # get the roots of each connection
            lower_root = find_root(basin_connections, i)
            upper_root = find_root(basin_connections, j)

            # if we have the same root, we dont have a new connection
            if lower_root == upper_root:
                continue
            # otherwise we note the new connection
            union(basin_connections, i, j)

        # reduce our connections
        basin_connections = compress_roots(basin_connections)

        # copy our previous groups
        previous_groups = [i for i in domain_groups]
        # create lists to store domains basin groups
        domain_groups = []
        domain_points = []
        num_domains = 0

        # Now loop over our basin connections and get our groups
        for basin_idx, basin_group in enumerate(basin_connections):
            # skip if this basin isn't assigned
            if basin_group == -1:
                continue

            # check if this group exists yet
            if domain_group_indices[basin_group] == -1:
                # create a new domain
                domain_groups.append([basin_idx])
                domain_group_indices[basin_group] = num_domains
                num_domains += 1
                # add a point sitting in this domain
                domain_points.append(basin_maxima_grid[basin_idx])

            # otherwise, append it to the existing domain
            else:
                group_index = domain_group_indices[basin_group]
                domain_groups[group_index].append(basin_idx)

        # We now have the groups that exist at the current level. We want to
        # check if they are different from the last set of groups. By construction,
        # the groups will always be ordered from lowest basin to highest

        # copy our previous group ids
        previous_domain_ids = [i for i in domain_ids]
        domain_ids = []
        same_groups = True

        for group in domain_groups:
            # check to see if this group exists in the previous groups
            group_found = False
            for pgroup, pgroup_idx in zip(previous_groups, previous_domain_ids):
                if len(pgroup) != len(group):
                    continue
                # check if all entries in the group equal the other group
                is_previous = True
                for i, j in zip(group, pgroup):
                    if i != j:
                        is_previous = False
                        break
                if is_previous:
                    # note we found a previous group and add the
                    # domains index
                    group_found = True
                    domain_ids.append(pgroup_idx)
                    break
            if not group_found:
                # Note we found a new domain
                same_groups = False
                domain_ids.append(unique_domains)
                unique_domains += 1

        #######################################################################
        # Update Dimensionalities
        #######################################################################
        # copy previous dimensionalities
        old_dimensionalities = [i for i in new_dimensionalities]
        new_dimensionalities = []

        previous_solid = new_solid
        new_solid = data >= value
        # calculate new dimensionalities
        root_mask, parent, offset_x, offset_y, offset_z, roots, cycles, dims = (
            get_connected_groups(
                solid=new_solid,
                previous_solid=previous_solid,
                root_mask=root_mask,
                roots=roots,
                cycles=cycles,
                parent=parent,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_z=offset_z,
                size=size,
                neighbors=neighbor_transforms,
            )
        )

        for idx, (domain_point, domain_id) in enumerate(zip(domain_points, domain_ids)):
            # get this domains dimensionality
            new_dim = get_domain_dimensionality(
                parent=parent,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_z=offset_z,
                roots=roots,
                dims=dims,
                domain_point=domain_point,
                ny_nz=ny_nz,
                nz=nz,
            )
            new_dimensionalities.append(new_dim)
            # check if this domain exists in the previous group and if so, check
            # if it has the same dimensionality
            for prev_idx, prev_dim in zip(previous_domain_ids, old_dimensionalities):
                if prev_idx == domain_id and prev_dim != new_dim:
                    # this is actually a new domain, so we update its id
                    domain_ids[idx] = unique_domains
                    unique_domains += 1
                    same_groups = False
                    break

        # if a new domain appeared, we append the current groups
        if not same_groups:
            bifurcation_values.append(value)
            bifurcation_domains.append(domain_groups)
            bifurcation_domain_indices.append(domain_ids)
            bifurcation_domain_dims.append(new_dimensionalities)

        # update our previous elf value
        previous_value = value

        # if we've found a single domain containing all basins that is 3D, we
        # can break as no further bifurcations will be found
        if len(domain_groups) == 1:
            if len(domain_groups[0]) == num_basins and new_dimensionalities[0] == 3:
                break

    #######################################################################
    # Organize Domains
    #######################################################################
    # reverse values to go from low to high
    bifurcation_values.reverse()
    bifurcation_domains.reverse()
    bifurcation_domain_indices.reverse()
    bifurcation_domain_dims.reverse()

    # Create arrays to track domains
    domain_basins = [[-1] for i in range(unique_domains)]
    domain_min_values = np.empty(unique_domains, dtype=np.float64)
    domain_max_values = np.empty(unique_domains, dtype=np.float64)
    domain_dims = np.empty(unique_domains, dtype=np.int64)
    domain_parents = np.empty(unique_domains, dtype=np.int64)

    # Add our initial domains where all possible basin connections exist. This
    # will be at our lowest value.
    # BUGFIX: There could be multiple atoms/molecules that
    # are separated by vacuum resulting in more than one root in our graph
    min_data = data.min()
    dom_count = 0
    for dom_basins, feat_dim in zip(bifurcation_domains[0], bifurcation_domain_dims[0]):
        domain_basins[dom_count] = dom_basins
        domain_min_values[dom_count] = min_data
        domain_dims[dom_count] = feat_dim
        domain_parents[dom_count] = -1  # No parent
        dom_count += 1

    # Now we loop over our elf values.
    # NOTE: the basins at each value are the ones that exist at or below that
    # value. Therefore, the nodes that appear right above that value are those
    # in the next index of the list
    for bif_idx, value in enumerate(bifurcation_values[:-1]):
        # get the domains that exist exactly at this value
        old_domain_indices = bifurcation_domain_indices[bif_idx]
        # old_dimensions = bifurcation_domain_dims[bif_idx]
        # get the domains that appear right above this value
        new_domains = bifurcation_domains[bif_idx + 1]
        new_domain_indices = bifurcation_domain_indices[bif_idx + 1]
        new_dimensions = bifurcation_domain_dims[bif_idx + 1]
        # Now we loop over the new domains and add any new ones to our graph
        for dom_idx, dom_basins, feat_dim in zip(
            new_domain_indices, new_domains, new_dimensions
        ):
            # check if this domain exists in the previous set of indices
            if dom_idx in old_domain_indices:
                continue

            # if we're still here, this is a new domain and we record its attributes
            domain_basins[dom_count] = dom_basins
            domain_min_values[dom_count] = value
            domain_dims[dom_count] = feat_dim

            # find the parent of this domain. We need to iterate backwards over
            # the previously found domains and find the first one containing
            # the basins in the current domain
            possible_basins = domain_basins[:dom_count]
            possible_basins.reverse()

            for idx, (parent_basins) in enumerate(possible_basins):
                if np.all(np.isin(dom_basins, parent_basins)):
                    parent_idx = dom_count - idx - 1
                    break
            # add the parent connection
            domain_parents[dom_count] = parent_idx

            # note this parent disappears at this value
            domain_max_values[parent_idx] = value

            # if this domain is irreducible, add its max value
            if len(dom_basins) == 1:
                domain_max_values[dom_count] = basin_maxima_ref_values[dom_basins[0]]

            # note we found a new domain
            dom_count += 1

    return (
        domain_basins,
        domain_min_values,
        domain_max_values,
        domain_dims,
        domain_parents,
    )
