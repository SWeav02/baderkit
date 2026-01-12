# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange

from baderkit.core.utilities.basic import coords_to_flat, wrap_point
from baderkit.core.utilities.union_find import find_root_no_compression

from .bifurcation_base import get_connected_groups


@njit(cache=True, parallel=True)
def get_connected_voids(
    solid,
    previous_solid,
    basin_labels,
    basin_domain_map,
    num_domains,
    root_mask,
    roots,
    cycles,
    parent,
    offset_x,
    offset_y,
    offset_z,
    size,
    neighbors,
    vacuum_mask,
):
    """
    Finds unions and periodic offsets for points in a periodic solid. Slower than
    scipy's implementation, but allows for iterative updates with increasingly
    large solids.

    Also counts the number of contact points between each domain and void which
    is used to determine which groups surround each other later
    """
    nx, ny, nz = solid.shape
    ny_nz = ny * nz

    # get the connected points for this solid
    (
        root_mask,
        parent,
        offset_x,
        offset_y,
        offset_z,
        roots,
        cycles,
        dimensionalities,
    ) = get_connected_groups(
        solid,
        previous_solid,
        root_mask,
        roots,
        cycles,
        parent,
        offset_x,
        offset_y,
        offset_z,
        size,
        neighbors,
        vacuum_mask,
    )

    # create a mask to indicate the number of connections between domains and
    # and voids
    connection_counts = np.zeros((len(roots), num_domains), dtype=np.uint32)
    # loop over indices and count connections
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):

                # skip anything not in the solid
                if not solid[i, j, k] or vacuum_mask[i,j,k]:
                    continue

                root_idx = -1  # we don't calculate these unless we have to
                # loop over neighbors
                for di, dj, dk in neighbors:
                    # get wrapped neighbor
                    ni, nj, nk = wrap_point(i + di, j + dj, k + dk, nx, ny, nz)
                    # skip anything in the solid
                    if solid[ni, nj, nk]:
                        continue

                    # otherwise, get the connection here
                    if root_idx == -1:
                        idx = coords_to_flat(i, j, k, ny_nz, nz)
                        root = find_root_no_compression(parent, idx)
                        root_idx = np.searchsorted(roots, root)

                    neigh_basin = basin_labels[ni, nj, nk]
                    domain = basin_domain_map[neigh_basin]

                    # this isn't technically safe, but I think it will usually
                    # have very little effect and it makes a big difference in
                    # speed
                    connection_counts[root_idx, domain] += 1

    return (
        root_mask,
        parent,
        offset_x,
        offset_y,
        offset_z,
        roots,
        cycles,
        dimensionalities,
        connection_counts,
    )


@njit(cache=True)
def find_atom_domains(
    atom_coords,
    parent,
    solid,
    basin_labels,
    basin_domain_map,
    all_domains,
    connection_counts,
    void_roots,
    domain_dims,
    void_dims,
    surrounding_domains,
):
    """
    Finds which domains completely surround an atom.
    """
    nx, ny, nz = basin_labels.shape
    ny_nz = ny * nz
    i, j, k = atom_coords

    # get the void or domain the atom sits in
    if solid[i, j, k]:
        atom_flat_idx = coords_to_flat(i, j, k, ny_nz, nz)
        root = find_root_no_compression(parent, atom_flat_idx)
        current_group = np.searchsorted(void_roots, root)
        is_void = True
    else:
        basin_idx = basin_labels[i, j, k]
        current_group = basin_domain_map[basin_idx]
        is_void = False

    # iteratively search for a group (domain or void) of the opposite type
    # that surrounds the current group. For a group to be "surrounded" it must
    # follow a set of rules:
    # 1. The group itself must be finite (0D).
    # 2. The surrounding group must have the most connecting points of any
    # possible group
    # NOTE: Any finite domain/void must by definition be surrounded by a group
    # of the opposite type or it wouldn't be finite. This group will also always
    # have more surface area in contact with the domain/void than any other
    # group. In other words, the most connecting points.
    while_count = 0
    while True:
        if while_count > 100:
            raise Exception()
        while_count += 1
        if is_void:
            # get dimensionality
            void_dim = void_dims[current_group]
            # if the void is infinite, it can't be surrounded and we break
            if void_dim > 0:
                break
            # otherwise, we find the domain with the most connections to this
            # void
            current_group = np.argmax(connection_counts[current_group])
            is_void = False
        else:
            # add this domain to our list
            domain_idx = all_domains[current_group]
            surrounding_domains.append(domain_idx)
            # get its dimensionality
            domain_dim = domain_dims[domain_idx]
            # if the domain is infinite, it can't be surrounded and we break
            if domain_dim > 0:
                break
            # otherwise, we find the void with the most connections to this
            # domain
            current_group = np.argmax(connection_counts[:, current_group])
            is_void = True
    return surrounding_domains


@njit(parallel=True, cache=True)
def find_all_atom_domains(
    atom_grid_coords,
    parent,
    solid,
    basin_labels,
    basin_domain_map,
    all_domains,
    connection_counts,
    void_roots,
    domain_dims,
    void_dims,
):
    """
    Finds which domains completely surround each atom.
    """

    # create lists to store which domains surround each atom
    all_surrounding_domains = []
    for i in range(len(atom_grid_coords)):
        domain_list = [-1]
        domain_list = domain_list[1:]  # -1 is placeholder for numba typing
        all_surrounding_domains.append(domain_list)

    # TODO: Create tracker for if atom sits inside domain vs. surrounded by it?
    # Should be easy, but I'm not sure I would use it yet
    for atom_idx in prange(len(atom_grid_coords)):
        atom_coords = atom_grid_coords[atom_idx]
        surrounding_domains = all_surrounding_domains[atom_idx]

        find_atom_domains(
            atom_coords,
            parent,
            solid,
            basin_labels,
            basin_domain_map,
            all_domains,
            connection_counts,
            void_roots,
            domain_dims,
            void_dims,
            surrounding_domains,
        )

    return all_surrounding_domains


@njit(cache=True)
def get_domains_surrounding_atoms(
    possible_values,
    domain_basins,
    domain_min_values,
    domain_max_values,
    domain_dims,
    domain_parents,
    atom_grid_coords,
    neighbor_transforms,
    basin_labels,
    data,
    num_basins,
    vacuum_mask,
):
    """
    Finds the values at which changes in void connections occur and checks if
    there is a change in which domains surround each atom
    """
    nx, ny, nz = data.shape
    N = nx * ny * nz

    # create map from basin labels to domain labels
    basin_domain_map = np.empty(num_basins, dtype=np.int64)

    # create a map for old domains to new
    new_domain_map = np.full(len(domain_parents), -1, dtype=np.int64)

    # create an initial empty solid
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

    # create lists for updated domains
    new_domain_basins = []
    new_domain_min_values = []
    new_domain_max_values = []
    new_domain_dims = []
    new_domain_parents = []

    # create lists to track which domains surround atoms
    all_domain_atoms = []

    current_domains = [-1]
    current_domains = current_domains[1:]
    current_domain_atoms = [[-1]]
    current_domain_atoms = current_domain_atoms[1:]
    # iterate over all possible values
    for current_value in possible_values:
        basin_domain_map[:] = -1
        previous_domains = [i for i in current_domains]
        current_domains = []
        new_domains = []
        num_domains = 0
        # get the domains that exist at this elf value and construct basin map
        for domain_idx, (min_value, max_value) in enumerate(
            zip(domain_min_values, domain_max_values)
        ):

            if min_value <= current_value and max_value > current_value:
                # This domain currently has some volume in the isosolid
                # create map for each basin in this domain
                # add it to our list
                current_domains.append(domain_idx)
                # map each basin in this domain back to the domain
                for basin_idx in domain_basins[domain_idx]:
                    basin_domain_map[basin_idx] = num_domains

                # note we found a new domain
                num_domains += 1
                if min_value == current_value:
                    new_domains.append(domain_idx)

        # get void/domain connection information at this elf value
        previous_solid = new_solid
        new_solid = data <= current_value
        # TODO: I can probably improve speed further by freezing domains/basins
        # that have been found to not surround anything, decreasing the number
        # of voxels that need to be checked for unions

        (
            root_mask,
            parent,
            offset_x,
            offset_y,
            offset_z,
            roots,
            cycles,
            dimensionalities,
            connection_counts,
        ) = get_connected_voids(
            solid=new_solid,
            previous_solid=previous_solid,
            basin_labels=basin_labels,
            basin_domain_map=basin_domain_map,
            num_domains=num_domains,
            root_mask=root_mask,
            roots=roots,
            cycles=cycles,
            parent=parent,
            offset_x=offset_x,
            offset_y=offset_y,
            offset_z=offset_z,
            size=size,
            neighbors=neighbor_transforms,
            vacuum_mask=vacuum_mask,
        )

        # Get which domains surround each atom
        all_surrounding_domains = find_all_atom_domains(
            atom_grid_coords=atom_grid_coords,
            parent=parent,
            solid=new_solid,
            basin_labels=basin_labels,
            basin_domain_map=basin_domain_map,
            all_domains=current_domains,
            connection_counts=connection_counts,
            void_roots=roots,
            domain_dims=domain_dims,
            void_dims=dimensionalities,
        )

        # Get the atoms surrounded by each domain
        # track if no atoms are surrounded
        no_atoms_surrounded = True
        previous_domain_atoms = [i.copy() for i in current_domain_atoms]
        current_domain_atoms = []
        for i in range(len(current_domains)):
            domain_list = [-1]
            domain_list = domain_list[1:]
            current_domain_atoms.append(domain_list)
        # Add atoms to domain lists
        for atom_idx, domain_list in enumerate(all_surrounding_domains):
            if len(domain_list) == 0:
                continue
            no_atoms_surrounded = False
            for domain_idx in domain_list:
                for sub_idx, current_domain_idx in enumerate(current_domains):
                    if domain_idx == current_domain_idx:
                        current_domain_atoms[sub_idx].append(atom_idx)
                        break

        # append any new domains
        for domain_idx, domain_atoms in zip(current_domains, current_domain_atoms):
            append = False
            is_new = False
            if domain_idx in new_domains:
                append = True
            else:
                # check if this domain changed the number of atoms it contains
                for old_domain_idx, old_domain_atoms in zip(
                    previous_domains, previous_domain_atoms
                ):
                    if old_domain_idx == domain_idx:
                        if len(old_domain_atoms) != len(domain_atoms):
                            append = True
                            is_new = True
                        break
            if append:
                new_domain_basins.append(domain_basins[domain_idx])
                new_domain_min_values.append(current_value)
                new_domain_max_values.append(domain_max_values[domain_idx])
                new_domain_dims.append(domain_dims[domain_idx])
                all_domain_atoms.append(domain_atoms)
                if is_new:
                    previous_idx = new_domain_map[domain_idx]
                    # This domain changed atom counts. Update its old max
                    # value to be the current value
                    new_domain_max_values[previous_idx] = current_value
                    # set a new mapping for this domain
                    new_domain_map[domain_idx] = len(new_domain_parents)
                    # set the parent to the previous version surrounding a different
                    # number of atoms
                    new_domain_parents.append(previous_idx)
                else:
                    # update this domains index
                    new_domain_map[domain_idx] = len(new_domain_parents)
                    # get the old parent for this domain
                    old_parent = domain_parents[domain_idx]
                    if old_parent == -1:
                        new_domain_parents.append(-1)
                        continue
                    # get the parents new index
                    new_parent = new_domain_map[old_parent]
                    new_domain_parents.append(new_parent)

        # if no atoms are surrounded, they never will be again and we are done
        # here
        if no_atoms_surrounded:
            break

    # we need to fill in the data for any remaining domains that don't surround
    # atoms
    for domain_idx in range(len(domain_min_values)):
        # check if this domain was ever given a mapping
        if not new_domain_map[domain_idx] == -1:
            continue
        # otherwise, we add all of the needed domains
        new_domain_basins.append(domain_basins[domain_idx])
        new_domain_min_values.append(domain_min_values[domain_idx])
        new_domain_max_values.append(domain_max_values[domain_idx])
        new_domain_dims.append(domain_dims[domain_idx])
        all_domain_atoms.append([-1])
        all_domain_atoms[-1] = all_domain_atoms[-1][1:]
        # update this domains index
        new_domain_map[domain_idx] = len(new_domain_parents)
        # get the old parent for this domain
        old_parent = domain_parents[domain_idx]
        if old_parent == -1:
            new_domain_parents.append(-1)
            continue
        # get the parents new index
        new_parent = new_domain_map[old_parent]
        new_domain_parents.append(new_parent)

    return (
        new_domain_basins,
        np.array(new_domain_min_values, dtype=np.float64),
        np.array(new_domain_max_values, dtype=np.float64),
        np.array(new_domain_dims, dtype=np.int64),
        np.array(new_domain_parents, dtype=np.int64),
        all_domain_atoms,
    )
