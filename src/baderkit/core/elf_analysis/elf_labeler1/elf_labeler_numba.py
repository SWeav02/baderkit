# -*- coding: utf-8 -*-

from numba import njit, prange, types

import numpy as np
from numpy.typing import NDArray

from baderkit.core.utilities.transforms import INT_TO_IMAGE, IMAGE_TO_INT
from baderkit.core.utilities.interpolation import linear_slice

@njit(cache=True, parallel=True)
def get_core_dist_ratios(
    labels,
    basin_frac_coords,
    atom_frac_coords,
    nna_indices,
    core_basins,
    volume_bond_fracs,
        ):
    
    basin_core_dist_ratios = np.zeros(len(nna_indices), dtype=np.float64)
    
    for nna_idx in prange(len(nna_indices)):
        # skip cores
        local_idx = nna_indices[nna_idx]
        local_coords = basin_frac_coords[local_idx]
        local_bond_frac = volume_bond_fracs[local_idx]
        all_fracs = []
        for atom_idx, atom_image, frac in local_bond_frac:
            atom_coords = atom_frac_coords[int(atom_idx)] + INT_TO_IMAGE[int(atom_image)]
            # labels between the coords
            label_line = linear_slice(labels, atom_coords, local_coords, method="nearest")
            # get the last point that is part of the core
            for idx, i in enumerate(label_line):
                if core_basins[int(i)] == -1:
                    break
            if idx == 0:
                continue
            # get ratio of nna / core
            frac = (len(label_line)-idx) / (idx+1)
            all_fracs.append(frac)
        if not all_fracs:
            continue
        all_fracs = np.array(all_fracs, dtype=np.float64)
        average_frac = np.average(all_fracs)
        basin_core_dist_ratios[nna_idx] = average_frac
    return basin_core_dist_ratios