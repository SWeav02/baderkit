# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True)
def _compute_tetra_weights_fast(
        energy_grid, 
        e,
        ):
    e1, e2, e3, e4 = e[:, 0, :], e[:, 1, :], e[:, 2, :], e[:, 3, :]
    
    # Precompute differences safely
    e21, e31, e41, e32, e42, e43 = e2-e1, e3-e1, e4-e1, e3-e2, e4-e2, e4-e3
    
    c21 = np.where(e21 > 1e-12, 1.0 / np.maximum(e21 * e31 * e41, 1e-12), 0.0)
    c32_4 = np.where(e32 > 1e-12, -(e31 + e42) / np.maximum(e31 * e41 * e32 * e42, 1e-12), 0.0)
    c32_3 = np.where(e32 > 1e-12, 3.0 / np.maximum(e31 * e41, 1e-12), 0.0)
    c43 = np.where(e43 > 1e-12, -1.0 / np.maximum(e43 * e42 * e41, 1e-12), 0.0)
    
    num_points = len(energy_grid)
    ntetra, nbands = e1.shape
    G_out = np.zeros((num_points, ntetra, nbands))
    
    for ie in prange(num_points):
        E = energy_grid[ie]
        for t in range(ntetra):
            for b in range(nbands):
                ve1 = E - e1[t, b]
                ve2 = E - e2[t, b]
                ve3 = E - e3[t, b]
                ve4 = E - e4[t, b]
                
                if ve1 > 0 and ve2 <= 0:
                    G_out[ie, t, b] = 3.0 * c21[t, b] * ve1 * ve1
                elif ve2 > 0 and ve3 <= 0:
                    G_out[ie, t, b] = (c32_3[t, b] * (e2[t, b] - e1[t, b])) + ve2 * (2.0 * c32_3[t, b] + 3.0 * ve2 * c32_4[t, b])
                elif ve3 > 0 and ve4 <= 0:
                    G_out[ie, t, b] = -3.0 * c43[t, b] * ve4 * ve4
    return G_out