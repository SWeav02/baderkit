import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
def _integrate_tetrahedra_spectral_density_numba(
    egrid,
    tetra_indices,
    eigenvalues,
    cached_metrics,
    tetra_weight
):
    """
    Parallelized JIT-compiled linear tetrahedron engine adhering strictly to 
    Blöchl's linear property-weight interpolation scheme.
    """
    n_omega = egrid.shape[0]
    n_tetra = tetra_indices.shape[0]
    n_spin = eigenvalues.shape[0]
    n_bands = eigenvalues.shape[2]
    n_metrics = cached_metrics.shape[3]

    out = np.zeros((n_metrics, n_spin, n_omega), dtype=np.float64)

    for s in prange(n_spin):
        for b in range(n_bands):

            local = np.zeros((n_metrics, n_omega), dtype=np.float64)

            for t in range(n_tetra):

                k1 = tetra_indices[t, 0]
                k2 = tetra_indices[t, 1]
                k3 = tetra_indices[t, 2]
                k4 = tetra_indices[t, 3]

                # Extract corner energies
                e1 = eigenvalues[s, k1, b]
                e2 = eigenvalues[s, k2, b]
                e3 = eigenvalues[s, k3, b]
                e4 = eigenvalues[s, k4, b]

                # Track and sort indices alongside energies to preserve corner properties
                idx = np.array([k1, k2, k3, k4])
                
                # Stable bubble sort for the 4 corners
                for i in range(3):
                    for j in range(3 - i):
                        if eigenvalues[s, idx[j], b] > eigenvalues[s, idx[j+1], b]:
                            tmp = idx[j]
                            idx[j] = idx[j+1]
                            idx[j+1] = tmp

                # Map sorted energies and their original k-point index array locations
                e1 = eigenvalues[s, idx[0], b]
                e2 = eigenvalues[s, idx[1], b]
                e3 = eigenvalues[s, idx[2], b]
                e4 = eigenvalues[s, idx[3], b]

                idx1 = idx[0]
                idx2 = idx[1]
                idx3 = idx[2]
                idx4 = idx[3]

                # Skip completely flat or invalid tetrahedra
                if e4 - e1 < 1e-7:
                    continue

                e21 = e2 - e1
                e31 = e3 - e1
                e41 = e4 - e1
                e42 = e4 - e2
                e32 = e3 - e2
                e43 = e4 - e3

                for w in range(n_omega):
                    E = egrid[w]

                    if E < e1 or E >= e4:
                        continue

                    w1, w2, w3, w4 = 0.0, 0.0, 0.0, 0.0

                    # CASE 1: e1 <= E < e2 (Triangular Cross-Section)
                    if E < e2:
                        if e21 > 1e-7 and e31 > 1e-7 and e41 > 1e-7:
                            G = 3.0 * (E - e1)**2 / (e21 * e31 * e41)
                            w1 = G * (1.0 - (E - e1) * (1.0/e21 + 1.0/e31 + 1.0/e41) / 3.0)
                            w2 = G * (E - e1) / (3.0 * e21)
                            w3 = G * (E - e1) / (3.0 * e31)
                            w4 = G * (E - e1) / (3.0 * e41)

                    # CASE 2: e2 <= E < e3 (Quadrilateral Cross-Section)
                    elif E < e3:
                        if e41 > 1e-7 and e31 > 1e-7 and e42 > 1e-7 and e32 > 1e-7:
                            
                            # 1. Total exact quadratic DOS contribution for the interval
                            G = 3.0 * (E - e1) * (e3 - E) / (e41 * e31 * e32) + 3.0 * (e4 - E) * (E - e2) / (e41 * e42 * e32)
                            
                            # 2. Exact analytical Blöchl weights for the outer corners
                            w1 = (1.0 / (e41 * e32)) * ( 
                                ((E - e1) * (e3 - E)**2) / (e31**2) + 
                                ((E - e1) * (e3 - E) * (e4 - E)) / (e41 * e31) + 
                                ((e4 - E)**2 * (E - e2)) / (e41 * e42) 
                            )
                            
                            w4 = (1.0 / (e41 * e32)) * ( 
                                ((e4 - E) * (E - e2)**2) / (e42**2) + 
                                ((e4 - E) * (E - e2) * (E - e1)) / (e41 * e42) + 
                                ((E - e1)**2 * (e3 - E)) / (e41 * e31) 
                            )
                            
                            # 3. Exact inner corners via strict energy and charge conservation
                            w2 = (G * (e3 - E) - w1 * e31 + w4 * e43) / e32
                            w3 = G - w1 - w2 - w4

                    # CASE 3: e3 <= E < e4 (Triangular Cross-Section)
                    else:
                        if e41 > 1e-7 and e42 > 1e-7 and e43 > 1e-7:
                            G = 3.0 * (e4 - E)**2 / (e41 * e42 * e43)
                            w1 = G * (e4 - E) / (3.0 * e41)
                            w2 = G * (e4 - E) / (3.0 * e42)
                            w3 = G * (e4 - E) / (3.0 * e43)
                            w4 = G * (1.0 - (e4 - E) * (1.0/e41 + 1.0/e42 + 1.0/e43) / 3.0)

                    # Accumulate contracted metrics using the calculated property coordinates
                    for m in range(n_metrics):
                        val1 = cached_metrics[s, idx1, b, m]
                        val2 = cached_metrics[s, idx2, b, m]
                        val3 = cached_metrics[s, idx3, b, m]
                        val4 = cached_metrics[s, idx4, b, m]

                        integrated_val = (w1*val1 + w2*val2 + w3*val3 + w4*val4) * tetra_weight
                        local[m, w] += integrated_val

            for m in range(n_metrics):
                for w in range(n_omega):
                    out[m, s, w] += local[m, w]

    return out
