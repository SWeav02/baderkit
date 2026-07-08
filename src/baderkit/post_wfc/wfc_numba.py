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
    
    UPDATED: Synchronized boundary intervals and flat-band delta-spike 
    deposition to guarantee absolute mathematical consistency with the analytic charge engine.
    """
    n_omega = egrid.shape[0]
    n_tetra = tetra_indices.shape[0]
    n_spin = eigenvalues.shape[0]
    n_bands = eigenvalues.shape[2]
    n_metrics = cached_metrics.shape[3]

    out = np.zeros((n_metrics, n_spin, n_omega), dtype=np.float64)
    
    # Calculate energy grid spacing for delta function normalization
    delta_e = egrid[1] - egrid[0] if n_omega > 1 else 1.0

    for s in prange(n_spin):
        for b in range(n_bands):
            local = np.zeros((n_metrics, n_omega), dtype=np.float64)

            for t in range(n_tetra):
                k1 = tetra_indices[t, 0]
                k2 = tetra_indices[t, 1]
                k3 = tetra_indices[t, 2]
                k4 = tetra_indices[t, 3]

                e1 = eigenvalues[s, k1, b]
                e2 = eigenvalues[s, k2, b]
                e3 = eigenvalues[s, k3, b]
                e4 = eigenvalues[s, k4, b]

                idx = np.array([k1, k2, k3, k4])
                
                # Stable bubble sort for vertex alignment
                for i in range(3):
                    for j in range(3 - i):
                        if eigenvalues[s, idx[j], b] > eigenvalues[s, idx[j+1], b]:
                            tmp = idx[j]
                            idx[j] = idx[j+1]
                            idx[j+1] = tmp

                e1 = eigenvalues[s, idx[0], b]
                e2 = eigenvalues[s, idx[1], b]
                e3 = eigenvalues[s, idx[2], b]
                e4 = eigenvalues[s, idx[3], b]

                idx1, idx2, idx3, idx4 = idx[0], idx[1], idx[2], idx[3]

                # FIXED: Handle flat bands as exact Dirac Delta Spikes 
                # instead of discarding them.
                if e4 - e1 < 1e-7:
                    # Find where the flat band energy lands on our discrete grid
                    w_closest = int(np.round((e1 - egrid[0]) / delta_e))
                    if 0 <= w_closest < n_omega:
                        for m in range(n_metrics):
                            val1 = cached_metrics[s, idx1, b, m]
                            # Deposit property weight normalized by dE
                            local[m, w_closest] += (val1 * tetra_weight) / delta_e
                    continue

                e21 = e2 - e1
                e31 = e3 - e1
                e41 = e4 - e1
                e42 = e4 - e2
                e32 = e3 - e2 if (e3 - e2) > 1e-12 else 1.0
                e43 = e4 - e3 if (e4 - e3) > 1e-12 else 1.0

                for w in range(n_omega):
                    E = egrid[w]

                    # Left-Closed, Right-Open boundary tracking matching the charge engine
                    if E < e1 or E >= e4:
                        continue

                    w1, w2, w3, w4 = 0.0, 0.0, 0.0, 0.0

                    # CASE 1: e1 <= E < e2
                    if E < e2:
                        if e21 > 1e-7 and e31 > 1e-7 and e41 > 1e-7:
                            G = 3.0 * (E - e1)**2 / (e21 * e31 * e41)
                            w1 = G * (1.0 - (E - e1) * (1.0/e21 + 1.0/e31 + 1.0/e41) / 3.0)
                            w2 = G * (E - e1) / (3.0 * e21)
                            w3 = G * (E - e1) / (3.0 * e31)
                            w4 = G * (E - e1) / (3.0 * e41)

                    # CASE 2: e2 <= E < e3
                    elif E < e3:
                        if e41 > 1e-7 and e31 > 1e-7 and e42 > 1e-7 and e32 > 1e-7:
                            G = 3.0 * (E - e1) * (e3 - E) / (e41 * e31 * e32) + 3.0 * (e4 - E) * (E - e2) / (e41 * e42 * e32)
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
                            w2 = (G * (e3 - E) - w1 * e31 + w4 * e43) / e32
                            w3 = G - w1 - w2 - w4

                    # CASE 3: e3 <= E < e4
                    else:
                        if e41 > 1e-7 and e42 > 1e-7 and e43 > 1e-7:
                            G = 3.0 * (e4 - E)**2 / (e41 * e42 * e43)
                            w1 = G * (e4 - E) / (3.0 * e41)
                            w2 = G * (e4 - E) / (3.0 * e42)
                            w3 = G * (e4 - E) / (3.0 * e43)
                            w4 = G * (1.0 - (e4 - E) * (1.0/e41 + 1.0/e42 + 1.0/e43) / 3.0)

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


@njit(parallel=True, fastmath=True, cache=True)
def _integrate_tetrahedra_analytic_charge_numba(
    energy_grid: np.ndarray,
    tetra_indices: np.ndarray,
    eigenvalues: np.ndarray,
    tetra_weight: float,
    rspin: float
) -> np.ndarray:
    """
    Analytically integrates the cumulative state weight N(E) across all micro-cell 
    tetrahedra. Loop hierarchy matched perfectly to the spectral density layout 
    for maximum caching efficiency and synchronized boundary mappings.
    """
    num_points = len(energy_grid)
    num_tets = tetra_indices.shape[0]
    nspin = eigenvalues.shape[0]
    nbands = eigenvalues.shape[2]
    
    # Store per-spin arrays to preserve race-condition-free prange scaling
    out_spin = np.zeros((nspin, num_points), dtype=np.float64)
    
    for s in prange(nspin):
        for b in range(nbands):
            local_charge = np.zeros(num_points, dtype=np.float64)
            
            for t in range(num_tets):
                k0 = tetra_indices[t, 0]
                k1 = tetra_indices[t, 1]
                k2 = tetra_indices[t, 2]
                k3 = tetra_indices[t, 3]
                
                e1 = eigenvalues[s, k0, b]
                e2 = eigenvalues[s, k1, b]
                e3 = eigenvalues[s, k2, b]
                e4 = eigenvalues[s, k3, b]
                
                # Zero-allocation inline scalar sorting network
                if e1 > e2: e1, e2 = e2, e1
                if e3 > e4: e3, e4 = e4, e3
                if e1 > e3: e1, e3 = e3, e1
                if e2 > e4: e2, e4 = e4, e2
                if e2 > e3: e2, e3 = e3, e2
                
                # FIXED: Flat band step-function tracking protects against electron loss
                if e4 - e1 < 1e-7:
                    for w in range(num_points):
                        if energy_grid[w] >= e1:
                            local_charge[w] += tetra_weight
                    continue
                
                e21 = e2 - e1
                e31 = e3 - e1
                e41 = e4 - e1
                e42 = e4 - e2
                e32 = e3 - e2 if (e3 - e2) > 1e-12 else 1.0
                e43 = e4 - e3 if (e4 - e3) > 1e-12 else 1.0
                denom_base = e31 * e41
                
                for w in range(num_points):
                    E = energy_grid[w]
                    
                    # FIXED: Synchronized Left-Closed, Right-Open boundary checks
                    if E < e1:
                        continue
                    elif E >= e4:
                        local_charge[w] += tetra_weight
                    elif e1 <= E < e2:
                        denom = e21 * e31 * e41
                        if denom > 1e-12:
                            local_charge[w] += tetra_weight * ((E - e1) ** 3) / denom
                    elif e2 <= E < e3:
                        if denom_base > 1e-12:
                            h = E - e2
                            v1 = e2 - e1
                            factor = (e31 + e42) / (e32 * e42)
                            term = v1**2 + 3.0*v1*h + 3.0*h**2 - factor * (h**3)
                            local_charge[w] += tetra_weight * term / denom_base
                    elif e3 <= E < e4:
                        denom = e41 * e42 * e43
                        if denom > 1e-12:
                            local_charge[w] += tetra_weight * (1.0 - ((e4 - E) ** 3) / denom)
            
            for w in range(num_points):
                out_spin[s, w] += local_charge[w]
                
    # Contract spin dimensions and scale by system degeneracies
    total_charge = np.zeros(num_points, dtype=np.float64)
    for w in range(num_points):
        for s in range(nspin):
            total_charge[w] += out_spin[s, w]
            
    return total_charge * rspin
