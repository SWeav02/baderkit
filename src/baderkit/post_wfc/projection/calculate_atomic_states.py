import logging
import re
from pathlib import Path
import traceback
import numpy as np
from mendeleev import element as get_mendeleev_element
from pyscf import dft, gto, lib, scf
from pyscf.data.nist import HARTREE2EV

# ==============================================================================
# SECTION 1: Global Configurations & Logging Setup
# ==============================================================================

START_IDX = 1
END_IDX = 93  # 1 to 92 (H through U)
BASIS = "dyall-ae3z"
MAX_MEMORY_MB = 16000  # 16 GB memory limit
NOBLE_GASES = {2, 10, 18, 36, 54, 86}

# Configure logging format for tracking execution stages and failure points
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Configure PySCF threading and memory allocation
lib.num_threads(lib.num_threads())
lib.param.MAX_MEMORY = MAX_MEMORY_MB

# Angular momentum quantum number mapping
L_MAP = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


# ==============================================================================
# SECTION 2: Labeling, Symmetry & Expectation Value Helpers
# ==============================================================================

def update_status(folder: Path, status: str, err_msg: str = None):
    """Updates status flag files cleanly in the atom directory and logs errors."""
    for s in ["RUNNING", "SUCCEEDED", "FAILED"]:
        (folder / s).unlink(missing_ok=True)
    (folder / status).touch()
    if err_msg and status == "FAILED":
        err_file = folder / "error.log"
        err_file.write_text(err_msg)


def get_noble_core_orbitals(z: int) -> int:
    """Returns the number of core spatial orbitals based on noble gas core."""
    return next(c for c in [86, 54, 36, 18, 10, 2, 0] if z > c) // 2


def extract_subshell_base(label: str) -> str:
    """Extracts principal quantum number and orbital type (e.g. '4px' -> '4p')."""
    m = re.match(r"(\d+[spdfgh])", label)
    return m.group(1) if m else label


def assign_physical_labels(mol, mo_coeff, energies, occ=None) -> list:
    """Assigns spectroscopic quantum labels (1s, 2s, 2p, 3s...) sorted by Fock energy."""
    ao_l, ao_comp = [], []
    for _, _, nl, comp in mol.ao_labels(fmt=False):
        l_char = nl[-1]
        ao_l.append(L_MAP[l_char])
        ao_comp.append((l_char, comp))

    ao_l = np.array(ao_l)
    S = mol.intor_symmetric("int1e_ovlp")
    pop = mo_coeff * (S @ mo_coeff)

    mo_l, mo_comp_str = [], []
    for i in range(mo_coeff.shape[1]):
        l_pop = [pop[ao_l == l, i].sum() for l in range(6)]
        dom_l = np.argmax(l_pop)
        mo_l.append(dom_l)

        sub_indices = np.where(ao_l == dom_l)[0]
        dom_k = sub_indices[np.argmax(np.abs(pop[sub_indices, i]))]
        l_char, comp = ao_comp[dom_k]
        mo_comp_str.append(f"{l_char}{comp}" if comp else l_char)

    mo_l = np.array(mo_l)
    labels = [""] * len(energies)
    occ_weights = np.zeros(mo_coeff.shape[1]) if occ is None else np.round(occ, 4)

    for l_val in range(6):
        idx_l = np.where(mo_l == l_val)[0]
        if len(idx_l) == 0:
            continue

        sort_keys = np.lexsort((energies[idx_l], -occ_weights[idx_l]))
        idx_l_sorted = idx_l[sort_keys]

        deg, n_curr = 2 * l_val + 1, l_val + 1
        for ptr in range(0, len(idx_l_sorted), deg):
            for idx in idx_l_sorted[ptr : ptr + deg]:
                labels[idx] = f"{n_curr}{mo_comp_str[idx]}"
            n_curr += 1

    return labels


def symmetrize_subshell_values(values, labels) -> np.ndarray:
    """Averages values across degenerate (2L + 1) subshells to preserve spherical symmetry."""
    subshell_groups = {}
    for idx, label in enumerate(labels):
        base_subshell = extract_subshell_base(label)
        subshell_groups.setdefault(base_subshell, []).append(idx)

    symmetrized_values = np.copy(values)
    for _, indices in subshell_groups.items():
        avg_val = np.mean(values[indices])
        symmetrized_values[indices] = avg_val

    return symmetrized_values


def compute_vnuc_expectation(mol, natorbs) -> np.ndarray:
    """Computes expectation value of Nuclear Attraction Operator <V_nuc> in Atomic Units."""
    V_ao = mol.intor_symmetric("int1e_nuc")
    return np.einsum("pi,pq,qi->i", natorbs, V_ao, natorbs)


# ==============================================================================
# SECTION 3: Valence Subshell Targeter
# ==============================================================================

def get_needed_virtual_subshells(symbol: str, z: int, occ_subshell_counts: dict) -> list:
    """
    Determines required virtual orbital counts per l-symmetry to complete all 
    partially filled or unpopulated valence subshells for a given element.
    """
    if z in NOBLE_GASES:
        return []

    elem = get_mendeleev_element(symbol)
    P = elem.period

    if P == 1:
        candidates = [("1s", 0, 1)]
    elif P in [2, 3]:
        candidates = [
            ("2s" if P == 2 else "3s", 0, 1),
            ("2p" if P == 2 else "3p", 1, 3),
        ]
        if P == 3:
            candidates.append(("3d", 2, 5))
    elif P in [4, 5]:
        candidates = [(f"{P}s", 0, 1), (f"{P-1}d", 2, 5), (f"{P}p", 1, 3)]
    elif P in [6, 7]:
        candidates = [(f"{P}s", 0, 1)]
        if (P == 6 and z >= 57) or (P == 7 and z >= 89):
            candidates.append((f"{P-2}f", 3, 7))
        candidates.extend([(f"{P-1}d", 2, 5), (f"{P}p", 1, 3)])

    needed = []
    for sub_label, l_val, deg in candidates:
        n_occ = occ_subshell_counts.get(sub_label, 0)
        n_needed = max(0, deg - n_occ)
        if n_needed > 0:
            needed.append((l_val, n_needed))

    return needed


# ==============================================================================
# SECTION 4: Modified Virtual Orbital (MVO) & Subspace Canonicalization
# ==============================================================================

import scipy.linalg

def compute_mvo_subshell_orbitals(mol, mf, symbol: str, z: int):
    """
    Generates V_{N-1} MVOs to complete incomplete valence subshells and 
    canonicalizes orbitals under the unperturbed neutral Fock operator F^(N).
    Safely handles both restricted (RKS/ROKS) and unrestricted (UKS) mean-field objects.
    """
    logging.info(f"[{symbol}] Constructing neutral Fock matrix F^(N)...")
    
    S = mol.intor_symmetric("int1e_ovlp")
    
    # Handle UKS (tuple of alpha/beta) vs RKS/ROKS structures
    if isinstance(mf.mo_coeff, tuple):
        Ca, Cb = mf.mo_coeff
        Occa, Occb = mf.mo_occ
        Fa, Fb = mf.get_fock()
        
        # Average alpha and beta to form spatial representation
        F_N = 0.5 * (Fa + Fb)
        mo_occ = 0.5 * (Occa + Occb)
        _, mo_coeff = scipy.linalg.eigh(F_N, S)
        
        dm_a, dm_b = mf.make_rdm1()
        dm_total = dm_a + dm_b
    else:
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        F_N = mf.get_fock()
        if np.ndim(F_N) == 3:
            F_N = 0.5 * (F_N[0] + F_N[1])
        dm_total = mf.make_rdm1(mo_coeff, mo_occ)

    occ_idx = np.where(mo_occ > 0)[0]
    virt_idx = np.where(mo_occ == 0)[0]

    C_occ = mo_coeff[:, occ_idx]
    C_virt = mo_coeff[:, virt_idx]

    # Count occupied orbitals per subshell base
    dummy_energies = np.diag(C_occ.T @ F_N @ C_occ)
    occ_labels = assign_physical_labels(mol, C_occ, dummy_energies, occ=mo_occ[occ_idx])
    occ_subshell_counts = {}
    for lbl in occ_labels:
        base = extract_subshell_base(lbl)
        occ_subshell_counts[base] = occ_subshell_counts.get(base, 0) + 1

    needed_virtuals = get_needed_virtual_subshells(symbol, z, occ_subshell_counts)
    logging.info(f"[{symbol}] Target unoccupied valence virtual requirements: {needed_virtuals}")

    # Return occupied orbitals alone for Noble Gases or fully closed valence shells
    if not needed_virtuals or len(virt_idx) == 0:
        logging.info(f"[{symbol}] No virtual orbitals targeted. Canonicalizing occupied space only.")
        F_occ_sub = C_occ.T @ F_N @ C_occ
        e_occ_phys, U_occ = np.linalg.eigh(F_occ_sub)
        C_occ_phys = C_occ @ U_occ
        return C_occ_phys, e_occ_phys, mo_occ[occ_idx], 0

    logging.info(f"[{symbol}] Building V_N-1 cation density matrix...")
    occ_vnminus1 = mo_occ.copy()
    homo_idx = occ_idx[-1]
    occ_vnminus1[homo_idx] = max(0.0, occ_vnminus1[homo_idx] - 1.0)

    if np.sum(mo_occ) > 0:
        dm_vnminus1 = dm_total * (np.sum(occ_vnminus1) / np.sum(mo_occ))
    else:
        dm_vnminus1 = np.zeros_like(dm_total)

    # Build V_{N-1} Fock operator using the spatial cation density matrix
    mf_dummy = dft.RKS(mol)
    F_vnminus1 = mf_dummy.get_fock(dm=dm_vnminus1)
    if np.ndim(F_vnminus1) == 3:
        F_vnminus1 = 0.5 * (F_vnminus1[0] + F_vnminus1[1])

    logging.info(f"[{symbol}] Diagonalizing V_N-1 Fock operator in virtual space...")
    F_virt_mvo = C_virt.T @ F_vnminus1 @ C_virt
    e_mvo, U_mvo = np.linalg.eigh(F_virt_mvo)
    C_mvo_virt = C_virt @ U_mvo

    ao_l = []
    for _, _, nl, _ in mol.ao_labels(fmt=False):
        ao_l.append(L_MAP[nl[-1]])
    ao_l = np.array(ao_l)

    pop_virt = C_mvo_virt * (S @ C_mvo_virt)
    mvo_l = np.array([
        np.argmax([pop_virt[ao_l == l, i].sum() for l in range(6)])
        for i in range(C_mvo_virt.shape[1])
    ])

    selected_virt_indices = []
    for l_val, n_needed in needed_virtuals:
        l_indices = np.where(mvo_l == l_val)[0]
        if len(l_indices) >= n_needed:
            selected_virt_indices.extend(l_indices[:n_needed])

    selected_virt_indices = np.array(selected_virt_indices, dtype=int)
    C_mvo_selected = C_mvo_virt[:, selected_virt_indices]

    logging.info(f"[{symbol}] Performing neutral Fock subspace canonicalization...")
    F_occ_sub = C_occ.T @ F_N @ C_occ
    e_occ_phys, U_occ = np.linalg.eigh(F_occ_sub)
    C_occ_phys = C_occ @ U_occ

    if C_mvo_selected.shape[1] > 0:
        F_virt_sub = C_mvo_selected.T @ F_N @ C_mvo_selected
        e_virt_phys, U_virt = np.linalg.eigh(F_virt_sub)
        C_virt_phys = C_mvo_selected @ U_virt
    else:
        C_virt_phys = np.empty((C_occ.shape[0], 0))
        e_virt_phys = np.array([])

    C_final = np.column_stack([C_occ_phys, C_virt_phys])
    E_final = np.concatenate([e_occ_phys, e_virt_phys])
    Occ_final = np.concatenate([mo_occ[occ_idx], np.zeros(C_virt_phys.shape[1])])

    return C_final, E_final, Occ_final, C_virt_phys.shape[1]


# ==============================================================================
# SECTION 5: Per-Atom Workflow Execution Driver
# ==============================================================================

def run_mvo_subshell_atom(symbol: str, subfolder: Path):
    """Executes robust UKS/RKS DFT + MVO workflow with ADIIS and SOSCF fallback."""
    elem = get_mendeleev_element(symbol)
    z = elem.atomic_number
    spin = elem.ec.unpaired_electrons()
    ncore_noble = get_noble_core_orbitals(z)

    logging.info(f"[{symbol}] Initializing atomic calculation (Z={z}, Spin Multiplicity={spin+1})...")
    try:
        mol = gto.M(
            atom=f"{symbol} 0 0 0",
            spin=spin,
            basis=BASIS,
            symmetry=True,
            max_memory=MAX_MEMORY_MB,
            verbose=3,
        )
    except Exception:
        logging.warning(f"[{symbol}] Basis {BASIS} failed to initialize. Falling back to cc-pVTZ-DK...")
        mol = gto.M(
            atom=f"{symbol} 0 0 0",
            spin=spin,
            basis="cc-pVTZ-DK",
            symmetry=True,
            max_memory=MAX_MEMORY_MB,
            verbose=3,
        )

    # --- Pass 1 DFT (UKS/RKS + ADIIS + Level Shift for Open-Shell Transition Metals) ---
    logging.info(f"[{symbol}] Starting Pass 1 DFT (ADIIS + Level Shift SCF)...")
    mf_pre = (dft.UKS if spin > 0 else dft.RKS)(mol).x2c()
    mf_pre.xc = "CAM-B3LYP"
    mf_pre.level_shift = 0.5  # Higher level shift suppresses 3d/4s level crossing oscillations
    mf_pre.damp = 0.3
    mf_pre.DIIS = scf.ADIIS   # ADIIS is vastly superior for atomic transition metal open shells
    mf_pre.conv_tol = 1e-6
    mf_pre.max_cycle = 150
    mf_pre.verbose = 4
    mf_pre.kernel()

    dm_init = mf_pre.make_rdm1() if mf_pre.converged else None
    if not mf_pre.converged:
        logging.warning(f"[{symbol}] Pass 1 DFT did not converge strictly; proceeding with generated density guess.")

    # --- Pass 2 DFT (High-Precision SCF with ADIIS) ---
    logging.info(f"[{symbol}] Starting Pass 2 DFT (High-precision ADIIS SCF)...")
    mf = (dft.UKS if spin > 0 else dft.RKS)(mol).x2c()
    mf.xc = "CAM-B3LYP"
    mf.DIIS = scf.ADIIS
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.verbose = 4
    mf.kernel(dm0=dm_init)

    # --- SOSCF Fallback: If standard SCF oscillates/fails, invoke Newton-Raphson solver ---
    if not mf.converged:
        logging.warning(f"[{symbol}] Standard SCF failed to converge. Switching to SOSCF (Newton-Raphson solver)...")
        mf_n = mf.newton()
        mf_n.conv_tol = 1e-9
        mf_n.max_cycle = 100
        mf_n.verbose = 4
        mf_n.kernel(dm0=mf.make_rdm1())
        if mf_n.converged:
            mf = mf_n
            logging.info(f"[{symbol}] SOSCF converged successfully.")
        else:
            raise RuntimeError(f"X2C DFT calculation and SOSCF fallback failed to converge for {symbol}.")

    # --- Compute MVOs and Canonicalize ---
    logging.info(f"[{symbol}] DFT converged successfully (Total Energy: {mf.e_tot:.8f} Ha). Computing MVOs...")
    mo_coeff, mo_energy, mo_occ, n_virt_retained = compute_mvo_subshell_orbitals(
        mol, mf, symbol, z
    )

    # --- Assign Labels, Calculate <V_nuc>, and Symmetrize ---
    logging.info(f"[{symbol}] Assigning physical labels and computing expectation values...")
    raw_labels = assign_physical_labels(mol, mo_coeff, mo_energy, occ=mo_occ)
    v_nuc_raw = compute_vnuc_expectation(mol, mo_coeff)

    labels = raw_labels
    occ = symmetrize_subshell_values(mo_occ, labels)
    v_nuc = symmetrize_subshell_values(v_nuc_raw, labels)
    energies = symmetrize_subshell_values(mo_energy, labels)

    # --- Terminal Output Summary ---
    print("\n" + "=" * 85)
    print("   X2C CAM-B3LYP + VALENCE VIRTUAL MVO (NEUTRAL FOCK CANONICALIZED)")
    print("=" * 85)
    print(
        f"Element: {symbol} (Z={z}) | Basis: {mol.basis} | Total Basis Functions:"
        f" {len(mo_coeff)}"
    )
    print(
        f"Noble Core: {ncore_noble} orbs | Retained Valence Virtuals:"
        f" {n_virt_retained}"
    )
    print("-" * 85)
    print(
        f"{'Index':<6} {'Orbital':<10} {'Occupation':<14} {'<V_nuc> (Ha)':<14}"
        f" {'Energy (Eh)':<14} {'Energy (eV)':<14}"
    )
    print("-" * 85)

    for i, (lbl, o, v, e) in enumerate(zip(labels, occ, v_nuc, energies)):
        print(
            f"{i+1:<6d} {lbl:<10s} {o:<14.6f} {v:<14.4f} {e:<14.4f}"
            f" {e*HARTREE2EV:<14.2f}"
        )
    print("=" * 85 + "\n")

    # --- Export Results Summary & Arrays ---
    results_file = subfolder / "results.txt"
    with open(results_file, "w") as f:
        f.write(f"Symbol: {symbol} (Z={z})\n")
        f.write(f"Basis Set: {mol.basis}\n")
        f.write("Method: Valence Virtual V_{N-1} MVO (Neutral Fock Canonicalized)\n")
        f.write(f"DFT Energy: {mf.e_tot:.8f} Ha\n")
        f.write(f"Noble Core Orbitals: {ncore_noble}\n")
        f.write(f"Retained Valence Virtuals: {n_virt_retained}\n\n")
        f.write(
            f"{'Index':<6} {'Orbital':<10} {'Occupation':<14} {'<V_nuc> (Ha)':<14}"
            f" {'Energy (Eh)':<14} {'Energy (eV)':<14}\n"
        )
        f.write("-" * 80 + "\n")

        for i, (lbl, o, v, e) in enumerate(zip(labels, occ, v_nuc, energies)):
            f.write(
                f"{i+1:<6d} {lbl:<10s} {o:<14.6f} {v:<14.4f} {e:<14.4f}"
                f" {e*HARTREE2EV:<14.2f}\n"
            )

    np.save(subfolder / "retained_mvo_orbs.npy", mo_coeff)
    return mf.e_tot, len(mo_occ), mol.basis


# ==============================================================================
# SECTION 6: Main Entry Point & Processing Loop with Error Logging
# ==============================================================================

if __name__ == "__main__":
    folder = Path(".")

    for idx in range(START_IDX, END_IDX):
        symbol = get_mendeleev_element(idx).symbol
        subfolder = folder / symbol
        subfolder.mkdir(exist_ok=True)

        if (subfolder / "SUCCEEDED").exists():
            logging.info(f"Skipping Element: {symbol} (Already completed)")
            continue

        logging.info(f"{'=' * 60}\nProcessing Element: {symbol} (Z={idx})\n{'=' * 60}")
        update_status(subfolder, "RUNNING")

        try:
            e_tot_dft, n_retained, used_basis = run_mvo_subshell_atom(
                symbol, subfolder
            )
            logging.info(
                f"SUCCESS [{symbol}]: X2C DFT [{used_basis}] Total Energy ="
                f" {e_tot_dft:.8f} Ha ({n_retained} orbitals retained)"
            )
            update_status(subfolder, "SUCCEEDED")
        except Exception as exc:
            error_trace = traceback.format_exc()
            logging.error(f"FAILED [{symbol}]: {exc}\n{error_trace}")
            update_status(subfolder, "FAILED", err_msg=error_trace)