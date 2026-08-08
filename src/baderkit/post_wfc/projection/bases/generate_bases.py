"""This script generates the minimal atomic orbital basis sets used for

generating Bloch IAOs. We choose to include extra unoccupied subshells up to
the next period, as these orbitals are commonly involved in bonding. Thus we
believe they are necessary to form the proper depolarized reference state
for IAOs. This is techhnically arbitrary and we provide this script as a
template for those who want to generate their own basis set.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import re
import traceback

from mendeleev import element as get_mendeleev_element
import numpy as np
from pyscf import dft, gto, lib, scf
from pyscf.data.nist import HARTREE2EV
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback
import scipy.linalg
from scipy.special import gamma

# ==============================================================================
# SECTION 1: Global Configurations & Physical Conversion Constants
# ==============================================================================

# Install rich exception handler for beautiful, detailed tracebacks
install_rich_traceback(show_locals=False)
console = Console()

# Execution range for atomic calculations (Z=1 to Z=92; Hydrogen to Uranium)
START_IDX = 1
END_IDX = 93

# Default relativistic all-electron basis set choice
BASIS = "dyall-ae3z"

# Hardware memory limit allocation for PySCF numerical routines (16 GB)
MAX_MEMORY_MB = 16000

# Atomic numbers corresponding to closed-shell Noble Gases (used as core cutoffs)
NOBLE_GASES = {2, 10, 18, 36, 54, 86}

# Physical Conversion Constants for unit alignment across atomic/angstrom standards
HARTREE_TO_EV = 27.211386245988  # Convert energy from Atomic Units (Hartree) to eV
BOHR_TO_ANG = 0.5291772109  # Convert length from Bohr atomic units to Angstroms
BOHR_SQ = (
    BOHR_TO_ANG**2
)  # Square Bohr-to-Angstrom conversion factor for exponents

# Configure logging to use RichHandler with colored levels and structured formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
    ],
)

# Configure PySCF global CPU threading limit and memory pool limits
lib.num_threads(lib.num_threads())
lib.param.MAX_MEMORY = MAX_MEMORY_MB

# Map orbital string designations to integer angular momentum quantum numbers (l)
L_MAP = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


# ==============================================================================
# SECTION 2: Labeling, Symmetry & Ground-State Validation Helpers
# ==============================================================================


def get_noble_core_orbitals(z: int) -> int:
    """Determines the number of spatial core orbitals corresponding to the nearest
    
    underlying noble gas configuration.
    """
    return next(c for c in [86, 54, 36, 18, 10, 2, 0] if z > c) // 2


def extract_subshell_base(label: str) -> str:
    """Strips directional magnetic components from orbital string labels.
    
    Example: '4px' -> '4p', '3dz2' -> '3d'.
    """
    m = re.match(r"(\d+[spdfgh])", label)
    return m.group(1) if m else label


def assign_physical_labels(mol, mo_coeff, energies, occ=None) -> list:
    """Assigns quantum labels (1s, 2s, 2p, 3s...) to molecular orbitals by
    
    analyzing Mulliken population projections and ordering by Fock energy.
    """
    # Step 1: Extract AO l quantum numbers and components
    ao_l, ao_comp = [], []
    for _, _, nl, comp in mol.ao_labels(fmt=False):
      l_char = nl[-1]
      ao_l.append(L_MAP[l_char])
      ao_comp.append((l_char, comp))
    
    ao_l = np.array(ao_l)
    
    # Step 2: Compute Mulliken population matrix P = C * (S * C)
    S = mol.intor_symmetric("int1e_ovlp")
    pop = mo_coeff * (S @ mo_coeff)
    
    # Step 3: Find dominant l quantum number and orbital character per MO column
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
    occ_weights = (
        np.zeros(mo_coeff.shape[1]) if occ is None else np.round(occ, 4)
    )
    
    # Step 4 & 5: Sort MOs by (occ, energy) per l-channel and assign quantum labels
    for l_val in range(6):
      idx_l = np.where(mo_l == l_val)[0]
      if len(idx_l) == 0:
        continue
    
      sort_keys = np.lexsort((energies[idx_l], -occ_weights[idx_l]))
      idx_l_sorted = idx_l[sort_keys]
    
      deg, n_curr = 2 * l_val + 1, l_val + 1
      for ptr in range(0, len(idx_l_sorted), deg):
        for idx in idx_l_sorted[ptr : min(ptr + deg, len(idx_l_sorted))]:
          labels[idx] = f"{n_curr}{mo_comp_str[idx]}"
        n_curr += 1
    
    return labels


def symmetrize_subshell_values(values, labels) -> np.ndarray:
    """Enforces spherical symmetry by averaging expectation values across
    
    degenerate (2L + 1) magnetic subshells.
    """
    subshell_groups = {}
    for idx, label in enumerate(labels):
      base_subshell = extract_subshell_base(label)
      subshell_groups.setdefault(base_subshell, []).append(idx)
    
    symmetrized_values = np.copy(values)
    for _, indices in subshell_groups.items():
      avg_val = np.mean(values[indices])
      symmetrized_values[indices] = avg_val
    
    return symmetrized_values


def _normalize_key(key) -> tuple[int, int]:
    """Converts string labels, mixed tuples, or orbital objects into standard
    
    (int n, int l) tuples.
    """
    if isinstance(key, tuple):
      n, l = key
      l_int = L_MAP[l.lower()] if isinstance(l, str) else int(l)
      return (int(n), l_int)
    
    if isinstance(key, str):
      return (int(key[:-1]), L_MAP[key[-1].lower()])
    
    l_val = getattr(key, "l")
    l_int = L_MAP[l_val.lower()] if isinstance(l_val, str) else int(l_val)
    return (int(getattr(key, "n")), l_int)


def validate_ground_state_configuration(
    element_symbol: str, unique_subshells: list, tol: float = 1e-3
) -> bool:
    """Validates that extracted occupied subshells strictly match the ground-state
    
    electron configuration provided by the Mendeleev database.
    """
    m_elem = get_mendeleev_element(element_symbol)
    expected_conf = {_normalize_key(k): float(v) for k, v in m_elem.ec.conf.items()}
    
    actual_conf = defaultdict(float)
    for subshell in unique_subshells:
      if subshell["occ"] > tol:
        actual_conf[(int(subshell["n"]), int(subshell["l"]))] += float(
            subshell["occ"]
        )
    
    mismatches = []
    
    total_expected = sum(expected_conf.values())
    total_actual = sum(actual_conf.values())
    if abs(total_expected - total_actual) > tol:
      mismatches.append(
          f"Total electron count mismatch: Expected {total_expected:.1f}, Got"
          f" {total_actual:.4f}"
      )
    
    l_symbols = dict(enumerate("spdfgh"))
    all_keys = sorted(set(expected_conf) | set(actual_conf))
    
    for n, l in all_keys:
      exp_occ = expected_conf.get((n, l), 0.0)
      act_occ = actual_conf.get((n, l), 0.0)
    
      if abs(exp_occ - act_occ) > tol:
        l_sym = l_symbols.get(l, f"l={l}")
        mismatches.append(
            f"Subshell {n}{l_sym}: Expected {exp_occ:.4f} e-, Got {act_occ:.4f} e-"
        )
    
    if mismatches:
      raise ValueError(
          f"Ground State Configuration Validation FAILED for [{element_symbol}]:\n"
          + "\n".join(f"  - {m}" for m in mismatches)
      )
    
    logging.info(
        f"[bold green][{element_symbol}][/bold green] Configuration Validation"
        " PASSED: Ground-state subshells verified."
    )
    return True


# ==============================================================================
# SECTION 3: Valence Subshell Targeter
# ==============================================================================


def get_virtual_subshells(
    symbol: str, z: int, occ_subshell_counts: dict
) -> list:
    """Determines required virtual orbital counts per l-symmetry to complete all
    
    partially filled or unpopulated valence subshells for a given element.
    """
    elem = get_mendeleev_element(symbol)
    
    if elem.group_id == 18:
      return []
    
    P = elem.period
    z = z or elem.atomic_number
    
    if P == 1:
      subshells = ["1s"]
    elif P in (2, 3):
      subshells = [f"{P}s", f"{P}p"] + (["3d"] if P == 3 else [])
    elif P in (4, 5):
      subshells = [f"{P}s", f"{P-1}d", f"{P}p"]
    elif P in (6, 7):
      has_f = z >= (57 if P == 6 else 89)
      subshells = (
          [f"{P}s"] + ([f"{P-2}f"] if has_f else []) + [f"{P-1}d", f"{P}p"]
      )
    
    needed = []
    for label in subshells:
      l_val = L_MAP[label[-1]]
      deg = 2 * l_val + 1
      n_occ = occ_subshell_counts.get(label, 0)
    
      if deg > n_occ:
        needed.append((l_val, deg - n_occ))
    
    return needed


# ==============================================================================
# SECTION 4: Modified Virtual Orbital (MVO) & Subspace Canonicalization
# ==============================================================================


def compute_mvo_subshell_orbitals(mol, mf, symbol: str, z: int):
    """Generates V_{N-1} MVOs to complete incomplete valence subshells and
    
    canonicalizes orbitals under the unperturbed neutral Fock operator F^(N).
    """
    logging.info(f"[{symbol}] Constructing neutral Fock matrix F^(N)...")
    
    S = mol.intor_symmetric("int1e_ovlp")
    
    if isinstance(mf.mo_coeff, tuple):
      Ca, Cb = mf.mo_coeff
      Occa, Occb = mf.mo_occ
      Fa, Fb = mf.get_fock()
    
      F_N = 0.5 * (Fa + Fb)
      mo_occ = Occa + Occb  # Total spatial orbital occupancy
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
    
    dummy_energies = np.diag(C_occ.T @ F_N @ C_occ)
    occ_labels = assign_physical_labels(
        mol, C_occ, dummy_energies, occ=mo_occ[occ_idx]
    )
    occ_subshell_counts = {}
    for lbl in occ_labels:
      base = extract_subshell_base(lbl)
      occ_subshell_counts[base] = occ_subshell_counts.get(base, 0) + 1
    
    needed_virtuals = get_virtual_subshells(symbol, z, occ_subshell_counts)
    logging.info(
        f"[{symbol}] Target unoccupied valence virtual requirements:"
        f" {needed_virtuals}"
    )
    
    if not needed_virtuals or len(virt_idx) == 0:
      logging.info(
          f"[{symbol}] No virtual orbitals targeted. Canonicalizing occupied"
          " space only."
      )
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
    
    mf_dummy = dft.RKS(mol)
    F_vnminus1 = mf_dummy.get_fock(dm=dm_vnminus1)
    if np.ndim(F_vnminus1) == 3:
      F_vnminus1 = 0.5 * (F_vnminus1[0] + F_vnminus1[1])
    
    logging.info(
        f"[{symbol}] Diagonalizing V_N-1 Fock operator in virtual space..."
    )
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
    
    logging.info(
        f"[{symbol}] Performing neutral Fock subspace canonicalization..."
    )
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
    Occ_final = np.concatenate(
        [mo_occ[occ_idx], np.zeros(C_virt_phys.shape[1])]
    )
    
    return C_final, E_final, Occ_final, C_virt_phys.shape[1]


# ==============================================================================
# SECTION 5: Radial Basis Compression & NPZ Exporter
# ==============================================================================


def export_radial_basis_npz(
    mol,
    mo_coeff,
    mo_energy,
    mo_occ,
    symbol: str,
    subfolder: Path,
    functional: str = "CAM-B3LYP",
) -> Path:
    """Compresses 3D atomic orbitals into spherically symmetric 1D radial
    
    wavefunctions and exports binary NPZ basis footprints matching the schema of
    generate_bases.py.
    """
    # Step 1: Build AO basis function mapping to contraction components (l, p, m)
    ao_map = []
    l_counts = {}
    ao_idx = 0
    
    for bas_id in range(mol.nbas):
      l = mol.bas_angular(bas_id)
      nctr = mol.bas_nctr(bas_id)
      start_p = l_counts.get(l, 0)
    
      for c in range(nctr):
        p = start_p + c
        for m in range(2 * l + 1):
          ao_map.append((ao_idx, l, p, m))
          ao_idx += 1
      l_counts[l] = start_p + nctr
    
    # Step 2: Assign physical labels and symmetrize occupancies/energies
    labels = assign_physical_labels(mol, mo_coeff, mo_energy, occ=mo_occ)
    symmetrized_occ = symmetrize_subshell_values(mo_occ, labels)
    symmetrized_energies = symmetrize_subshell_values(mo_energy, labels)
    
    # Step 3: Group degenerate magnetic states into subshells by base label (e.g. '1s', '2p', '3d')
    subshell_groups = defaultdict(list)
    for iband, lbl in enumerate(labels):
      base_lbl = extract_subshell_base(lbl)
      l_char = base_lbl[-1]
      l_val = L_MAP[l_char]
      n_val = int(base_lbl[:-1])
    
      state_info = {
          "energy": float(symmetrized_energies[iband] * HARTREE_TO_EV),
          "coeff": mo_coeff[:, iband],
          "occ": float(symmetrized_occ[iband]),
          "dominant_l": l_val,
          "n": n_val,
          "base_label": base_lbl,
      }
      subshell_groups[base_lbl].append(state_info)
    
    unique_subshells = []
    for base_lbl, states in subshell_groups.items():
      base_l = states[0]["dominant_l"]
      base_n = states[0]["n"]
      avg_energy = states[0]["energy"]
      total_occ = float(np.sum([s["occ"] for s in states]))
    
      unique_subshells.append({
          "energy": avg_energy,
          "l": base_l,
          "n": base_n,
          "occ": total_occ,
          "states": states,
          "c_p_pure": None,
          "base_label": base_lbl,
      })
    
    unique_subshells.sort(key=lambda x: x["energy"])
    
    # Step 4: Validate extracted ground-state electron configuration
    validate_ground_state_configuration(symbol, unique_subshells)
    
    # Step 5: PASS 1 - Extract pure radial profile c_p via SVD across magnetic components
    for subshell in unique_subshells:
      state_l = subshell["l"]
      n_contract = l_counts[state_l]
    
      feature_vectors = []
      for state in subshell["states"]:
        coeff = state["coeff"]
        matrix_l = np.zeros((n_contract, 2 * state_l + 1))
        for idx_ao, ao_l, p, ao_m in ao_map:
          if ao_l == state_l:
            matrix_l[p, ao_m] = coeff[idx_ao]
        for m in range(2 * state_l + 1):
          feature_vectors.append(matrix_l[:, m])
    
      feature_matrix = np.column_stack(feature_vectors)
      U, S, Vt = np.linalg.svd(feature_matrix, full_matrices=False)
      c_p_pure = U[:, 0]
    
      # Enforce positive sign convention on dominant radial peak
      if np.sum(c_p_pure) < 0:
        c_p_pure = -c_p_pure
      subshell["c_p_pure"] = c_p_pure
    
    # Step 6: PASS 2 - Analytical contracted AO overlap matrix generation
    contracted_defs = {}
    for bas_id in range(mol.nbas):
      l = mol.bas_angular(bas_id)
      if l not in contracted_defs:
        contracted_defs[l] = []
      exps_bohr = mol.bas_exp(bas_id)
      exps_ang = exps_bohr / BOHR_SQ
      coeffs_mat = mol.bas_ctr_coeff(bas_id)
      gto_norms = np.array(
          [mol.gto_norm(l, alpha) for alpha in exps_bohr], dtype=np.float64
      )
      normalized_coeffs_mat = (
          coeffs_mat * gto_norms[:, np.newaxis] * (BOHR_TO_ANG ** -(l + 1.5))
      )
    
      nctr = normalized_coeffs_mat.shape[1]
      for c in range(nctr):
        contracted_defs[l].append({
            "exps": exps_ang,
            "coeffs": normalized_coeffs_mat[:, c],
        })
    
    S_contracted_blocks = {}
    for l, def_list in contracted_defs.items():
      n_contract = len(def_list)
      S_mat = np.zeros((n_contract, n_contract), dtype=np.float64)
      gamma_factor = gamma(l + 1.5)
    
      for p in range(n_contract):
        def_p = def_list[p]
        for q in range(n_contract):
          def_q = def_list[q]
          s_val = 0.0
          for k in range(len(def_p["exps"])):
            for j in range(len(def_q["exps"])):
              A = def_p["exps"][k] + def_q["exps"][j]
              integral = 0.5 * (A ** -(l + 1.5)) * gamma_factor
              s_val += def_p["coeffs"][k] * def_q["coeffs"][j] * integral
          S_mat[p, q] = s_val
      S_contracted_blocks[l] = S_mat
    
    # Step 7: PASS 3 - Energy-ordered radial Gram-Schmidt orthogonalization
    final_retained_subshells = []
    
    for l_channel in sorted(l_counts.keys()):
      S_metric = S_contracted_blocks[l_channel]
      l_subshells = [s for s in unique_subshells if s["l"] == l_channel]
    
      orthogonalized_vectors = []
      for subshell in l_subshells:
        v = subshell["c_p_pure"].copy()
    
        # Subtract projections onto previously orthogonalized subshells
        for u in orthogonalized_vectors:
          proj = float(u.T @ S_metric @ v) / float(u.T @ S_metric @ u)
          v -= proj * u
    
        norm = np.sqrt(float(v.T @ S_metric @ v))
    
        # Retain non-singular linearly independent radial wavefunctions
        if norm > 1e-5:
          v /= norm
          orthogonalized_vectors.append(v)
          subshell["c_p_pure"] = v
          final_retained_subshells.append(subshell)
        else:
          logging.info(
              f"Skipping redundant virtual subshell: Element {symbol},"
              f" l={l_channel}, Energy={subshell['energy']:.2f} eV (Linear"
              " Singularity)"
          )
    
    unique_subshells = final_retained_subshells
    unique_subshells.sort(key=lambda x: x["energy"])
    
    # Step 8: PASS 4 - Pack linear 1D state vectors into 2D block matrix layout
    num_subshells = len(unique_subshells)
    flat_vector_size = sum(l_counts.values())
    
    energies = np.zeros(num_subshells, dtype=np.float64)
    occupancies = np.zeros(num_subshells, dtype=np.float64)
    angular_momenta = np.zeros(num_subshells, dtype=np.int_)
    principal_quantum_numbers = np.zeros(num_subshells, dtype=np.int_)
    packed_state_vectors = np.zeros(
        (num_subshells, flat_vector_size), dtype=np.float64
    )
    
    for ishell, subshell in enumerate(unique_subshells):
      state_l = subshell["l"]
      energies[ishell] = subshell["energy"]
      occupancies[ishell] = subshell["occ"]
      angular_momenta[ishell] = state_l
      principal_quantum_numbers[ishell] = subshell["n"]
    
      state_vector = []
      for l in sorted(l_counts.keys()):
        mat_dim = l_counts[l]
        c_p_vector = np.zeros(mat_dim, dtype=np.float64)
        if l == state_l:
          c_p_vector[:] = subshell["c_p_pure"]
        state_vector.extend(c_p_vector.tolist())
    
      packed_state_vectors[ishell, :] = state_vector
    
    # Step 9: Build primitive metadata scaled to Angstrom units and export NPZ binary
    basis_primitives = {}
    for bas_id in range(mol.nbas):
      l = mol.bas_angular(bas_id)
      l_str = str(l)
      if l_str not in basis_primitives:
        basis_primitives[l_str] = []
    
      exps_bohr = mol.bas_exp(bas_id)
      exps_ang = exps_bohr / BOHR_SQ
    
      coeffs_mat = mol.bas_ctr_coeff(bas_id)
      gto_norms = np.array(
          [mol.gto_norm(l, alpha) for alpha in exps_bohr], dtype=np.float64
      )
    
      absolute_coeffs_bohr = coeffs_mat * gto_norms[:, np.newaxis]
      normalized_coeffs_mat = absolute_coeffs_bohr * (BOHR_TO_ANG ** -(l + 1.5))
    
      g_prefactors = (np.pi / exps_ang) ** 1.5 * (1.0 / (2.0 * exps_ang)) ** l
      g_coeffs_mat = normalized_coeffs_mat * g_prefactors[:, np.newaxis]
    
      basis_primitives[l_str].append({
          "exponents": exps_ang.tolist(),
          "coefficients": normalized_coeffs_mat.tolist(),
          "g_coefficients": g_coeffs_mat.tolist(),
      })
    
    metadata = {
        "file_format": "SphericalRadialWavefunction_LinearVector_NPZ",
        "element": symbol,
        "functional": functional,
        "matrix_layout_dimensions": {str(l): n for l, n in l_counts.items()},
        "basis_primitives": basis_primitives,
        "basis": mol.basis,
        "units": {"energy": "eV", "length": "Angstrom"},
    }
    
    output_path = subfolder / f"{symbol}.npz"
    np.savez_compressed(
        output_path,
        energies=energies,
        occupancies=occupancies,
        angular_momenta=angular_momenta,
        principal_quantum_numbers=principal_quantum_numbers,
        packed_state_vectors=packed_state_vectors,
        metadata=json.dumps(metadata, indent=2),
    )
    logging.info(
        f"[{symbol}] Successfully generated NPZ basis footprint: {output_path}"
    )
    return output_path


# ==============================================================================
# SECTION 6: Per-Atom Workflow Execution Driver
# ==============================================================================


def run_mvo(symbol: str, subfolder: Path):
    """Executes complete atomic workflow: PySCF initialization, X2C CAM-B3LYP DFT,
    
    MVO canonicalization, expectation value evaluations, and NPZ basis export.
    """
    elem = get_mendeleev_element(symbol)
    z = elem.atomic_number
    spin = elem.ec.unpaired_electrons()
    ncore_noble = get_noble_core_orbitals(z)
    
    logging.info(
        f"[{symbol}] Initializing atomic calculation (Z={z}, Spin"
        f" Multiplicity={spin+1})..."
    )
    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        spin=spin,
        basis=BASIS,
        symmetry=True,
        max_memory=MAX_MEMORY_MB,
        verbose=3,
    )

    # Pass 1 DFT with Thermal Smearing (prevents f-electron localization trapping)
    logging.info(
        f"[{symbol}] Starting Pass 1 DFT (ADIIS + Level Shift + Thermal"
        " Smearing SCF)..."
    )
    mf_pre = (dft.UKS if spin > 0 else dft.RKS)(mol).x2c()
    mf_pre.xc = "CAM-B3LYP"
    mf_pre.level_shift = 0.5
    mf_pre.damp = 0.3
    mf_pre.DIIS = scf.ADIIS
    mf_pre.conv_tol = 1e-5
    mf_pre.max_cycle = 150
    mf_pre.verbose = 3
    
    # Apply Fermi-Dirac smearing for f-block elements (Z >= 57) or open-shell systems
    if z >= 57 or spin > 0:
      mf_pre = scf.addons.smearing(
          mf_pre, sigma=0.01, method="fermi"
      )  # 0.01 Ha ~ 0.27 eV smearing
    
    mf_pre.kernel()
    
    dm_init = mf_pre.make_rdm1() if mf_pre.converged else None
    
    # Step 4: Pass 2 DFT (Remove smearing, high-precision tight convergence)
    logging.info(f"[{symbol}] Starting Pass 2 DFT (High-precision ADIIS SCF)...")
    mf = (dft.UKS if spin > 0 else dft.RKS)(mol).x2c()
    mf.xc = "CAM-B3LYP"
    mf.DIIS = scf.ADIIS
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.verbose = 3
    mf.kernel(dm0=dm_init)

    if not mf.converged:
      logging.warning(
          f"[{symbol}] Standard SCF failed to converge. Switching to SOSCF"
          " (Newton-Raphson solver)..."
      )
      mf_n = mf.newton()
      mf_n.conv_tol = 1e-9
      mf_n.max_cycle = 100
      mf_n.verbose = 3
      mf_n.kernel(dm0=mf.make_rdm1())
      if mf_n.converged:
        mf = mf_n
        logging.info(f"[{symbol}] SOSCF converged successfully.")
      else:
        raise RuntimeError(
            "X2C DFT calculation and SOSCF fallback failed to converge for"
            f" {symbol}."
        )
    
    logging.info(
        f"[{symbol}] DFT converged successfully (Total Energy: {mf.e_tot:.8f}"
        " Ha). Computing MVOs..."
    )
    mo_coeff, mo_energy, mo_occ, n_virt_retained = compute_mvo_subshell_orbitals(
        mol, mf, symbol, z
    )
    
    logging.info(
        f"[{symbol}] Assigning physical labels and computing expectation"
        " values..."
    )
    raw_labels = assign_physical_labels(mol, mo_coeff, mo_energy, occ=mo_occ)
    
    labels = raw_labels
    occ = symmetrize_subshell_values(mo_occ, labels)
    energies = symmetrize_subshell_values(mo_energy, labels)
    
    # Output styled table using Rich
    table = Table(
        title=(
            "X2C CAM-B3LYP + VALENCE VIRTUAL MVO (NEUTRAL FOCK CANONICALIZED)\n"
            f"Element: [bold yellow]{symbol}[/bold yellow] (Z={z}) | Basis:"
            f" {mol.basis} | Total Basis Functions: {len(mo_coeff)}\nNoble Core:"
            f" {ncore_noble} orbs | Retained Valence Virtuals: {n_virt_retained}"
        ),
        title_style="bold cyan",
        header_style="bold magenta",
        border_style="bright_blue",
    )
    
    table.add_column("Index", justify="right", style="dim")
    table.add_column("Orbital", justify="center", style="bold yellow")
    table.add_column("Occupation", justify="right", style="green")
    table.add_column("Energy (Eh)", justify="right")
    table.add_column("Energy (eV)", justify="right", style="bold green")
    
    for i, (lbl, o, e) in enumerate(zip(labels, occ, energies)):
      table.add_row(
          str(i + 1), str(lbl), f"{o:.6f}", f"{e:.4f}", f"{e * HARTREE2EV:.2f}"
      )
    
    console.print(table)
    
    export_radial_basis_npz(
        mol, mo_coeff, mo_energy, mo_occ, symbol, subfolder, functional="CAM-B3LYP"
    )

    return mf.e_tot, len(mo_occ), mol.basis


# ==============================================================================
# SECTION 7: Main Entry Point & Periodic Processing Loop
# ==============================================================================

folder = Path("dyall")

for idx in range(START_IDX, END_IDX):
    symbol = get_mendeleev_element(idx).symbol
    
    console.print()
    console.print(
        Panel(
            f"[bold cyan]Processing Element:[/bold cyan] [bold"
            f" yellow]{symbol}[/bold yellow] (Z={idx})",
            expand=False,
            border_style="cyan",
        )
    )
    
    try:
      e_tot_dft, n_retained, used_basis = run_mvo(symbol, folder)
      logging.info(
          f"[bold green]SUCCESS [{symbol}]:[/bold green] X2C DFT [{used_basis}]"
          f" Total Energy = {e_tot_dft:.8f} Ha ({n_retained} orbitals retained)"
      )
    except Exception as exc:
      error_trace = traceback.format_exc()
      logging.error(f"[bold red]FAILED [{symbol}]:[/bold red] {exc}\n{error_trace}")