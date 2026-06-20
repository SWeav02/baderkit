# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
import numpy as np

from baderkit.toolkit import Structure

# TODO:
    # Ensure structure is read in correctly
    # Double check units and data ordering

# Extended conversion factors
BOHR_TO_ANG    = 0.529177249
RY_TO_EV       = 13.605826
HARTREE_TO_EV  = 2.0 * RY_TO_EV

def read_qe(prefix_save_dir):
    """
    Parses data-file-schema.xml inside prefix.save/ to extract metadata,
    band records, and pymatgen-compatible structural information.
    """
    xml_path = os.path.join(prefix_save_dir, "data-file-schema.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Could not find XML schema file at {xml_path}")
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # -------------------------------------------------------------------------
    # 1. STRUCTURE EXTRACTION (Pymatgen Compatible)
    # -------------------------------------------------------------------------
    atomic_structure = root.find(".//atomic_structure")
    cell_node = atomic_structure.find("cell")
    
    a1 = np.fromstring(cell_node.find("a1").text, sep=" ") * BOHR_TO_ANG
    a2 = np.fromstring(cell_node.find("a2").text, sep=" ") * BOHR_TO_ANG
    a3 = np.fromstring(cell_node.find("a3").text, sep=" ") * BOHR_TO_ANG
    lattice = np.array([a1, a2, a3])
    
    atoms = atomic_structure.findall(".//atomic_positions/atom")
    species = []
    coords = []
    for atom in atoms:
        symbol = atom.attrib.get("name")
        pos = np.fromstring(atom.text, sep=" ") * BOHR_TO_ANG
        species.append(symbol)
        coords.append(pos)
        
    structure = Structure(
        lattice=lattice, 
        species=species, 
        frac_coords=coords)

    # -------------------------------------------------------------------------
    # 2. ENCUT & EFERMI EXTRACTION
    # -------------------------------------------------------------------------
    ecutwfc_node = root.find(".//ecutwfc")
    encut = float(ecutwfc_node.text) * HARTREE_TO_EV if ecutwfc_node is not None else 0.0
    
    efermi_node = root.find(".//efermi")
    efermi = float(efermi_node.text) * HARTREE_TO_EV if efermi_node is not None else None

    # -------------------------------------------------------------------------
    # 3. SPIN AND BAND METADATA
    # -------------------------------------------------------------------------
    band_structure = root.find(".//band_structure")
    lsda = band_structure.find("lsda").text.lower() == "true" if band_structure.find("lsda") is not None else False
    noncolin = band_structure.find("noncolin").text.lower() == "true" if band_structure.find("noncolin") is not None else False
    
    nspin = 2 if lsda else 1
    ks_energies_nodes = band_structure.findall("ks_energies")
    
    total_blocks = len(ks_energies_nodes)
    nkpts = total_blocks // 2 if lsda else total_blocks
    
    first_eig_text = ks_energies_nodes[0].find("eigenvalues").text
    nbands = len(np.fromstring(first_eig_text, sep=" "))
    
    kpoint_frac = np.zeros((nkpts, 3), dtype=float)
    bands = np.zeros((nspin, nkpts, nbands), dtype=float)
    occs = np.zeros((nspin, nkpts, nbands), dtype=float)
    
    for idx, ks_node in enumerate(ks_energies_nodes):
        kp_coord = np.fromstring(ks_node.find("k_point").text, sep=" ")
        
        if lsda:
            ispin_idx = 0 if idx < nkpts else 1
            ikpt_idx = idx % nkpts
        else:
            ispin_idx = 0
            ikpt_idx = idx
            
        if ispin_idx == 0:
            kpoint_frac[ikpt_idx] = kp_coord
            
        eig_vals = np.fromstring(ks_node.find("eigenvalues").text, sep=" ") * HARTREE_TO_EV
        occ_vals = np.fromstring(ks_node.find("occupations").text, sep=" ")
        
        bands[ispin_idx, ikpt_idx, :] = eig_vals
        occs[ispin_idx, ikpt_idx, :] = occ_vals

    # Complete configuration passed downstream to the coefficient parser
    qe_dict = {
        "prefix_save_dir": prefix_save_dir,
        "nspin": nspin,
        "nkpts": nkpts,
        "nbands": nbands,
        "lsda": lsda,
        "noncolin": noncolin
    }

    return structure, kpoint_frac, occs, bands, encut, efermi, qe_dict

def read_qe_coefficients(
        ispin, 
        ikpt, 
        iband,
        prefix_save_dir,
        nbands,
        **kwargs
        ):
    """
    Extracts the plane-wave coefficients for a single target state.
    Automatically handles HDF5 datasets and legacy Fortran unformatted binaries.
    """
    kpt_num = ikpt + 1
    kpt_dir = os.path.join(prefix_save_dir, f"K{kpt_num:05d}")
    
    # -------------------------------------------------------------------------
    # CASE A: HDF5 COMPILATION TARGETS
    # -------------------------------------------------------------------------
    possible_hdf5 = [
        os.path.join(prefix_save_dir, f"wfc{kpt_num}.hdf5"),
        os.path.join(kpt_dir, "wfc.hdf5")
    ]
    
    hdf5_file = None
    for p in possible_hdf5:
        if os.path.exists(p):
            hdf5_file = p
            break

    if hdf5_file is not None:
        import h5py
        with h5py.File(hdf5_file, "r") as h5:
            ds_name = f"evc{ispin + 1}" if f"evc{ispin + 1}" in h5 else "evc"
            if ds_name in h5:
                dataset = h5[ds_name]
                if dataset.shape[0] == nbands:
                    return np.array(dataset[iband, :])
                else:
                    return np.array(dataset[:, iband])

    # -------------------------------------------------------------------------
    # CASE B: FORTRAN UNFORMATTED DATA STREAM TARGETS
    # -------------------------------------------------------------------------
    possible_filenames = [
        os.path.join(prefix_save_dir, f"wfc{kpt_num}.dat"),
        os.path.join(kpt_dir, "wfc.dat"),
        os.path.join(kpt_dir, f"wfc{ispin + 1}.dat"),
        os.path.join(kpt_dir, f"wfc{ispin}.dat")
    ]
    
    binary_file = None
    for p in possible_filenames:
        if os.path.exists(p):
            binary_file = p
            break
            
    if binary_file is not None:
        with open(binary_file, "rb") as f:
            # Record 1: Context Header
            rec_len1 = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(rec_len1, 1)
            f.seek(4, 1)
            
            # Record 2: Dimensions
            f.seek(4, 1)  # Skip leading record delimiter
            f.seek(4, 1)  # Skip ngw
            igwx = np.fromfile(f, dtype=np.int32, count=1)[0]
            npol = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(4, 1)  # Skip nbands_file
            f.seek(4, 1)  # Skip trailing record delimiter
            
            # Record 3: Reciprocal Lattice Matrices
            rec_len3 = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(rec_len3, 1)
            f.seek(4, 1)
            
            # Record 4: 'mill' Index Maps
            rec_len4 = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(rec_len4, 1)
            f.seek(4, 1)
            
            # Record 5+: Standalone state records
            bytes_per_band_record = 4 + (npol * igwx * 16) + 4
            
            # Direct seek to target band block
            f.seek(iband * bytes_per_band_record, 1)
            f.seek(4, 1)  # Skip leading band delimiter
            
            return np.fromfile(f, dtype=np.complex128, count=npol * igwx)

    raise FileNotFoundError(
        f"Could not locate valid wavefunctions for k-point {kpt_num} in '{prefix_save_dir}'."
    )