# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from baderkit.toolkit import Structure
from .base import BaseWfcReader, WfcMetadata, BOHR_TO_ANG, HARTREE_TO_EV

class QeReader(BaseWfcReader):
    """
    Reader implementation to harvest planewave states out of 
    Quantum ESPRESSO's data-file-schema.xml and unformatted binary records.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_binary_file(self, ispin: int, ikpt: int) -> Path:
        """Internal helper to identify file paths for separate spin/k-point data partners."""
        kpt_num = ikpt + 1
        
        if ispin == 0:
            # check for spin up file
            binary_file = self.directory / f"wfcup{kpt_num}.dat"
            # default back to standard
            if not binary_file.exists():
                binary_file = self.directory / f"wfc{kpt_num}.dat"
        elif ispin == 1:
            binary_file = self.directory / f"wfcdw{kpt_num}.dat"
        else:
            raise ValueError(f"Invalid spin index: {ispin}")
            
        if not binary_file.exists():
            raise FileNotFoundError(
                f"No coefficients file found for state at spin {ispin}, kpoint {ikpt} in {self.directory}"
            )
        return binary_file

    def read_metadata(
        self,
        nbands=None,
        bands=None,
    ) -> WfcMetadata:
        """
        Parses data-file-schema.xml inside the target directory to extract metadata,
        band records, and structural information in standard units (eV, Angstrom).
        """
        xml_path = self.directory / "data-file-schema.xml"
        if not xml_path.exists():
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
            coords=coords,
            coords_are_cartesian=True
        )

        # -------------------------------------------------------------------------
        # 2. ENCUT & EFERMI EXTRACTION
        # -------------------------------------------------------------------------
        ecutwfc_node = root.find(".//ecutwfc")
        encut = float(ecutwfc_node.text) * HARTREE_TO_EV if ecutwfc_node is not None else 0.0
        
        efermi_node = root.find(".//efermi")
        efermi = float(efermi_node.text) * HARTREE_TO_EV if efermi_node is not None else 0.0

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
        max_nbands = len(np.fromstring(first_eig_text, sep=" "))
        
        # DYNAMIC SELF-CORRECTION: Interrogate the binary record directly to see 
        # if the written wavefunction count differs from the calculated XML manifold.
        try:
            binary_file = self._get_binary_file(ispin=0, ikpt=0)
            with open(binary_file, "rb") as f:
                rec_len1 = np.fromfile(f, dtype=np.int32, count=1)[0]
                f.seek(rec_len1, 1)
                f.seek(4, 1)

                f.seek(4, 1)  # Skip leading record delimiter
                f.seek(4, 1)  # Skip ngw
                _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # n_gvec
                _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # nspin_file
                nbands_file = np.fromfile(f, dtype=np.int32, count=1)[0]
                if 0 < nbands_file < max_nbands:
                    max_nbands = nbands_file
        except Exception:
            pass  # Fallback gracefully to full XML estimation if files are locked
        
        # Resolve selected bands based on absolute available bands
        if bands is not None:
            selected_bands = np.array(bands, dtype=np.int32)
            if np.any(selected_bands < 0) or np.any(selected_bands >= max_nbands):
                raise ValueError(f"Selected band indices must be between 0 and {max_nbands - 1}")
        elif nbands is not None:
            assert 1 <= nbands <= max_nbands, f"Invalid manual nbands: {nbands}. Must be between 1 and {max_nbands}"
            selected_bands = np.arange(nbands, dtype=np.int32)
        else:
            selected_bands = np.arange(max_nbands, dtype=np.int32)
            
        n_selected_bands = len(selected_bands)
        
        # Initialize arrays mapped to the user-selected band counts
        kpoints = np.zeros((nkpts, 3), dtype=float)
        energies = np.zeros((nspin, nkpts, n_selected_bands), dtype=float)
        occupancies = np.zeros((nspin, nkpts, n_selected_bands), dtype=float)
        
        for idx, ks_node in enumerate(ks_energies_nodes):
            kp_coord = np.fromstring(ks_node.find("k_point").text, sep=" ")
            
            if lsda:
                ispin_idx = 0 if idx < nkpts else 1
                ikpt_idx = idx % nkpts
            else:
                ispin_idx = 0
                ikpt_idx = idx
                
            if ispin_idx == 0:
                kpoints[ikpt_idx] = kp_coord
                
            eig_vals = np.fromstring(ks_node.find("eigenvalues").text, sep=" ") * HARTREE_TO_EV
            occ_vals = np.fromstring(ks_node.find("occupations").text, sep=" ")
            
            # Map eigenvalues and occupancies according to user-requested band indices
            energies[ispin_idx, ikpt_idx, :] = eig_vals[selected_bands]
            occupancies[ispin_idx, ikpt_idx, :] = occ_vals[selected_bands]

        # Populate self.meta with physical and selected dimensions
        self.meta = WfcMetadata(
            structure=structure,
            kpoints=kpoints,
            occupancies=occupancies,
            energies=energies-efermi,
            energy_cutoff=encut,
            efermi=0.0,
            original_efermi=efermi,
            nspin=nspin,
            nkpts=nkpts,
            nbands=n_selected_bands,
            max_nbands=max_nbands,
            bands=selected_bands,
            cplx_dtype=np.complex128
        )
        
        # Cache additional parameters required for unformatted binary streaming
        self._lsda = lsda
        self._noncolin = noncolin
        
        return self.meta

    def read_coefficients(self, ispin: int, ikpt: int, iband: int) -> np.ndarray:
        """Extracts the plane-wave coefficients for a single target state."""
        return self.read_coefficients_batch(ispin, ikpt, [iband])[0]

    def read_coefficients_batch(self, ispin: int, ikpt: int, bands: list) -> np.ndarray:
        """Streams multiple bands from sequential Fortran records without reopening files."""
        binary_file = self._get_binary_file(ispin, ikpt)
        
        with open(binary_file, "rb") as f:
            # Record 1: Context Header
            rec_len1 = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(rec_len1, 1)
            f.seek(4, 1)

            # Record 2: Dimensions
            f.seek(4, 1)  # Skip leading record delimiter
            f.seek(4, 1)  # Skip ngw
            n_gvec = np.fromfile(f, dtype=np.int32, count=1)[0]
            nspin_file = np.fromfile(f, dtype=np.int32, count=1)[0]
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
            bytes_per_band_record = 4 + (nspin_file * n_gvec * 16) + 4
            start_pos = f.tell()
            
            coeffs_list = []
            for iband in bands:
                # Map the user's relative index to the absolute index inside the file
                abs_band = self.meta.bands[iband]
                
                # Direct seek to target band block using absolute addressing
                f.seek(start_pos + abs_band * bytes_per_band_record)
                f.seek(4, 1)  # Skip leading band delimiter
                coeffs_list.append(np.fromfile(f, dtype=self.meta.cplx_dtype, count=nspin_file * n_gvec))
                
        return np.array(coeffs_list).T

    def read_gvectors(self, ikpt: int) -> np.ndarray:
        """Extracts the precise integer Miller indices (h, k, l) from Record 4 ('mill')."""
        if ikpt in self._gvec_cache:
            return self._gvec_cache[ikpt]
            
        binary_file = self._get_binary_file(ispin=0, ikpt=ikpt)
        
        with open(binary_file, "rb") as f:
            # Record 1: Context Header
            rec_len1 = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(rec_len1, 1)
            f.seek(4, 1)

            # Record 2: Dimensions
            f.seek(4, 1)  # Skip leading record delimiter
            f.seek(4, 1)  # Skip ngw
            n_gvec = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(4, 1)  # Skip nspin_file
            f.seek(4, 1)  # Skip nbands_file
            f.seek(4, 1)  # Skip trailing record delimiter
            
            # Record 3: Reciprocal Lattice Matrices
            rec_len3 = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(rec_len3, 1)
            f.seek(4, 1)
            
            # Record 4: 'mill' Index Maps (Explicit Integer G-vectors)
            _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Read leading delimiter/length
            
            # Read the flat 3 * n_gvec coordinate array and reshape into (n_gvec, 3) layout
            gvectors = np.fromfile(f, dtype=np.int32, count=3 * n_gvec).reshape(-1, 3)
            
        self._gvec_cache[ikpt] = gvectors
        return gvectors