# -*- coding: utf-8 -*-

import numpy as np
from baderkit.toolkit import Structure
from .base import BaseWfcReader, WfcMetadata

def get_reciprocal_index(
    ispin,
    ikpt,
    iband,
    nkpts,
    nbands,  # Must ALWAYS represent the physical max_nbands of the file
    nspin
):
    """
    Maps 1-indexed quantum numbers to direct-access binary record locations
    inside the rigid Fortran layout structure of a VASP WAVECAR file.
    """
    assert 1 <= ispin <= nspin,  'Invalid spin index!'
    assert 1 <= ikpt <= nkpts,  'Invalid kpoint index!'
    assert 1 <= iband <= nbands+1, 'Invalid band index!'

    rec = 2 + (ispin - 1) * nkpts * (nbands + 1) + \
              (ikpt - 1) * (nbands + 1) + \
        iband
    return rec

class VaspReader(BaseWfcReader):
    """
    Reader implementation to harvest planewave states directly from VASP direct-access binary WAVECAR files.
    """
    def __init__(
        self,
        use_vasp4: bool = False,
        **kwargs
    ):
        self.use_vasp4 = use_vasp4
        super().__init__(**kwargs)
    
    def read_metadata(
        self,
        nbands = None,
        bands = None,
    ) -> WfcMetadata:
        """
        Parses structural matrices and record maps out of POSCAR and WAVECAR files
        contained within the targeted calculation directory.
        """
        wavecar_filename = self.directory / "WAVECAR"
        poscar_filename = self.directory / "POSCAR"
        
        if not wavecar_filename.exists():
            raise FileNotFoundError(f"Could not find WAVECAR at {wavecar_filename}")
            
        with open(wavecar_filename, "rb") as file:
            # 1. Read Record 1 to extract the record length and precision signatures
            file.seek(0)
            self.recl = int(np.fromfile(file, dtype=np.float64, count=1)[0])
            nspin = int(np.fromfile(file, dtype=np.float64, count=1)[0])
            rtag = int(np.fromfile(file, dtype=np.float64, count=1)[0])
            
            # Determine appropriate floating-point allocation profile based on precision flag
            if rtag == 45200:
                cplx_dtype = np.complex64
            elif rtag == 45210:
                cplx_dtype = np.complex128
            else:
                raise ValueError(f"Unknown or unsupported precision flag found inside WAVECAR: {rtag}")
                
            # 2. Advance to Record 2 to parse active dimensions, grids, and cells
            file.seek(self.recl)
            nkpts = int(np.fromfile(file, dtype=np.float64, count=1)[0])
            max_nbands = int(np.fromfile(file, dtype=np.float64, count=1)[0])
            
            # Resolve physical bands we want to target
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
            energy_cutoff = np.fromfile(file, dtype=np.float64, count=1)[0]
            
            # Real-space cell matrix layout (VASP standard 3x3 array row-wise)
            lattice_matrix = np.fromfile(file, dtype=np.float64, count=9).reshape(3, 3)
            
            # Reconcile crystal configuration references
            if poscar_filename.exists():
                structure = Structure.from_file(poscar_filename)
            else:
                structure = Structure(lattice_matrix, species=["X"], coords=[[0, 0, 0]])
                
            # Pre-allocate array blocks mapped to the SELECTED band size
            kpoints = np.zeros((nkpts, 3))
            energies = np.zeros((nspin, nkpts, n_selected_bands))
            occupancies = np.zeros((nspin, nkpts, n_selected_bands))
            
            # 3. Loop over records to extract eigenvalues, coordinates, and weights
            for ispin_idx in range(nspin):
                for ikpt_idx in range(nkpts):
                    # Seek using physical max_nbands so file positions line up perfectly
                    rec_header = get_reciprocal_index(ispin_idx + 1, ikpt_idx + 1, 1, nkpts, max_nbands, nspin) - 1
                    file.seek(rec_header * self.recl)
                    
                    # Read the plane-wave count and the fractional kpoint coordinates
                    header_data = np.fromfile(file, dtype=np.float64, count=4)
                    if ispin_idx == 0:
                        kpoints[ikpt_idx, :] = header_data[1:4]
                        
                    # Energies, weights, and occupancies: must read the complete layout (3 * max_nbands)
                    eb_data = np.fromfile(file, dtype=np.float64, count=3 * max_nbands).reshape(max_nbands, 3)
                    
                    # Store only the mapped physical bands
                    energies[ispin_idx, ikpt_idx, :] = eb_data[selected_bands, 0]
                    occupancies[ispin_idx, ikpt_idx, :] = eb_data[selected_bands, 2]
                    
            # Set a dynamic Fermi Level baseline using occupied bands if not explicitly given
            efermi = np.max(energies[occupancies > 0.1]) if np.any(occupancies > 0.1) else 0.0

        self.meta = WfcMetadata(
            structure=structure,
            kpoints=kpoints,
            occupancies=occupancies,
            energies=energies,
            energy_cutoff=energy_cutoff,
            efermi=efermi,
            nspin=nspin,
            nkpts=nkpts,
            nbands=n_selected_bands,
            max_nbands=max_nbands,
            bands=selected_bands,
            cplx_dtype=cplx_dtype
        )
        return self.meta

    def read_coefficients(self, ispin: int, ikpt: int, iband: int) -> np.ndarray:
        """Extracts a flat 1D array of complex plane-wave coefficients for a single target state."""
        return self.read_coefficients_batch(ispin, ikpt, [iband])[0]

    def read_coefficients_batch(self, ispin: int, ikpt: int, bands: list) -> np.ndarray:
        """Streams multiple targeted band blocks under a unified file open transaction session."""
        wavecar_filename = self.directory / "WAVECAR"
        
        with open(wavecar_filename, "rb") as file:
            # Locate the baseline k-point header block using physical max_nbands
            rec_band_header = get_reciprocal_index(
                ispin + 1,
                ikpt + 1,
                1,
                self.meta.nkpts,
                self.meta.max_nbands,
                self.meta.nspin
            ) - 1
            
            file.seek(rec_band_header * self.recl)
            npw = int(np.fromfile(file, dtype=np.float64, count=1)[0])
            
            coeffs_list = []
            for iband in bands:
                # Map selected subspace index back to physical file-level index
                abs_band = self.meta.bands[iband]
                
                rec_coeff = get_reciprocal_index(
                    ispin + 1,
                    ikpt + 1,
                    abs_band + 2, # Account for VASP's 1-indexed offset + header record
                    self.meta.nkpts,
                    self.meta.max_nbands,
                    self.meta.nspin
                ) - 1
                
                file.seek(rec_coeff * self.recl)
                coeffs_list.append(np.fromfile(file, dtype=self.meta.cplx_dtype, count=npw))
                
        return np.array(coeffs_list)

    def read_gvectors(self, ikpt: int) -> np.ndarray:
        """
        Reconstructs the precise integer Miller indices (h, k, l) matching VASP's
        internal plane-wave ordering layout for a specific k-point.
        """
        if ikpt in self._gvec_cache:
            return self._gvec_cache[ikpt]

        lattice = self.meta.structure.lattice.matrix
        kpt = self.meta.kpoints[ikpt]
        encut = self.meta.energy_cutoff
        
        # Kinetic energy factor matching standard VASP convention exactly
        HSQDTM = 3.8100198740807945
        
        # Include the 2*pi factor to get true Cartesian coordinates in A^-1
        B_mat = 2 * np.pi * np.linalg.inv(lattice).T
        
        # Determine the maximum safe bounding box coordinates via Cauchy-Schwarz projection
        R = np.sqrt(encut / HSQDTM)
        max_g = np.array([int(np.ceil(R * np.linalg.norm(lattice[i, :]))) + 1 for i in range(3)])
            
        # VASP visits points in FFT order: positive frequencies first, then negative frequencies
        def get_fft_ordered_sequence(max_val):
            return list(range(0, max_val + 1)) + list(range(-max_val, 0))
            
        seq1 = np.array(get_fft_ordered_sequence(max_g[0]), dtype=np.int32)
        seq2 = np.array(get_fft_ordered_sequence(max_g[1]), dtype=np.int32)
        seq3 = np.array(get_fft_ordered_sequence(max_g[2]), dtype=np.int32)
        
        # VECTORIZED GRID: Replicate VASP nested loops loop ordering (n3 outermost -> n2 -> n1 innermost)
        N3, N2, N1 = np.meshgrid(seq3, seq2, seq1, indexing='ij')
        g_all = np.stack([N1.ravel(), N2.ravel(), N3.ravel()], axis=-1)
        
        # Fetch expected plane wave count using max_nbands
        wavecar_file = self.directory / "WAVECAR"
        with open(wavecar_file, "rb") as f:
            rec_band_header = get_reciprocal_index(
                1, ikpt + 1, 1, self.meta.nkpts, self.meta.max_nbands, self.meta.nspin
            ) - 1
            f.seek(rec_band_header * self.recl)
            expected_npw = int(np.fromfile(f, dtype=np.float64, count=1)[0])
            
        # Screen all vectors globally using parallel matrix math
        k_cart_all = (kpt[np.newaxis, :] + g_all) @ B_mat
        energies_all = HSQDTM * np.sum(k_cart_all**2, axis=1)
        
        gvectors = g_all[energies_all <= encut]
        
        # Avoid sorting the final gvectors array directly, which destroys the VASP FFT loop order.
        if len(gvectors) != expected_npw:
            indices = np.where(energies_all <= encut)[0]
            if len(indices) < expected_npw:
                sorted_energies = np.sort(energies_all)
                effective_cutoff = sorted_energies[expected_npw - 1]
                indices = np.where(energies_all <= effective_cutoff)[0]
            
            # Truncate strictly to expected_npw while preserving the natural n3->n2->n1 loop order
            indices = indices[:expected_npw]
            gvectors = g_all[indices]
            
        self._gvec_cache[ikpt] = gvectors
        return gvectors