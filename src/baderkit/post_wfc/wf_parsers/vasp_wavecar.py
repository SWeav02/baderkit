# -*- coding: utf-8 -*-

import numpy as np

from baderkit.toolkit import Structure

BOHR_TO_ANG    = 0.529177249
RY_TO_EV   = 13.605826
TWOPI = 2*np.pi

def get_reciprocal_index(
        ispin,
        ikpt,
        iband,
        nkpts,
        nbands,
        nspin
        ):
    assert 1 <= ispin <= nspin,  'Invalid spin index!'
    assert 1 <= ikpt <= nkpts,  'Invalid kpoint index!'
    assert 1 <= iband <= nbands+1, 'Invalid band index!'

    rec = 2 + (ispin - 1) * nkpts * (nbands + 1) + \
              (ikpt - 1) * (nbands + 1) + \
        iband
    return rec

def read_vasp(
        poscar_filename,
        wavecar_filename,
        use_vasp4 = False,
        ):
    """
    Based on [VaspBandUnfolding](https://github.com/QijingZheng/VaspBandUnfolding)
    by Qijing Zheng.
    
    The format of VASP WAVECAR, as shown in
        http://www.andrew.cmu.edu/user/feenstra/wavetrans/
    is:
        Record-length #spin components RTAG(a value specifying the precision)
        #k-points #bands ENCUT(maximum energy for plane waves)
        LatVec-A
        LatVec-B
        LatVec-C
        Loop over spin
           Loop over k-points
              #plane waves, k vector
              Loop over bands
                 band energy, band occupation
              End loop over bands
              Loop over bands
                 Loop over plane waves
                    Plane-wave coefficient
                 End loop over plane waves
              End loop over bands
           End loop over k-points
        End loop over spin
    
    """
    # read structure
    structure = Structure.from_file(poscar_filename)
    
    with open(wavecar_filename, "rb") as file:
        #######################################################################
        # HEADER INFO
        #######################################################################
        # Read Header Information
        file.seek(0)
        recl, nspin, rtag = np.array(
            np.fromfile(file, dtype=np.float64, count=3),
            dtype=np.int64
        )
        if rtag == 45200:
            cplx_dtype = np.complex64
        elif rtag == 45210:
            cplx_dtype = np.complex128
        else:
            raise ValueError(f"Unknown RTAG format: {rtag}")
        # the second record
        file.seek(recl)
        # From VASP 5.x on, Fermi energy is also written in this line, hence
        # change 12 to 13
        if use_vasp4:
            dump = np.fromfile(file, dtype=np.float64, count=12)
            efermi = None
        else:
            dump = np.fromfile(file, dtype=np.float64, count=13)
            efermi = dump[12]

        nkpts = int(dump[0])                 # No. of k-points
        nbands = int(dump[1])                # No. of bands
        encut = dump[2]           # Energy cutoff (eV)
        
        #######################################################################
        # BAND INFO
        #######################################################################
        
        # create arrays to store final information
        # nplane_waves = np.zeros(nkpts, dtype=int)
        kpoint_frac = np.zeros((nkpts, 3), dtype=float)
        bands = np.zeros(
            (nspin, nkpts, nbands), dtype=float)
        occs = np.zeros(
            (nspin, nkpts, nbands), dtype=float)

        for ii in range(nspin):
            for jj in range(nkpts):
                rec = get_reciprocal_index(
                    ii+1,
                    jj+1,
                    1,
                    nkpts,
                    nbands,
                    nspin
                    ) - 1
                file.seek(rec * recl)
                dump = np.fromfile(file, dtype=np.float64,
                                   count=4+3*nbands)
                if ii == 0:
                    # nplane_waves[jj] = int(dump[0])
                    kpoint_frac[jj] = dump[1:4]
                dump = dump[4:].reshape((-1, 3))
                bands[ii, jj, :] = dump[:, 0]
                occs[ii, jj, :] = dump[:, 2]
                
    vasp_dict = {
        "wavecar_filename": wavecar_filename,
        "nspin": nspin,
        "nkpts": nkpts,
        "nbands": nbands,
        "recl": recl,
        "cplx_dtype": cplx_dtype
        }

    return structure, kpoint_frac, occs, bands, encut, efermi, vasp_dict

def read_pw_coefficients(
        wavecar_filename, 
        ispin, 
        ikpt, 
        iband, 
        nspin,
        nkpts,
        nbands,
        recl, 
        cplx_dtype,
        ):
    """
    Dynamically extracts the plane-wave coefficients for a single specific state
    without loading the rest of the WAVECAR file into memory.
    
    Parameters:
    -----------
    wavecar_filename : str
        Path to the VASP WAVECAR file.
    ispin : int
        0-indexed spin channel (0 for spin-up, 1 for spin-down).
    ikpt : int
        0-indexed k-point index.
    iband : int
        0-indexed band index.
        
    Returns:
    --------
    np.ndarray
        A 1D array of complex plane-wave coefficients for the chosen state.
    """
    with open(wavecar_filename, "rb") as file:
        # 1. Read the first record to get record length and precision tag
        file.seek(0)
            
        # 3. Find the number of plane waves (npw) for this specific k-point
        # In VASP, npw is stored at the beginning of the header record for this spin/kpt combo.
        # (This corresponds to setting the band index parameter to 1 in get_reciprocal_index)
        rec_band_header = get_reciprocal_index(
            ispin + 1,
            ikpt + 1,
            1,
            nkpts,
            nbands,
            nspin
        ) - 1
        
        file.seek(rec_band_header * recl)
        npw = int(np.fromfile(file, dtype=np.float64, count=1)[0])
        
        # 4. Seek directly to the target band's coefficient block
        # Coefficients blocks start at band index + 2 offset 
        rec_coeff = get_reciprocal_index(
            ispin + 1,
            ikpt + 1,
            iband + 2,
            nkpts,
            nbands,
            nspin
        ) - 1
        
        file.seek(rec_coeff * recl)
        
        # 5. Read exactly `npw` complex numbers and return
        coeffs = np.fromfile(file, dtype=cplx_dtype, count=npw)
        
    return coeffs
