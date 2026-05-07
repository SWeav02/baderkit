import numpy as np

ANG_TO_BOHR = 1.88973

def gradient_fft(rho, lattice_bohr):
    """
    Gradient on a uniform fractional grid for a general 3D cell.

    Parameters
    ----------
    rho : ndarray, shape (nx, ny, nz)
        Scalar field in real space.
    lattice_bohr : ndarray, shape (3, 3)
        Lattice vectors as rows in Bohr.

    Returns
    -------
    gx, gy, gz : ndarrays
        Cartesian components of grad(rho) in Bohr^-4 if rho is Bohr^-3.
    """
    nx, ny, nz = rho.shape

    # Integer Fourier indices on the fractional grid
    g1 = 2.0 * np.pi * np.fft.fftfreq(nx)# * nx
    g2 = 2.0 * np.pi * np.fft.fftfreq(ny)# * ny
    g3 = 2.0 * np.pi * np.fft.fftfreq(nz)# * nz
    g1, g2, g3 = np.meshgrid(g1, g2, g3, indexing="ij")

    rho_k = np.fft.fftn(rho)

    # Derivatives with respect to fractional coordinates s1, s2, s3
    dr_ds1 = np.fft.ifftn(1j * g1 * rho_k).real
    dr_ds2 = np.fft.ifftn(1j * g2 * rho_k).real
    dr_ds3 = np.fft.ifftn(1j * g3 * rho_k).real

    grad_s = np.stack([dr_ds1, dr_ds2, dr_ds3], axis=0)

    # Convert to Cartesian gradient:
    # ∇_r = A^{-T} ∇_s
    inv_lattice = np.linalg.inv(lattice_bohr)
    grad_r = np.tensordot(inv_lattice, grad_s, axes=(1, 0))

    return grad_r[0], grad_r[1], grad_r[2]


def compute_elf(rho, tau, lattice, spin=True, eps=1e-12):
    """
    ELF from VASP-style CHGCAR-like inputs:
      rho, tau = (physical field) * (NGXF*NGYF*NGZF) * V_cell
    with lattice in Angstrom.

    Returns spin-channel ELF on the same grid.
    """
    lattice_bohr = lattice * ANG_TO_BOHR
    
    volume_bohr3 = abs(np.linalg.det(lattice_bohr))
    # Undo CHGCAR scaling
    rho = rho / (volume_bohr3)
    tau = tau / (volume_bohr3)

    gx, gy, gz = gradient_fft(rho, lattice_bohr)
    grad2 = gx**2 + gy**2 + gz**2

    rho_safe = np.maximum(rho, eps)

    tau_w = grad2 / (8.0 * rho_safe)
    D = tau - tau_w

    D_heg = (3.0 / 5.0) * (6.0 * np.pi**2)**(2.0 / 3.0) * rho_safe**(5.0 / 3.0)

    chi = D / (D_heg + eps)
    elf = 1.0 / (1.0 + chi**2)

    elf[rho < eps] = 0.0
    return elf


import numpy as np
from scipy.fft import fftn, ifftn

def read_cube(file_path):
    """Reads a Gaussian .cube file and returns data, lattice, origin, and full header."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[:6]
    n_atoms = int(lines[2].split()[0])
    origin = np.array(lines[2].split()[1:4], dtype=float)
    
    shape = []
    lattice = []
    for i in range(3, 6):
        parts = lines[i].split()
        shape.append(int(parts[0]))
        # Lattice vector = step_vector * number_of_steps
        lattice.append(np.array(parts[1:4], dtype=float) * int(parts[0]))
    
    shape = tuple(shape)
    lattice = np.array(lattice)
    
    # Skip atom lines to get to the data
    data_start = 6 + abs(n_atoms)
    data = np.fromstring(" ".join(lines[data_start:]), sep=' ').reshape(shape)
    
    return data, lattice, origin, header, lines[6:data_start]

def write_cube(file_path, data, header, atom_lines):
    """Writes data to a .cube file using the original header and atom positions."""
    with open(file_path, 'w') as f:
        f.writelines(header)
        f.writelines(atom_lines)
        
        # Flatten and write data in blocks of 6 as per standard cube format
        flat_data = data.flatten()
        for i in range(0, len(flat_data), 6):
            f.write(" ".join(f"{x:13.5E}" for x in flat_data[i:i+6]) + "\n")

class QEAnalysis:
    def __init__(self, rho, tau, lattice):
        self.rho = rho # Charge density
        self.tau = tau # Kinetic energy density (kkin in QE) [cite: 17, 51]
        self.lattice = lattice
        self.shape = rho.shape
        # Reciprocal lattice vectors B such that A . B^T = 2*pi*I
        self.recip_lattice = 2 * np.pi * np.linalg.inv(lattice).T

    def get_gradient_sq(self):
        """Calculates |grad rho|^2 in reciprocal space for non-orthogonal cells."""
        nx, ny, nz = self.shape
        kx = np.fft.fftfreq(nx, d=1./nx)
        ky = np.fft.fftfreq(ny, d=1./ny)
        kz = np.fft.fftfreq(nz, d=1./nz)
        
        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Mapping to Cartesian G-vectors 
        Gx = Kx * self.recip_lattice[0,0] + Ky * self.recip_lattice[1,0] + Kz * self.recip_lattice[2,0]
        Gy = Kx * self.recip_lattice[0,1] + Ky * self.recip_lattice[1,1] + Kz * self.recip_lattice[2,1]
        Gz = Kx * self.recip_lattice[0,2] + Ky * self.recip_lattice[1,2] + Kz * self.recip_lattice[2,2]
        
        rho_g = fftn(self.rho)
        
        # Compute Cartesian gradients
        gx = np.real(ifftn(1j * Gx * rho_g))
        gy = np.real(ifftn(1j * Gy * rho_g))
        gz = np.real(ifftn(1j * Gz * rho_g))
        
        return gx**2 + gy**2 + gz**2

    def calculate_elf(self):
        """Reproduces the QE ELF formula [cite: 7, 48-50]."""
        grad_rho_sq = self.get_gradient_sq()
        
        # Constants from source [cite: 48, 50]
        # fac = 5 / (3 * (6 * pi^2)^(2/3))
        fac = 5.0 / (3.0 * (6.0 * np.pi**2)**(2.0/3.0))
        rho_limit = 1e-30 # [cite: 49]
        stability_shift = 1e-5 # [cite: 49]
        
        elf = np.zeros_like(self.rho)
        mask = self.rho > rho_limit
        
        # D = (fac / rho^(5/3)) * (tau - 0.25 * |grad rho|^2 / rho + 1e-5) [cite: 49, 50]
        d = (fac / (self.rho[mask]**(5.0/3.0))) * \
            (self.tau[mask] - 0.25 * grad_rho_sq[mask] / self.rho[mask] + stability_shift)
        
        # ELF = 1 / (1 + d^2) 
        elf[mask] = 1.0 / (1.0 + d**2)
        return elf

# --- Execution ---
if __name__ == "__main__":
    print("Loading chg.cube and kin.cube...")
    rho, lattice, origin, header, atoms = read_cube("chg.cube")
    tau, _, _, _, _ = read_cube("kin.cube")

    analyzer = QEAnalysis(rho, tau, lattice)
    
    print("Calculating ELF...")
    elf_data = analyzer.calculate_elf()
    
    print("Writing elf.cube...")
    # Update the title in the header
    header[0] = "ELF calculated from chg.cube and kin.cube\n"
    write_cube("elf.cube", elf_data, header, atoms)
    print("Done.")