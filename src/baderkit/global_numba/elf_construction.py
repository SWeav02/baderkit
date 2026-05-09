import numpy as np
from scipy.fft import fftn, ifftn
from baderkit.toolkit.grid import Grid


def get_gradient_fft(
        rho,
        lattice,
        ):
    """Calculate gradient via fft"""
    recip_lattice = 2 * np.pi * np.linalg.inv(lattice).T

    nx, ny, nz = rho.shape
    kx = np.fft.fftfreq(nx, d=1./nx)
    ky = np.fft.fftfreq(ny, d=1./ny)
    kz = np.fft.fftfreq(nz, d=1./nz)

    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')

    # Mapping to Cartesian G-vectors
    Gx = Kx * recip_lattice[0,0] + Ky * recip_lattice[1,0] + Kz * recip_lattice[2,0]
    Gy = Kx * recip_lattice[0,1] + Ky * recip_lattice[1,1] + Kz * recip_lattice[2,1]
    Gz = Kx * recip_lattice[0,2] + Ky * recip_lattice[1,2] + Kz * recip_lattice[2,2]

    rho_g = fftn(rho)

    # Compute Cartesian gradients
    gx = np.real(ifftn(1j * Gx * rho_g))
    gy = np.real(ifftn(1j * Gy * rho_g))
    gz = np.real(ifftn(1j * Gz * rho_g))

    return gx**2 + gy**2 + gz**2

def calculate_elf_dft(
        rho,
        tau,
        lattice,
        spin,
        ):
    """Modern (DFT-consistent) ELF"""

    grad_rho_sq = get_gradient_fft(
        rho=rho,
        lattice=lattice,
    )

    rho_limit = 1e-30
    elf = np.zeros_like(rho)
    mask = rho > rho_limit

    rho_m = rho[mask]
    tau_m = tau[mask]
    grad_m = grad_rho_sq[mask]

    # --- Weizsäcker kinetic energy ---
    tau_w = grad_m / (8.0 * rho_m)

    # --- Pauli kinetic energy ---
    tau_p = tau_m - tau_w

    # Optional stabilization (recommended)
    tau_p = np.maximum(tau_p, 0.0)

    # --- Thomas-Fermi kinetic energy ---
    if not spin:
        cf = (3.0/10.0) * (3.0 * np.pi**2)**(2.0/3.0)
    else:
        cf = (3.0/10.0) * (6.0 * np.pi**2)**(2.0/3.0)

    tau_tf = cf * rho_m**(5.0/3.0)

    # --- Dimensionless localization measure ---
    chi = tau_p / tau_tf

    # --- ELF ---
    elf[mask] = 1.0 / (1.0 + chi**2)

    return elf

def calculate_elf_hf(
        rho,
        tau,
        lattice,
        spin,
        ):
    """Calculate ELF using QE method"""
    grad_rho_sq = get_gradient_fft(
        rho=rho,
        lattice=lattice,
        )

    # set proper prefactor
    if not spin:
        fac = 5.0 / (3.0 * (6.0 * np.pi**2)**(2.0/3.0))
    else:
        fac = 5.0 / (3.0 * (3.0 * np.pi**2)**(2.0/3.0))
    rho_limit = 1e-30
    stability_shift = 1e-5
    elf = np.zeros_like(rho)
    mask = rho > rho_limit

    # D = (fac / rho^(5/3)) * (tau - 0.25 * |grad rho|^2 / rho + 1e-5)
    d = (fac / (rho[mask]**(5.0/3.0))) * \
        (tau[mask] - 0.25 * grad_rho_sq[mask] / rho[mask] + stability_shift)

    # ELF = 1 / (1 + d^2)
    elf[mask] = 1.0 / (1.0 + d**2)
    return elf

def compute_elf_from_grid(
        charge_grid,
        ked_grid,
        spin=True,
        use_be=False,
        ):
    """
    ELF from BaderKit charge grids.
    """
    ANG_TO_BOHR = 1.88973

    lattice = charge_grid.matrix

    lattice_bohr = lattice * ANG_TO_BOHR

    volume_bohr3 = abs(np.linalg.det(lattice_bohr))

    # convert to charge density in a.u.
    rho = charge_grid.total / (volume_bohr3)
    tau = ked_grid.total / (volume_bohr3)

    # calculate elf
    if use_be:
        elf = calculate_elf_hf(
            rho=rho,
            tau=tau,
            lattice=lattice_bohr,
            spin=spin
            )
    else:
        elf = calculate_elf_dft(
            rho=rho,
            tau=tau,
            lattice=lattice_bohr,
            spin=spin
            )

    # get Grid object
    elf_grid = Grid(
        structure = charge_grid.structure,
        data={"total": elf},
        data_type="elf",
        source_format=charge_grid.source_format
        )

    return elf_grid