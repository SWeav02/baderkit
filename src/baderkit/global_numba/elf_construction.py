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
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    kz = np.fft.fftfreq(nz, d=1.0 / nz)

    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing="ij")

    # Mapping to Cartesian G-vectors
    Gx = Kx * recip_lattice[0, 0] + Ky * recip_lattice[1, 0] + Kz * recip_lattice[2, 0]
    Gy = Kx * recip_lattice[0, 1] + Ky * recip_lattice[1, 1] + Kz * recip_lattice[2, 1]
    Gz = Kx * recip_lattice[0, 2] + Ky * recip_lattice[1, 2] + Kz * recip_lattice[2, 2]

    rho_g = fftn(rho)

    # Compute Cartesian gradients
    gx = np.real(ifftn(1j * Gx * rho_g))
    gy = np.real(ifftn(1j * Gy * rho_g))
    gz = np.real(ifftn(1j * Gz * rho_g))

    return gx**2 + gy**2 + gz**2


def get_derivatives_fft(rho, lattice):
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T

    nx, ny, nz = rho.shape
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    kz = np.fft.fftfreq(nz, d=1.0 / nz)

    K = np.stack(np.meshgrid(kx, ky, kz, indexing="ij"), axis=-1)
    G = K @ reciprocal

    Gx, Gy, Gz = G[..., 0], G[..., 1], G[..., 2]
    G2 = Gx**2 + Gy**2 + Gz**2

    rho_g = fftn(rho)

    gx = ifftn(1j * Gx * rho_g).real
    gy = ifftn(1j * Gy * rho_g).real
    gz = ifftn(1j * Gz * rho_g).real
    grad_sq = gx**2 + gy**2 + gz**2

    laplacian = ifftn(-G2 * rho_g).real

    return grad_sq, laplacian


def calculate_elf_qe(
    rho,
    tau,
    lattice,
    spin,
):
    """Calculate ELF using QE method"""
    grad_rho_sq, laplacian = get_derivatives_fft(
        rho=rho,
        lattice=lattice,
    )

    pi_sq = np.pi**2

    tau_corr = laplacian / 2

    tau_bos = (1 / 4) * (grad_rho_sq / rho)

    if not spin:
        dh = 0.2 / pi_sq * (3 * pi_sq * rho) ** (5.0 / 3.0)
    else:
        dh = 0.2 / pi_sq * (6 * pi_sq * rho) ** (5.0 / 3.0)

    dh[rho <= 0.0] = 0.0

    D = (tau + tau_corr - tau_bos) / np.maximum(dh, 1e-08)

    # ELF = 1 / (1 + d^2)
    elf = 1.0 / (1.0 + D**2)
    return elf


def compute_elf_from_grid(
    charge_grid,
    ked_grid,
    spin=True,
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
    elf = calculate_elf_qe(rho=rho, tau=tau, lattice=lattice_bohr, spin=spin)

    # get Grid object
    elf_grid = Grid(
        structure=charge_grid.structure,
        data={"total": elf},
        data_type="elf",
        source_format=charge_grid.source_format,
    )

    return elf_grid
