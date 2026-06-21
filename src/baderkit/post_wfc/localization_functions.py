# -*- coding: utf-8 -*-

import numpy as np

###############################################################################
# BUILDING BLOCKS
###############################################################################

pi2 = np.pi ** 2

def eli_heg(
    rho,
    single_channel = False
        ):
    # select appropriate prefactor. This differs depending on if we are considering
    # one spin channel or the combined KED/rho of both
    if single_channel:
        prefactor = 3/5 * (6*pi2)**(2/3)
    else:
        prefactor = 3/5 * (3*pi2)**(2/3)
        
    D0 = np.where(rho > 0.0, prefactor * rho**(5./3), 0.0)
        
    # ensure numerical stability
    D0 = np.maximum(D0, 1e-08)
    
    return D0
    
def eli(
    rho,
    tau,
    lap_rho,
    grad_rho_sq,
        ):
    
    # ELI = tau + tau_correlation - tau_boson
    tau_corr = lap_rho / 2

    # ensure numerical stability
    rho = np.maximum(rho, 1e-08)
    
    tau_bos = (1 / 4) * (grad_rho_sq / rho)
    
    D = (tau + tau_corr - tau_bos)
    
    return D

###############################################################################
# KERNELS
###############################################################################

def elf_kernel(
    rho,
    tau,
    lap_rho,
    grad_rho_sq,
    single_channel = False,
        ):
    
    # HEG reference
    D0 = eli_heg(rho, single_channel)
    
    # ELI
    D = eli(rho, tau, lap_rho, grad_rho_sq)

    # ELF kernel with Savin shifting factor
    X = (D + 2.871e-5) / D0
    return X
    

def lol_kernel(
    rho,
    tau,
    single_channel=False,
        ):
    
    # HEG reference
    D0 = eli_heg(rho, single_channel)
    
    # LOL kernel
    X = tau / D0
    return X

def elid_kernel(
    rho,
    tau,
    lap_rho,
    grad_rho_sq,
        ):
    
    # ELI
    D = eli(rho, tau, lap_rho, grad_rho_sq)
    
    # ELI-D
    X = D * (rho ** (-8. / 3.))
    return X

###############################################################################
# STANDARD LOCALIZATION FUNCTIONS
###############################################################################
# These methods calculate the localization functions using the standard method
# straight from the provided properties. One option for partitioned ELF is to
# use these standard methods with the partial rho/tau

def elf(
    rho,
    tau,
    lap_rho,
    grad_rho_sq,
    single_channel = False,
        ):
    X = elf_kernel(
        rho,
        tau,
        lap_rho,
        grad_rho_sq,
        single_channel,
        )
    return 1 / (1 + X**2)

def lol(
    rho,
    tau,
    single_channel=False,
        ):
    X = lol_kernel(
        rho,
        tau,
        single_channel,
        )
    return 1 / (1 + X)

def elid(
    rho,
    tau,
    lap_rho,
    grad_rho_sq
        ):
    X = elid_kernel(
        rho, 
        tau, 
        lap_rho, 
        grad_rho_sq,
        )
    return X

###############################################################################
# NORMALIZED PARTIAL LOCALIZATION FUNCTIONS
###############################################################################
# These methods take inspiration from the work of [Pilme](https://onlinelibrary.wiley.com/doi/10.1002/jcc.24672)
# In Pilme's method, the localization function is first calculated for the total
# system. Then, the ratio of partial charge density to the total charge density
# is used to calculate a partial ELF.

# This original implementation uses a factor of 2 so that the spin-separated
# systems map back to the total if there is no polarization. However, this is
# not fully satisfactory. If one chooses the total rho/tau as their "partial"
# system, the factor of 2 results in the ELF denominator heavily decreasing,
# artificially increasing the final value. To counteract this, we change the
# # factor of 2 to the ratio of total charge to partial charge. This in effect
# # creates an imaginary system with the same total charge as the original system,
# # but with different occupations of the orbitals. In effect, this highlights
# # how the localization would change if certain orbitals were or were not occupied
# # while keeping the system uncharged.

# def elf_relative(
#     rho,
#     tau,
#     lap_rho,
#     grad_rho_sq,
#     partial_rho,
#     partial_tau,
#     partial_lap_rho,
#     partial_grad_rho_sq,
#     single_channel = False,
#         ):
#     # get total kernel
#     X = elf_kernel(
#         rho,
#         tau,
#         lap_rho,
#         grad_rho_sq,
#         single_channel,
#         )
#     # get partial kernel
#     x = elf_kernel(
#         partial_rho,
#         partial_tau,
#         partial_lap_rho,
#         partial_grad_rho_sq,
#         single_channel,
#         )
    
#     # x is divided by X because the ELF should increase if as chi decreases i.e.
#     # if x is smaller than X the relative ELF should be larger
    
#     return 1 / (1 + (x/X)**2)

# def lol_relative(
#     rho,
#     tau,
#     partial_rho,
#     partial_tau,
#     single_channel=False,
#         ):
#     # get total kernel
#     X = lol_kernel(
#         rho,
#         tau,
#         single_channel,
#         )
#     # get partial kernel
#     x = lol_kernel(
#         partial_rho,
#         partial_tau,
#         single_channel
#         )
    
#     return 1 / (1 + (x/X))

# def elid_relative(
#     rho,
#     tau,
#     lap_rho,
#     grad_rho_sq,
#     partial_rho,
#     partial_tau,
#     partial_lap_rho,
#     partial_grad_rho_sq,
#         ):
#     # get total kernel
#     X = elid_kernel(
#         rho, 
#         tau, 
#         lap_rho, 
#         grad_rho_sq,
#         )
#     # get partial kernel
#     x = elid_kernel(
#         partial_rho, 
#         partial_tau, 
#         partial_lap_rho, 
#         partial_grad_rho_sq,
#         )
    
#     return (x/X)