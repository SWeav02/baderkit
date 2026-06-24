# -*- coding: utf-8 -*-

import numpy as np

###############################################################################
# BUILDING BLOCKS
###############################################################################

pi2 = np.pi ** 2

def eli_heg(rho, single_channel=False):
    if single_channel:
        prefactor = 3/5 * (6*pi2)**(2/3)
    else:
        prefactor = 3/5 * (3*pi2)**(2/3)
        
    return np.where(rho > 0.0, prefactor * rho**(5./3), 0.0)
    
def eli(rho, tau, lap_rho, grad_rho_sq):
    tau_corr = lap_rho / 2
    
    # Safe division: only divide where rho > 0.0
    tau_bos = np.divide(
        0.25 * grad_rho_sq, 
        rho, 
        out=np.zeros_like(rho), 
        where=rho > 0.0
    )
    
    return tau + tau_corr - tau_bos
    
    # tau_w_corr = tau + tau_corr
    # tau_bos = np.minimum(tau_w_corr, tau_bos)
    
    # D = (tau_w_corr - tau_bos)
    # return D

###############################################################################
# KERNELS
###############################################################################

def elf_kernel(rho, tau, lap_rho, grad_rho_sq, savin_correction=True, single_channel=False):
    D0 = eli_heg(rho, single_channel)
    D = eli(rho, tau, lap_rho, grad_rho_sq)
    
    numerator = (D + 2.871e-5) if savin_correction else D

    # Safe division: only divide where D0 > 0.0
    X = np.divide(
        numerator, 
        D0, 
        out=np.zeros_like(numerator), 
        where=D0 > 0.0
    )
    return X
    

def lol_kernel(rho, tau, lap_rho, savin_correction=True, single_channel=False):
    tau = np.maximum(tau, 0.0)
    tau_corr = lap_rho / 2
    D0 = eli_heg(rho, single_channel)
    
    numerator = (tau +tau_corr+ 2.871e-5) if savin_correction else tau + tau_corr
    
    # Safe division: only divide where D0 > 0.0
    X = np.divide(
        numerator, 
        D0, 
        out=np.zeros_like(numerator), 
        where=D0 > 0.0
    )
    return X

def elid_kernel(rho, tau, lap_rho, grad_rho_sq):
    D = eli(rho, tau, lap_rho, grad_rho_sq)
    
    # Avoid negative powers directly on 0.0 by using division instead
    rho_power = rho ** (8. / 3.)
    
    # Safe division: D / (rho ** (8/3)) only where rho > 0.0
    X = np.divide(
        D, 
        rho_power, 
        out=np.zeros_like(D), 
        where=rho > 0.0
    )
    return X

###############################################################################
# STANDARD LOCALIZATION FUNCTIONS
###############################################################################

def elf(rho, tau, lap_rho, grad_rho_sq, savin_correction=True, single_channel=False):
    X = elf_kernel(rho, tau, lap_rho, grad_rho_sq, savin_correction, single_channel)
    elf_val = 1 / (1 + X**2)
    return np.where(rho > 0.0, elf_val, 0.0)

def lol(rho, tau, savin_correction=True, single_channel=False):
    X = lol_kernel(rho, tau, savin_correction, single_channel)
    lol_val = 1 / (1 + X)
    return np.where(rho > 0.0, lol_val, 0.0)

def elid(rho, tau, lap_rho, grad_rho_sq):
    return elid_kernel(rho, tau, lap_rho, grad_rho_sq)