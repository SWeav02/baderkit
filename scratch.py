#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from baderkit.core.elf_analysis.overlap.overlap import BasinOverlap
from baderkit.core.elf_analysis.elf_labeler1.elf_labeler import ElfLabeler
from baderkit.core.utilities.transforms import INT_TO_REV_INT, INT_TO_IMAGE
import numpy as np

from pathlib import Path
path = Path(".")

data_dict = {}


nna_neigh_zeffs = {}
nna_vol_ratios = {}
nna_charges = {}
nna_volumes = {}
nna_dists = {}
nna_fracs = {}


for folder in path.iterdir():
    if folder.name == "Tc":
        continue
    chgcar = folder / "CHGCAR"
    elf = folder / "ELFCAR"
    tot = folder / "CHGCAR_sum"
    pot = folder / "POTCAR"
    element = folder.name
    data_dict[element] = {}
    
    labeler = ElfLabeler.from_vasp(
        charge_filename=chgcar,
        reference_filename=elf,
        total_charge_filename=tot,
        persistence_tol=0.01,
        pseudopotential_filename = pot,
        )
    overlap: BasinOverlap = labeler.overlap
    bader = overlap.local_bader
    
    types = labeler.basin_types
    
    nna_indices = np.array([i for i, j in enumerate(types) if j == "nna"])

    data_dict[element]["nna_zeffs"] = labeler.neighbor_zeffs[nna_indices]
    data_dict[element]["nna_veffs"] = labeler.neighbor_veffs[nna_indices]
    data_dict[element]["nna_vol_ratios"] = labeler.core_volume_ratios[nna_indices]
    data_dict[element]["nna_charges"] = overlap.local_bader.basin_charges[nna_indices]
    data_dict[element]["nna_volumes"] = overlap.local_bader.basin_volumes[nna_indices]
    data_dict[element]["nna_dists"] = labeler.nna_bond_dists
    data_dict[element]["nna_fracs"] = labeler.nna_bond_fracs
    data_dict[element]["nna_dist_beyond_atom"] = labeler.nna_bond_dists * labeler.nna_bond_fracs
    data_dict[element]["nna_potential_energies"] = labeler.nna_potential_energies
    
elements = [i for i in data_dict.keys()]
alphabetical_elements = np.sort(elements)

# plotting
workfunctions = np.array([ # alphabetical
    4.33,
    2.87,
    4.08,
    4.46,
    4.08,
    5.22,
    4.98,
    4.71,
    3.1,
    4.05,
    ])

resistivities = np.array([
    1.587,
    3.11,
    6.8,
    4.85,
    15.2,
    9.78,
    4.3,
    7.1,
    59.6,
    38.8,
    ])

workfunction_indices = np.argsort(workfunctions)
sorted_workfunctions = workfunctions[workfunction_indices]
sorted_elements = alphabetical_elements[workfunction_indices]

res_indices = np.argsort(resistivities)
sorted_res = resistivities[res_indices]
sorted_elements_res = alphabetical_elements[res_indices]

import matplotlib.pyplot as plt

x = []
y = []
y_std = []
for workfunction, element in zip(sorted_workfunctions, sorted_elements):
    # get x-axis
    x.append(workfunction)
    # get y-axis
    dict_el = data_dict[element]
    dists = dict_el["nna_dists"]
    fracs = dict_el["nna_fracs"]
    volumes = dict_el["nna_volumes"]
    charges = dict_el["nna_charges"]
    zeffs = dict_el["nna_zeffs"]
    veffs = dict_el["nna_veffs"]
    vol_ratios = dict_el["nna_vol_ratios"]
    dist_beyond_atom = dict_el["nna_dist_beyond_atom"]
    weighted_coulomb = dict_el["nna_potential_energies"]
    
    atom_rhos = zeffs / veffs
    basin_rhos = charges / volumes
    
    charge_frac = charges / charges.sum()
    volume_frac = volumes / volumes.sum()
    nna_charge_total = charges.sum()
    # rho_frac = (charges/volumes) / (charges/volumes).sum()
    
    # value = basin_rhos/(dist_beyond_atom) # best so far
    value = weighted_coulomb

    
    y.append(np.sum(value*volume_frac))
    
    
    # values = vol_ratios
    # avg = np.mean(values)
    # stddev = np.std(values)
    # y.append(avg)
    # y_std.append(stddev)

plt.scatter(x, y)

plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Workfunction vs. 1/dist")
plt.show()

# plt.errorbar(
#     x, y,
#     yerr=y_std,
#     fmt='o',        # 'o' = circular markers (scatter style)
#     capsize=4,      # little caps on error bars
#     linestyle='none' # no connecting line
# )

# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title(f"Resistivities vs. 1/dist")
# plt.show()len(atom_zeffs)