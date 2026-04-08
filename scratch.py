#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from baderkit.core.elf_analysis.overlap.overlap import BasinOverlap
from baderkit.core.elf_analysis.elf_labeler1.elf_labeler import ElfLabeler
from baderkit.core.utilities.transforms import INT_TO_REV_INT, INT_TO_IMAGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
path = Path(".")

element_df = pd.read_csv("workfunctions.csv")
elements = np.array(element_df["Formula"])
workfunctions = np.array(element_df["Work Function"])
# remove points without workfunction


data_dict = {}


nna_neigh_zeffs = {}
nna_vol_ratios = {}
nna_charges = {}
nna_volumes = {}
nna_dists = {}
nna_fracs = {}

elements_w_results = []

for folder in path.iterdir():
    if not folder.is_dir():
        continue

    subfolder = folder / "static_nospin"
    if not (subfolder / "CHGCAR_sum").exists():
        continue


    subfolder = folder / "static_nospin"
    chgcar = subfolder / "CHGCAR"
    elf = subfolder / "ELFCAR"
    tot = subfolder / "CHGCAR_sum"
    loc = subfolder / "LOCPOT"
    pot = subfolder / "POTCAR"

    element = folder.name
    data_dict[element] = {}
    elements_w_results.append(element)

    labeler = ElfLabeler.from_vasp(
        charge_filename=chgcar,
        reference_filename=elf,
        total_charge_filename=tot,
        persistence_tol=0.001,
        # potential_filename=loc,
        pseudopotential_filename = pot,
        )
    overlap: BasinOverlap = labeler.overlap
    bader = overlap.local_bader
    types = labeler.basin_types
    structure = labeler.reference_grid.structure

    nna_indices = np.array([i for i, j in enumerate(types) if j == "nna"], dtype=np.int64)

    data_dict[element]["nna_zeffs"] = labeler.neighbor_zeffs[nna_indices]
    data_dict[element]["nna_veffs"] = labeler.neighbor_veffs[nna_indices]
    data_dict[element]["nna_vol_ratios"] = labeler.core_volume_ratios[nna_indices]
    data_dict[element]["nna_charges"] = overlap.local_bader.basin_charges[nna_indices]
    data_dict[element]["nna_volumes"] = overlap.local_bader.basin_volumes[nna_indices]
    data_dict[element]["nna_dists"] = labeler.nna_bond_dists
    data_dict[element]["nna_fracs"] = labeler.nna_bond_fracs
    data_dict[element]["nna_dist_beyond_atom"] = labeler.nna_bond_dists * labeler.nna_bond_fracs
    data_dict[element]["nna_potential_energies"] = labeler.nna_potential_energies
    data_dict[element]["nna_potentials"] = labeler.nna_potentials
    data_dict[element]["nna_avg_potentials"] = labeler.nna_avg_potentials
    data_dict[element]["nna_charge_dens"] = labeler.nna_charge_densities
    data_dict[element]["structure"] = structure
    data_dict[element]["grid_shape"] = labeler.reference_grid.shape

    data_dict[element]["test"] = labeler.nna_test

elements_w_results = np.array(elements_w_results)

# remove points without elements or without workfunction values
has_wf = np.where((workfunctions!=np.nan) & np.isin(elements, elements_w_results))[0]
elements = elements[has_wf]
workfunctions = workfunctions[has_wf]

workfunction_indices = np.argsort(workfunctions)
sorted_workfunctions = workfunctions[workfunction_indices]
sorted_elements = elements[workfunction_indices]

for workfunction, element in zip(sorted_workfunctions, sorted_elements):
    data_dict[element]["workfunction"] = workfunction



no_nnas = []
no_workfunction = []

x = []
y = []
y_std = []
for idx in workfunction_indices:
    element = elements[idx]

    # get y-axis
    dict_el = data_dict.get(element, None)
    if dict_el is None:
        continue
    workfunction = dict_el["workfunction"]


    if len(dict_el["nna_dists"]) == 0:
        no_nnas.append(element)
        continue

    if np.isnan(workfunction):
        no_workfunction.append(element)
        continue

    # get x-axis
    x.append(workfunction)

    dists = dict_el["nna_dists"]
    fracs = dict_el["nna_fracs"]
    volumes = dict_el["nna_volumes"]
    charges = dict_el["nna_charges"]
    zeffs = dict_el["nna_zeffs"]
    veffs = dict_el["nna_veffs"]
    vol_ratios = dict_el["nna_vol_ratios"]
    dist_beyond_atom = dict_el["nna_dist_beyond_atom"]
    potential_energies = dict_el["nna_potential_energies"]
    potentials = dict_el["nna_potentials"]
    avg_potentials = dict_el["nna_avg_potentials"]
    structure = dict_el["structure"]
    test = dict_el["test"]
    charge_dens = dict_el["nna_charge_dens"].copy()
    shape = dict_el["grid_shape"]

    cell_vol = structure.volume / shape.prod()
    charge_dens *= cell_vol

    charge_ratios = charges / zeffs

    atom_rhos = zeffs / veffs
    basin_rhos = charges / volumes

    charge_frac = charges / charges.sum()
    volume_frac = volumes / volumes.sum()
    nna_charge_total = charges.sum()

    # value = 1/dists # best
    # value = potential_energies/(charges*2) # most physical
    # value = potentials / dists # best with physical meaning
    value = test


    y.append(np.sum(value*charge_frac))


    # y.append(np.sum(value)/charges.sum())

    # avg = np.mean(value)
    # stddev = np.std(value)
    # y.append(avg)
    # y_std.append(stddev)

x = np.array(x)
y = np.array(y)

plt.scatter(x, y)

# plt.errorbar(
#     x, y,
#     yerr=y_std,
#     fmt='o',        # 'o' = circular markers (scatter style)
#     capsize=4,      # little caps on error bars
#     linestyle='none' # no connecting line
# )

# Calculate line of best fit (degree 1 = linear)
m, b = np.polyfit(x, y, 1)

# Predicted values
y_pred = m * x + b


# R^2 calculation
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Equation text
equation = f"y = {m:.2f}x + {b:.2f}\n$R^2$ = {r2:.3f}"

# Add text to plot
plt.text(min(x), max(y), equation, fontsize=10, verticalalignment='top')

# Plot the line
plt.plot(x, m * x + b)

plt.xlabel("Work Function (eV)")
plt.ylabel("1/dist (1/A^3)")
plt.title("Work Function vs. 1/dist")
# plt.savefig("work_function_vs_dist.png")
plt.show()

# Problem elements:
    # Be - high work function, low potential energy
    # U - lower work function, high potential energy