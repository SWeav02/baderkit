# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:44:39 2026

@author: sammw
"""


from baderkit.core.elf_analysis.elf_labeler1.elf_labeler import ElfLabeler
from baderkit.core import Grid
from pathlib import Path

labeler = ElfLabeler.from_vasp(
    persistence_tol=0.01,
    # potential_filename="LOCPOT",
    )

# potential = labeler.electrostatic_potential
test = labeler.nna_potential_energies
print(test)
# test = labeler.nna_approx_potential_energies