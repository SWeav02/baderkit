# QTAIM/Local Overlap

All of our ELF analysis methods start from the QTAIM/ELF overlap theory developed by [Raub and Jansen](https://link.springer.com/article/10.1007/s002140100268) and [Wagner et. al.](https://pubs.acs.org/doi/10.1021/acs.inorgchem.5b00135). Any localization function such as the ELF, ELI-D, or LOL can be used. We will use ELF as a general term here.

## Overlap Definition
First the electron charge density $\rho$ and ELF $\sigma$ are separated into Bader basins $\Omega^\rho_i$ and $\Omega^\sigma_j$ respectively. Here $i$ and $j$ refer to the basin indices. Typically, $\rho$ is topologically simple and each $\Omega^\rho_i$ corresponds to a single atom in the system. The $\rho$ is usually much more complex and each $\Omega^\sigma_j$ basin corresponds to some chemical feature such as an atomic shell, covalent bond, lone-pair, or other non-nuclear attractors (NNAs). These basins can be overlayed to construct overlap basins.

$$
\Omega_s=\Omega^\rho_i\ \cap \Omega^\sigma_j\
$$

Each ELF basin consists of one or more overlap basin(s). Each overlap basins corresponds to one QTAIM atom and represents the charge in that overlap basin represents the portion of the ELF basin that 'belongs' to the QTAIM atom.

## Uses
The overlap basins provide a simple, yet powerful metric for how polarized an ELF basin is. If two atoms take equal parts of an ELF basin, that ELF basin is non-polar as in the case of a covalent bond. If the ELF basin is only overlapped by a single atomic basin it is fully polar like a lone-pair or atomic shell. In between this are many degrees of polarization accounting for ionic bonding and hetero-covalent bonds.

This forms the basis for our `ElfLabeler` class which uses the QTAIM/ELF overlap to determine what chemical features exist in the system. This information is then used to calculate the radii or oxidation states in the `ElfRadii` and `Badelf` classes.