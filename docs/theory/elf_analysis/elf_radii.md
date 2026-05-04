# Elf Radii

In the original [BadELF paper](https://pubs.acs.org/doi/10.1021/jacs.3c10876), we noted that the ELF provides a very convenient metric for calculating ionic radii. In particular, we noted that we could use the ELF to calculate ionic radii that on average closely matched the tables of Shannon and Prewitt. Unlike these tables however, this method provides the exact radius in a given system.

![radii_comparison](/images/radii_comparison.png)

The original method worked by determining the minimum closest to the center of the bond. This prevented it from being used for covalent bonds. With the development of the `ElfLabeler` we have updated the method to adapt based on the bonding environment. We consider four scenarios.
1. Ionic Bonds
    The unshared ionic shell of one atom is in direct contact with the core shells of another. The radius is marked at the minima where these basins make contact.
2. Covalent Bonds
    A covalent basin separates the atomic shells of the atoms. The radius is marked at the maximum of this covalent bond.
3. Metallic/Multi-centered Bonds
    The non-nuclear attractor (NNA) associated with the bond sits between multiple atoms and may or may not separate the atomic cores. If it does the radius is determined by the maximum along path between the two atoms that belongs to the NNA. If it doesn't, the bond is treated similar to the ionic situation.
4. Non-bonding
    If the bonding line between two atoms contains basins of other atoms, we consider the system to be non-bonding. We place the radius halfway between the atoms. This has no physical meaning but is used to form complete voronoi-like separations in methods like BadELF.

!!! Note
    The `ElfRadii` class is a work in progress. If you run into issues or have suggestions, please let our team know on our [github issues page](https://github.com/SWeav02/elf-analyzer/issues).
