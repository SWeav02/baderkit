# Background

!!! Note
    This page assumes prior knowledge of Bader's method for quantifying atomic charge and oxidation states. See [Bader](../../bader/background) for more information.

## The Electron Localization Function

The Electron Localization Function (ELF), as its name suggests, is a measure of the localization of electrons in a system. It was originally developed in 1990 by [Becke and Edgecombe](https://doi.org/10.1063/1.458517) and was very quickly adopted as a tool for the qualitative and quantitative analysis of chemical features and bonding.

![elf features](/images/elf_features.png)

There are many interpretations of the ELF, but our favorite comes from [J.K. Burdett and T.A. McCormick](https://pubs.acs.org/doi/10.1021/jp9820774). For those who are not so math oriented, don't worry. The equations simplify very quickly.

For a system with spin $\sigma$, the charge density, $\rho_{\sigma}$, and a set of occupied molecular orbitals, ${\psi_{i\sigma}}$, the ELF is given by

$$
\begin{equation}
\text{ELF} = \frac{1}{1 + (\frac{D_{\sigma}}{D_{\sigma,gas}})^2}\
\label{eq:1}
\end{equation}
$$

where,

$$
\begin{equation}
D_{\sigma}/D_{\sigma,gas} = 0.3483\rho^{-5/3}_{\sigma}\Big[\sum_{i}|\nabla\psi_{i\sigma}|^2 - \frac{1}{8}|\nabla\rho_{\sigma}|^2/\rho_{\sigma}\Big]
\label{eq:2}
\end{equation}
$$

The $D_{\sigma}$ is the first term in the Taylor expansion of the spherically averaged $\sigma$-spin pair probability, and the subscript "gas" refers to the corresponding value for the homogenous electron gas. 

The ELF ranges from 0 to 1 and has a value of 0.5 for the homogenous electron gas. Therefore, an ELF value of 1.0 corresponds to a fully localized region and a value of 0.5 to a fully delocalized region. Values below 0.5 are less intuitive. A useful observation is that the first term in the brackets of $\eqref{eq:2}$ is the kinetic energy density at a given point and the latter term is proportional to the Weiszäcker functional (the kinetic energy density at a point if only one occupied orbital were to contribute to $\rho$). The first term is always greater than the second and in practice is typically much larger. This means that for most systems, the dominant part of equation $\eqref{eq:2}$ is

$$
\sum_i\frac{|\nabla\psi_{i\sigma}|^2}{\rho^{5/3}}
$$

Thus, the ELF is primarily dependent on the kinetic energy density (numerator) and charge density (denomenator). As this term is inverted in equation $\eqref{eq:1}$, the ELF increases with charge density and decreases with kinetic energy. This is chemically intuitive, as it suggests high localization where there are many electrons or where electrons are not moving significantly. Another useful note is that the ELF tends to decrease heavily in regions where there are a significant number of nodes in the occupied orbitals. In these regions, the increase in charge density is outweighed by the contribution of nodes to the kinetic energy. Thus, an ELF value below 0.5 can be understood as a region with very little electron density (i.e. interstitial areas between atoms) or a region with a great number of nodes.

---

## Topology of the ELF

!!! note
    This discussion is largely based on this review by Carlo Gatti: [Chemical bonding in crystals: new directions](https://tutorials.crystalsolutions.eu/tutorials/topond/Zeit2005.pdf).

Much like Bader analysis, the ELF can be separated into multiple regions at zero-flux surfaces. These regions are called *basins* and each contains a single local maximum called an *attractor*. Typically, attractors are single points, but in areas of high symmetry they may be rings or cages. Each basin has its own set of useful properties such as charge, volume, position, shape, etc. The advantage of the ELF over the charge density is its ability to show localized features that the charge density does not (e.g. covalent bonds, lone-pairs, electrides, etc.), which can enable more in-depth analysis of chemical interactions.

A helpful tool for analyzing ELF basins is the *bifurcation plot*. The x-axis of a bifurcation plot spans the elf values from 0 to 1. Each value of x defines a set of solids called *domains* where the ELF is greater than or equal to x. As x is increased, some domains split (bifurcate) at saddle points into smaller domains. New domains are represented by points at the x value at which they appear and are connected to their parent by a line. The domains that never split are exactly the same as the basins described earlier. In our bifurcation plots, we represent these as rectangles that span their total ELF depth. Thus, the bifurcation plot maps out the relationships between various basins and their parent domains.

![bifurcation_plot](/images/bifurcation_plot.png)
<p style="line-height: 1;">
<small><strong>The bifurcation plot for Ca<sub>2</sub>N</strong>. The first two bars represent the Ca shells, the third bar the N shell, and the final red bar an electride electron. The electride and N atom form a larger domain at ~0.1 as evidenced by the lines connecting them to a reducible domain. This domain does not connect to the Ca atoms until a lower value of ~0.06. The `depth` of each domain is represented by horizontal lines for reducible domains and the length of the bars for irreducible domains (basins). Note that the Ca and N only display single shells due to the use of pseudopotentials.</small>
</p>

The bifurcation plot provides us with a *depth* for each domain which corresponds to the range of x values in which the domain exists. Our team uses this to make a distinction between *basins* and *chemical features* (or ELF features). Chemical features are what a chemist would likely use to analyze a system such as covalent bonds, lone-pairs, atom shells, etc. In most cases, these features map directly to a single basin, but in some cases two or more basins are separated by a very shallow depth. In these cases, it is often more intuitive to consider the sum of these basins as a single chemical feature. For example, the outer shell of anions is often slighly polarized towards the nearby cations, leading to several very shallow basins.

![depth_plot](/images/LiI_plot.png)
<p style="line-height: 1;">
<small><strong>Depth shematic for LiI</strong>. The central purple atom is iodide and the yellow isosurface is its outermost shell at the given ELF value. The second image shows the bifurcation of the shell into smaller basins. These disappear at a slightly higher ELF value, resulting in a very shallow depth. Because these basins are so shallow relative to the domain formed below 0.56, it is more intuitive to describe them as a single feature.</small>
</p>

---

## Labeling Chemical features

Labeling chemical features in the ELF is best done by rigorous analysis of an expert. Despite this, its beneficial to have a method that automatically guesses the types of features in a system. This is the aim of the `ElfLabeler` class.

Following Bader, we can broadly categorize features into three categories:

1. **Core:** Atomic shells, not including the valence shell.
2. **Bonding:** Features involved in attractive forces. This can further be separated into two categories.
    - *shared:* covalent, metallic, multi-centered, etc.
    - *unshared:* ionic, electrostatic, van der Waals, etc.
3. **Non-bonding:** lone-pairs, f-centers, electrides, etc.

In practice, many of these are difficult to distinguish without human chemical intuition. We have so far developed methods to label only some of these features, and in many cases multiple features may be categorized together. We list the current labels and describe our criteria for labeling them below.

1. **Atomic Shells** appear as point attractor at the center of the atom or as cage attractors fully surrounding an atom. Currently, we do not distinguish between *core* shells and *valence* shells, and this label may also be applied to unshared bonding features.

2. **Covalent Bonds** are point or ring attractors that sit directly along a bond between an atom and its neighbor. This is also a common feature of the outermost shells of anions in ionic bonds, and an arbitrary cutoff must be applied.

3. **Metallic-covalent Bonds** are essentially the same as covalent bonds, but involve a metallic species. They may also have significantly lower charge and localization if they are part of a larger metallic network.

4. **Lone-pairs** typically appear in systems with covalent bonds and join with covalent bonds to form a domain surrounding multiple atoms. In some cases, lone-pairs can also form in ionic systems (e.g. SnO) in which case they will bifurcate from the related atom's outermost shell at low ELF values and have a large depth.

5. **Metallic** features often appear as many, shallow depth point attractors forming a network spread throughout the system. They are very similar to multi-centered bonds and non-bonding features, with arbitrary cutoffs based on depth.

6. **Multi-centered Bonds** sit at interstitial sites between multiple atoms. They are often found in metallic systems, and share many similarities with metallic features. In most cases they are really only distince from metallic features in that they have a greater depth and are more likely to have a higher ELF value.

7. **Bare Electrons** include features like f-centers and electrides. Like metallic features, it is difficult to distinguish them from multi-centered bonds, and we are currently working to determine reasonable criteria for distinguishing them.

!!! Note
    The 'ElfLabeler' is a work in progress and we are looking to improve it. If you have suggestions or desired features, please let our team know on our [github issues page](https://github.com/SWeav02/elf-analyzer/issues).

