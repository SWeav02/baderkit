# Background

!!! Note
    This page assumes prior knowledge of Bader's method for quantifying atomic charge and oxidation states. See [Bader](../../bader/background) for more information.

## The Electron Localization Function

The Electron Localization Function (ELF), as its name suggests, is a measure of the localization of electrons in a system. It was originally developed in 1990 by [Becke and Edgecombe](https://doi.org/10.1063/1.458517) and was very quickly adopted as a tool for the qualitative and quantitative analysis of chemical features and bonding

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

The ELF ranges from 0 to 1 and has a value of 0.5 for the homogenous electron gas. Therefore, an ELF value of 1.0 corresponds to a fully localized region and a value of 0.5 to a fully delocalized region. Values below 0.5 are less intuitive. The key point that will allow us to simplify our current interpretation of the ELF is that the first term of $\eqref{eq:2}$ is the kinetic energy density at a given point and the latter term is proportional to the Weiszäcker functional (the kinetic energy density at a point if only one occupied orbital were to contribute to $\rho$). The first term is always greater than the second and in practice is typically much larger. This means that for most systems, the dominant part of equation $\eqref{eq:2}$ is

$$
\sum_i\frac{|\nabla\psi_{i\sigma}|^2}{\rho^{5/3}}
$$

Thus, the ELF is primarily dependent on the kinetic energy density (numerator) and charge density (denomenator). As this term is inverted in equation $\eqref{eq:1}$, the ELF increases with charge density and decreases with kinetic energy. This is chemically intuitive, as it suggests high localization where there are many electrons or where electrons are not moving significantly. Another useful note is that the ELF tends to decrease heavily in regions where there are a significant number of nodes in the occupied orbitals. The increase in charge density in regions with nodes is far outweighed by the contribution of nodes to the kinetic energy. Thus an ELF value below 0.5 can be understood as a region with very little electron density (i.e. interstitial areas between atoms) or regions with a great number of nodes.

---

## Topology of the ELF

!!! note
 This discussion is largely based on this review by Carlo Gatti: [Chemical bonding in crystals: new directions](https://doi.org/10.1524/zkri.220.5.399.65073).
 
The advantage of the ELF over the charge density is its ability to show localized features that the charge density does not (e.g. covalent bonds, lone-pairs, electrides, etc.). Historically, however, it has been a very time consuming process to sort through the ELF and develop a clear chemical picture. We therefore strive to design a program that automates this process. To begin, it is necessary to highlight some of the topological methods used to descuss the ELF.

Much like Bader analysis, the ELF can be separated into multiple regions at zero-flux surfaces. These regions are called *basins* and each contains a single local maximum called an *attractor*. Typically, attractors are single points, but in areas of high symmetry they may be rings or cages. Each basin has a number of useful properties that can help interperet chemical interactions such as a charge, volume, position, and shape.

A helpful tool for analyzing ELF basins is the *bifurcation plot*. The x-axis of a bifurcation plot spans the elf values from 0 to 1. Each value of x defines a set of solids called *domains* where the ELF is greater than or equal to x. As x is increased, some domains split (bifurcate) at saddle points into smaller domains. New domains are represented by points at the x value at which they appear and are connected to their parent by a line. The domains that never split at higher values are exactly the same as the basins described earlier. Thus, the bifurcation plot maps out the relationship between various basins.

*** INSERT PLOT

The bifurcation plot provides us with a sense of *depth* for each domain which corresponds to the range of x values in which a domain exists. This allows us to make a very important distinction between a *basin* and what our team describes as a *chemical feature*. Chemical features are what a chemist would like to use to analyze a system such as covalent bonds, lone-pairs, atom shells, etc. In most cases, these features map directly to a single basin (e.g. single covalent bonds). In some cases however, two or more basins may combine at relatively shallow depths. Often the new domain has a significantly larger depth and it is more chemically intuitive to think of the combined domain as a single entity. For example, this often happens in the highly polarizable shells of larger atoms such as iodide.

*** INSERT PLOT

Because of this, we combine basins that are significantly more shallow than their parent domains into individual features (this also helps with voxelation). All of the properties of the child basins are combined into one making it easier to pull out coherent results.

!!! Note
    This means that in our interpretation *basins* are NOT always *chemical features*

## Labeling Chemical features

Following Bader, we can broadly categorize features into three categories: *core*, *bonding*, and *non-bonding*. Cores include the atomic shells. Bonding can be split into shared (e.g. covalent and metallic bonds) and unshared (e.g. ionic, hydrogen, electrostatic, and van der Waals bonds). Non-bonding includes features that do not contribute to a bond or shell (e.g. lone-pairs, f-centers, electrides). 

