# The Electron Localization Function

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

## ELF-based Analyses

The ELF can be combined with the electron charge density to gain a significant amount of information from the chemical system. We have developed methods for finding information such as [Bond Polarization](../basin_overlap), [Covalent/Ionic Radii](../elf_radii), [Chemical Feature Labeling](../elf_labeler), [Electride Charge Analysis](../badelf/background), and more!