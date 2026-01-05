# Background

!!! Note
    This page assumes prior knowledge of Bader's method for quantifying atomic charge and oxidation states as well as the Electron Localization Function. See the [Bader](../../bader/background) and [ElfLabeler](../../elf_labeler/background) for more information.


In most cases, Bader's analysis of the charge density suffices for examining local charges. Where it fails is when there is interest in the charge of chemical features which do not appear in the charge density. For example, one may be interested in the charge located at a covalent bond as atoms move closer together, or the charge in the bare electrons of electrides. It is these situations where the Electron Localization Function (ELF) is more useful. Unlike the charge density, the ELF typically displays maxima in chemically interesting areas. This allows each feature to be analyzed independently and can make it easier to visualize interesting bonding interactions.

![elf features](/images/elf_features.png)

Nearly since its conception, the ELF has been used to integrate local charge of atoms and chemical features (see for example, [Kohout et. al.](https://www.lct.jussieu.fr/pagesperso/savin/papers/kohout-elfshells/KohSav-96.pdf)). This is typically accomplished using a technique similar to that of [Bader](../../bader/background). The ELF is first separated into regions called *basins* which contain a singular local maximum, sometimes termed an *attractor*. The charge density in the basins can then be integrated to calculate a local charge. Historically, the separation of basins has been done using the same method as Bader, with regions partitioned by zero-flux surfaces. We have also developed two additional methods which make use of voronoi-like planes (see [Methods](/methods) for details). These methods are the original source of the term BadELF, but it can more generally be used to describe any combination of Bader-like charge analysis of the ELF.

!!! Warning
    Our BadELF implementation performs additional analysis to label chemical features in the ELF to help provide a more directly useful interpretation. It is important to note that for a variety of reasons the final chemical *features* do not always correspond directly with a single basin. For more information, see the [ElfLabeler](../../elf_labeler/background).
