# ELF Labeler - Theory

Our classification of ELF basins into various Lewis-like chemical features is based on three metrics.

1. Topology
    While most maxima in the ELF are points, under the right symmetry they may be ring-like or cage-like.
2. Position
    The location of an ELF basin's maximum is influenced by the atoms around them.
3. Polarization Index
    The degree of polarization can be estimated from the ratio of contributions from each overlaping QTAIM atom with each ELF basin. A value of 0 is fully non-polar while a
    value of 1 is fully polar.

The following table summarized the current labeling scheme.

| Topology   | Position       | Polarization Index       | Result        |
| ---------- | -------------- | ------------------------ | ------------- |
| Point/Ring | Along Bond     | 0-0.5                    | Covalent bond |
| Point/Ring | Along Bond     | 0.5-1.0                  | Ionic bond    |
| Point/Ring | Not Along Bond | <1.0                     | Multi-centered bond|
| Cage       | —              | <1.0                     | Ionic shell   |
| Cage       | —              | 1.0                      | Atomic shell  |
| Point      | Atom center    | 1.0                      | Atomic shell  |
| Point      | Elsewhere      | 1.0                      | Lone pair     |

In reality, many extremely polar features (e.g. lone-pairs, core shells) may have some overlap with
multiple atoms. This decreases the polarization index very slightly, so for features requiring full polarization we apply a small tolerance.

!!! Note
    The 'ElfLabeler' is a work in progress and we are looking to improve it. If you have suggestions or desired features, please let our team know on our [github issues page](https://github.com/SWeav02/elf-analyzer/issues).
