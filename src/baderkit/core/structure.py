# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Lattice, Species
from pymatgen.core import Structure as PymatgenStructure

# This allows for Self typing and is compatible with python versions before 3.11
Self = TypeVar("Self", bound="Structure")


class Structure(PymatgenStructure):
    """
    This class is a wraparound for Pymatgen's Structure class with additional
    properties and methods.
    """

    def __init__(
        self,
        lattice: Lattice,
        species: list[Species],
        frac_coords: list[NDArray],
        **kwargs,
    ):
        super().__init__(lattice, species, frac_coords, **kwargs)
        # add labels to sites. This is to add backwards compatability to the
        # relabel_sites method that doesn't exist in earlier versions of pymatgen
        for site in self:
            site.label = site.specie.symbol

    @property
    def labels(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The list of labels for each site

        """
        return [site.label for site in self]

    @labels.setter
    def labels(self, labels: list[str]):
        assert len(labels) == len(self), "Labels must be the same length structure."

    def get_cart_from_miller(self, h: int, k: int, l: int) -> NDArray[float]:
        """
        Gets the cartesian coordinates of the vector perpendicular to the provided
        miller indices

        Parameters
        ----------
        h : int
            First miller index.
        k : int
            Second miller index.
        l : int
            Third miller index.


        Returns
        -------
        NDArray[float]
            The cartesian coordinates of the vector perpendicular to the plane
            defined by the provided miller indices.

        """
        lattice = self.lattice
        # Get three points that define the plane from miller indices. For indices
        # of zero we can just take one of the other points and add 1 along the
        # lattice direction of interest to make a parallel line
        if h != 0:
            a1 = np.array([1 / h, 0, 0])
        else:
            a1 = None
        if k != 0:
            a2 = np.array([0, 1 / k, 0])
        else:
            a2 = None
        if l != 0:
            a3 = np.array([0, 0, 1 / l])
        else:
            a3 = None

        if a1 is None:
            if a2 is not None:
                a1 = a2.copy()
            else:
                a1 = a3.copy()
            a1[0] += 1

        if a2 is None:
            if a1 is not None:
                a2 = a1.copy()
            else:
                a2 = a3.copy()
            a2[1] += 1

        if a3 is None:
            if a1 is not None:
                a3 = a1.copy()
            else:
                a3 = a2.copy()
            a3[2] += 1

        # get real space coords from fractional coords
        a1_real = lattice.get_cartesian_coords(a1)
        a2_real = lattice.get_cartesian_coords(a2)
        a3_real = lattice.get_cartesian_coords(a3)

        vector1 = a2_real - a1_real
        vector2 = a3_real - a1_real
        normal_vector = np.cross(vector1, vector2)
        return normal_vector / np.linalg.norm(normal_vector)

    def relabel_sites(self, ignore_uniq: bool = False) -> Self:
        """
        This method is an exact copy of [Pymatgen's](https://github.com/materialsproject/pymatgen/blob/v2025.5.28/src/pymatgen/core/structure.py).
        It is not available in some older version of pymatgen so I've added it
        to increase the dependency range.

        Relabel sites to ensure they are unique.

        Site labels are updated in-place, and relabeled by suffixing _1, _2, ..., _n for duplicates.
        Call Structure.copy().relabel_sites() to avoid modifying the original structure.


        Parameters
        ----------
        ignore_uniq : bool, optional
            If True, do not relabel sites that already have unique labels.
            The default is False.

        Returns
        -------
        Self
            Structure: self with relabeled sites.

        """

        grouped = defaultdict(list)
        for site in self:
            grouped[site.label].append(site)

        for label, sites in grouped.items():
            if len(sites) == 0 or (len(sites) == 1 and ignore_uniq):
                continue

            for idx, site in enumerate(sites):
                site.label = f"{label}_{idx + 1}"

        return self
