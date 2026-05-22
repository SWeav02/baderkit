# -*- coding: utf-8 -*-

"""
This file defines options for feature types
"""

import logging
from enum import Enum


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)


class FeatureType(str, Enum):
    unknown = "unknown"
    ionic = "ionic bond"
    ionic_shell = "ionic shell"
    core = "core shell"
    covalent = "covalent bond"
    metallic = "metallic bond"
    lone_pair = "lone-pair"
    multi_centered = "multi-centered bond"
    electride = "electride"
    nna = "non-nuclear attractor"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @classproperty
    def shared(cls):
        return [cls.covalent, cls.metallic, cls.multi_centered]

    @classproperty
    def unshared(cls):
        return [cls.ionic, cls.ionic_shell, cls.core, cls.lone_pair, cls.electride]

    @classproperty
    def valence(cls):
        return [
            cls.covalent,
            cls.metallic,
            cls.lone_pair,
            cls.ionic,
            cls.ionic_shell,
            cls.multi_centered,
            cls.electride,
            cls.nna,
        ]

    @classproperty
    def metal_like(cls):
        return [cls.metallic, cls.multi_centered, cls.electride, cls.nna]

    @classproperty
    def bonding(cls):
        return [
            cls.covalent,
            cls.metallic,
            cls.ionic,
            cls.ionic_shell,
            cls.multi_centered,
            cls.nna,
        ]

    @classproperty
    def _FEATURE_DUMMY_ATOMS(cls):
        return {
            cls.unknown: "X",
            cls.ionic: "Xs",
            cls.ionic_shell: "Xs",
            cls.core: "Xs",
            cls.covalent: "Xc",
            cls.metallic: "Xm",
            cls.lone_pair: "Xlp",
            cls.multi_centered: "Xmc",
            cls.electride: "Xmc",
            cls.nna: "Xmc",
        }

    @property
    def is_bonding(self):
        return self in self.bonding

    @property
    def dummy_species(self):
        species = self._FEATURE_DUMMY_ATOMS.get(self, None)
        if species is None:
            logging.warning(
                f"No dummy species label found for feature of type {self.name}. Using 'X'"
            )
            return "X"
        return species
