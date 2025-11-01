# -*- coding: utf-8 -*-

"""
This file defines options for feature labels and settings for colors and
dummy atoms
"""

from enum import Enum
import logging

DOMAIN_COLORS = {
    "reducible - root": "rgba(128, 128, 128, 1)",  # grey
    "reducible": "rgba(128, 128, 128, 1)",  # grey
    "reducible - domain": "rgba(128, 128, 128, 1)",  # grey
    "reducible - dimensionality": "rgba(154, 128, 128, 1)",  # redish grey
    "reducible - contained atoms": "rgba(128, 128, 154, 1)",  # bluish grey
    "irreducible": "rgba(47, 79, 79, 1)", # dark slate gray
    "irreducible - point": "rgba(47, 79, 79, 1)", # dark slate gray
    "irreducible - ring": "rgba(47, 79, 79, 1)", # dark slate gray
    "irreducible - cage": "rgba(47, 79, 79, 1)", # dark slate gray
    }

FEATURE_COLORS = {
    "shell": "rgba(60, 60, 60, 1)", # dark grey
    "deep shell": "rgba(60, 60, 60, 1)", # dark grey
    "core": "rgba(0, 0, 0, 1)", # black
    "covalent": "rgba(0, 255, 255, 1)",  # aqua
    "covalent-metallic": "rgba(101, 191, 224, 1)",  # aqua/gray
    "metallic": "rgba(112, 128, 144, 1)", # slate gray
    "lone-pair": "rgba(128, 0, 128, 1)",  # purple
    "non-nuclear attractor": "rgba(128, 0, 0, 1)",  # maroon
    "electride": "rgba(170, 0, 0, 1)",  # dark red
    "bare electron": "rgba(170, 0, 0, 1)",  # dark red
    }

FEATURE_DUMMY_ATOMS = {
    "unknown": "X",
    "shell": "Xs",
    "deep shell": "Xds",
    "core": "Xc",
    "covalent": "Z",
    "covalent-metallic": "Z",
    "metallic": "M",
    "lone-pair": "Lp",
    "non-nuclear attractor": "Xn",
    "electride": "E",
    "bare electron": "Le"
    }

LINE_COLOR = "rgba(128, 128, 128, 1)" # grey

class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)

class DomainSubtype(str, Enum):
    reducible = "reducible"
    root = "reducible - root"
    reducible_dom = "reducible - domain"
    reducible_dim = "reducible - dimensionality"
    reducible_atom = "reducible - contained atoms"
    irreducible = "irreducible"
    irreducible_point = "irreducible - point"
    irreducible_ring = "irreducible - ring"
    irreducible_cage = "irreducible - cage"
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
    
    @classproperty
    def reducible_types(cls):
        return [
            cls.root,
            cls.reducible,
            cls.reducible_dom,
            cls.reducible_dim,
            cls.reducible_atom,
        ]

    @classproperty
    def irreducible_types(cls):
        return [
            cls.irreducible,
            cls.irreducible_point,
            cls.irreducible_ring,
            cls.irreducible_cage,
        ]
    @classproperty
    def subtypes(cls):
        return {
            "ReducibleNode": cls.reducible_types,
            "IrreducibleNode": cls.irreducible_types,
        } 
    
    @property
    def plot_color(self):
        color = DOMAIN_COLORS.get(self.value, None)
        if color is None:
            logging.warning(f"No plot color found for domain of type {self.name}")
        return color

class FeatureType(str, Enum):
    unknown = "unknown"
    shell = "shell"
    deep_shell = "deep shell"
    core = "core"
    covalent = "covalent"
    covalent_metallic = "covalent-metallic"
    metallic = "metallic"
    lone_pair = "lone-pair"
    non_nuclear_attractor = "non-nuclear attractor"
    electride = "electride"
    bare_electron = "bare electron"
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @classproperty
    def atomic_types(cls):
        return [cls.shell, cls.deep_shell, cls.core, cls.lone_pair]

    @classproperty
    def valence_types(cls):
        return [
            cls.covalent,
            cls.covalent_metallic,
            cls.metallic,
            cls.non_nuclear_attractor,
            cls.electride,
            cls.bare_electron
        ] 
    
    @classproperty
    def bare_types(cls):
        return [
            cls.non_nuclear_attractor,
            cls.electride,
            cls.bare_electron
            ]
    
    @property
    def plot_color(self):
        color = FEATURE_COLORS.get(self.value, None)
        if color is None:
            logging.warning(f"No plot color found for feature of type {self.name}")
        return color
    
    @property
    def dummy_species(self):
        species = FEATURE_DUMMY_ATOMS.get(self.value, None)
        if species is None:
            logging.warning(f"No dummy species label found for feature of type {self.name}. Using 'X'")
            return "X"
        return species
