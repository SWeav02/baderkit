# -*- coding: utf-8 -*-

"""
This file defines options for feature types
"""

from enum import Enum


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)


class FeatureType(str, Enum):
    unknown = "unknown"
    ionic = "ionic bond"
    ionic_shell = "ionic shell"
    core = "core"
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
            cls.nna
        ]
    
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
    
    @property
    def is_bonding(self):
        return self in self.bonding

        
