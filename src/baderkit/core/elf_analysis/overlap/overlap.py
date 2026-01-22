# -*- coding: utf-8 -*-

import itertools
import json
import logging
import time

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from pymatgen.analysis.local_env import CrystalNN

from baderkit.core import Bader, Structure
from baderkit.core.bader.methods.shared_numba import get_edges

from .enum_and_styling import LINE_COLOR, DomainSubtype, FeatureType

# from elf_analyzer.core.utilities import IonicRadiiTools
from .graph_numba import (
    find_domain_bifurcations,
    find_domain_connections,
    find_potential_saddle_points,
    get_domains_surrounding_atoms,
)
from .nodes import IrreducibleNode, NodeBase, ReducibleNode


class BasinOverlap:
    """
    A convenience class for calculating the overlap between basins calculated
    in the charge density and a localization density such as ELF.
    """

    def __init__(
        self,
        charge_bader: Bader,
        local_bader: Bader,
    ):

        self.charge_bader = charge_bader
        self.local_bader = local_bader
        
        self._reset_properties()


    ###########################################################################
    # Set Properties
    ###########################################################################
    def _reset_properties(
        self,
        include_properties: list[str] = None,
        exclude_properties: list[str] = [],
    ):
        # if include properties is not provided, we wnat to reset everything
        if include_properties is None:
            include_properties = [
                "atomicities"
            ]
        # get our final list of properties
        reset_properties = [
            i for i in include_properties if i not in exclude_properties
        ]
        # set corresponding hidden variable to None
        for prop in reset_properties:
            setattr(self, f"_{prop}", None)
            
            
    