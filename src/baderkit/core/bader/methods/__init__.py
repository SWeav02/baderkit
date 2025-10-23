# -*- coding: utf-8 -*-

from enum import Enum


# We list all options for methods here so that they are consistent everywhere
class Method(str, Enum):
    weight = "weight"
    ongrid = "ongrid"
    neargrid = "neargrid"
    neargrid_weight = "neargrid-weight"
