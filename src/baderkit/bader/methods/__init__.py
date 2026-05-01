# -*- coding: utf-8 -*-

from enum import Enum

class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)

# We list all options for methods here so that they are consistent everywhere
class Method(str, Enum):
    weight = "weight"
    ongrid = "ongrid"
    neargrid = "neargrid"
    neargrid_weight = "neargrid-weight"
    
    @classproperty
    def default(cls):
        return cls.neargrid_weight
