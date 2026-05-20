# -*- coding: utf-8 -*-

from enum import Enum


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)


# We list all options for methods here so that they are consistent everywhere
class BadelfMethod(str, Enum):
    badelf = "badelf"
    voronelf = "voronelf"
    zero_flux = "zero-flux"

    @classproperty
    def default(cls):
        return cls.badelf
