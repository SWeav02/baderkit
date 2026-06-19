# -*- coding: utf-8 -*-

from enum import Enum


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)


class DftMethod(str, Enum):
    vasp = "vasp"

    @property
    def wf_reader(self):
        if self == DftMethod.vasp:
            from .vasp_wavecar import read_vasp
            return read_vasp
    
    @property
    def coeff_reader(self):
        if self == DftMethod.vasp:
            from .vasp_wavecar import read_pw_coefficients
            return read_pw_coefficients

