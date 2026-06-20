# -*- coding: utf-8 -*-

from enum import Enum


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)


class DftMethod(str, Enum):
    vasp = "vasp"
    qe = "qe"

    @property
    def wf_reader(self):
        if self == DftMethod.vasp:
            from .vasp_wavecar import read_vasp
            return read_vasp
        elif self == DftMethod.qe:
            from .qe_xml import read_qe
            return read_qe
    
    @property
    def coeff_reader(self):
        if self == DftMethod.vasp:
            from .vasp_wavecar import read_pw_coefficients
            return read_pw_coefficients
        elif self == DftMethod.qe:
            from .qe_xml import read_qe_coefficients
            return read_qe_coefficients
    

