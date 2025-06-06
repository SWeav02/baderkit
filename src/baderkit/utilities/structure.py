# -*- coding: utf-8 -*-

from pymatgen.core import Structure as PymatgenStructure

import numpy as np

class Structure(PymatgenStructure):
    """
    This class is a wraparound for Pymatgen's Structure class with additional
    properties and methods.
    """
    
    def get_cart_from_miller(self, h, k, l):
        lattice = self.lattice
        # Get three points that define the plane from miller indices. For indices
        # of zero we can just take one of the other points and add 1 along the
        # lattice direction of interest to make a parallel line
        try:
              a1 = np.array([1/h,0,0])
        except: a1 = None
        try:
              a2 = np.array([0,1/k,0])
        except: a2 = None
        try:
              a3 = np.array([0,0,1/l])
        except: a3 = None
        
        if a1 is None:
              try:
                    a1 = a2.copy()
                    a1[0] += 1
              except:
                    a1 = a3.copy()
                    a1[0] += 1
        if a2 is None:
              try:
                    a2 = a1.copy()
                    a2[1] += 1
              except:
                    a2 = a3.copy()
                    a2[1] += 1
        if a3 is None:
              try:
                    a3 = a1.copy()
                    a3[2] += 1
              except:
                    a3 = a2.copy()
                    a3[2] += 1
        # get real space coords from fractional coords
        a1_real = lattice.get_cartesian_coords(a1)
        a2_real = lattice.get_cartesian_coords(a2)
        a3_real = lattice.get_cartesian_coords(a3)
        
        vector1 = a2_real - a1_real
        vector2 = a3_real - a1_real
        normal_vector = np.cross(vector1, vector2)
        return normal_vector/np.linalg.norm(normal_vector)