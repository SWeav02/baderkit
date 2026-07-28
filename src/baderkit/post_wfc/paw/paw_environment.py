# -*- coding: utf-8 -*-

import numpy as np
from baderkit.post_wfc.base_env import BaseWavefunctionEnvironment


class PAWEnvironment(BaseWavefunctionEnvironment):
    """
    Standardized, code-agnostic post-processing driver executing 3D fast Fourier transforms 
    to map discrete reciprocal coefficients into dense localized real-space properties.
    """
    
    @property
    def projection_environment(self):
        if getattr(self, "_projection_environment", None) is None:
            from baderkit.post_wfc.projection.projection_environment import AtomicProjectionEnvironment
            self._projection_environment = AtomicProjectionEnvironment(self)
        return self._projection_environment
    
    def _construct_coefficients(self, ispin: int, ikpt: int, active_bands: list) -> np.ndarray:
        """Pulls native raw coefficients directly out of the binary file stream reader."""
        return self._wf_reader.read_coefficients_batch(ispin, ikpt, active_bands)
    

