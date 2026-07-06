# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np

from baderkit import Structure
from .augmentation_numba import (
    reconstruct_onsite_densities_at_point,
    reconstruct_onsite_densities,
    # precompute_cubic_spline_derivs,
    # fill_atomic_channels_aufbau,
    # compile_ae_radial_profile,
    # broadcast_ae_reference_to_grid,
    # compile_ae_radial_ke_profile,
    # evaluate_ae_reference_at_point,
)


class PAWAugmentationEnvironment:
    """
    Convenience interface managing localized core reconstruction spheres.
    Precomputes both all-electron and pseudo radial cross-product arrays
    once upfront to provide discrete point-wise and grid-wise density evaluations.
    """
    def __init__(self, structure, paw_datasets: dict):
        self.structure = structure
        self.lattice_matrix = self.structure.lattice.matrix
        self.inv_lattice_matrix = np.linalg.inv(self.lattice_matrix)
        self.paw_datasets = paw_datasets
        
        self.unique_elements = list(self.structure.symbol_set)
        element_to_idx = {elem: idx for idx, elem in enumerate(self.unique_elements)}
        
        self.site_element_indices = np.array(
            [element_to_idx[site.species_string] for site in self.structure],
            dtype=np.int64
        )
        
        self.max_cutoffs = np.array(
            [self.paw_datasets[elem].max_cutoff_radius for elem in self.unique_elements],
            dtype=np.float64
        )
        
    @property
    def valence_counts(self):
        """The number of valence electrons assigned to each species"""
        valence_counts = {}
        for element, dataset in self.paw_datasets.items():
            valence_counts[element] = dataset.Z
        return valence_counts
    
    def _prepare_numba_containers(self, density_matrices: list, use_kinetic: bool = False):
        """Extracts and sets up AE/PS component arrays alongside their precomputed cubic spline derivatives."""
        ae_pairs_list = []
        ps_pairs_list = []
        ae_derivs_list = []  # Added derivative tracker arrays
        ps_derivs_list = []
        r_grids_list = []
        l_list = []
        m_list = []
        
        for elem in self.unique_elements:
            dataset = self.paw_datasets[elem]
            if use_kinetic:
                ae_pairs_list.append(dataset.radial_all_electron_ke_pairs_matrix)
                ps_pairs_list.append(dataset.radial_pseudo_ke_pairs_matrix)
                ae_derivs_list.append(dataset.radial_all_electron_ke_pairs_derivs)
                ps_derivs_list.append(dataset.radial_pseudo_ke_pairs_derivs)
            else:
                ae_pairs_list.append(dataset.radial_all_electron_pairs_matrix)
                ps_pairs_list.append(dataset.radial_pseudo_pairs_matrix)
                ae_derivs_list.append(dataset.radial_all_electron_pairs_derivs)
                ps_derivs_list.append(dataset.radial_pseudo_pairs_derivs)
                
            r_grids_list.append(dataset.radial_grid)
            l_list.append(np.asarray(dataset.angular_momenta, dtype=np.int32))
            m_list.append(np.asarray(dataset.magnetic_nums, dtype=np.int32))
            
        occ_matrices_flat_list = []
        for i in range(len(self.structure)):
            flat_occ = np.asarray(density_matrices[i], dtype=np.float64).flatten()
            occ_matrices_flat_list.append(flat_occ)
            
        return (
            ae_pairs_list, ps_pairs_list, 
            ae_derivs_list, ps_derivs_list, 
            r_grids_list, l_list, m_list, occ_matrices_flat_list
        )

    # =========================================================================
    # --- AUGMENTATION METHODS (All-Electron and Pseudo Correction Split) ---
    # =========================================================================

    def calculate_onsite_densities_at_point(
        self, point_cart: list | np.ndarray, density_matrices: list, **kwargs
    ) -> tuple[float, float]:
        """
        Calculates localized core-sphere all-electron and pseudo onsite densities 
        at a single point using cubic splines.
        
        Returns:
            tuple: (all_electron_density, pseudo_density)
        """
        ae_p, ps_p, ae_d, ps_d, r, l, m, occ = self._prepare_numba_containers(
            density_matrices, 
            use_kinetic=kwargs.get('use_kinetic', False)
        )

        return reconstruct_onsite_densities_at_point(
            np.asarray(point_cart, dtype=np.float64), 
            self.lattice_matrix, 
            self.inv_lattice_matrix,
            self.structure.frac_coords, 
            self.site_element_indices, 
            ae_p, ps_p, ae_d, ps_d, r, l, m, occ,
            self.max_cutoffs
        )

    def calculate_onsite_densities(
        self, grid_dims: list | tuple | np.ndarray, density_matrices: list, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Broadcasts localized core-sphere all-electron and pseudo onsite grid fields 
        separately to a pair of 3D arrays using cubic splines.
        
        Returns:
            tuple: (all_electron_grid, pseudo_grid)
        """
        ae_p, ps_p, ae_d, ps_d, r, l, m, occ = self._prepare_numba_containers(
            density_matrices, 
            use_kinetic=kwargs.get('use_kinetic', False)
        )

        return reconstruct_onsite_densities(
            np.array(grid_dims, dtype=np.int64), 
            self.lattice_matrix, 
            self.structure.frac_coords, 
            self.site_element_indices, 
            ae_p, ps_p, ae_d, ps_d, r, l, m, occ,
            self.max_cutoffs
        )

    def calculate_onsite_ke_densities_at_point(
        self, point_cart: list | np.ndarray, density_matrices: list
    ) -> tuple[float, float]:
        """
        Calculates localized core-sphere all-electron and pseudo onsite kinetic energy 
        densities at a single point using cubic splines.
        
        Returns:
            tuple: (all_electron_ke_density, pseudo_ke_density)
        """
        ae_p, ps_p, ae_d, ps_d, r, l, m, occ = self._prepare_numba_containers(
            density_matrices, 
            use_kinetic=True
        )

        return reconstruct_onsite_densities_at_point(
            np.asarray(point_cart, dtype=np.float64), 
            self.lattice_matrix, 
            self.inv_lattice_matrix,
            self.structure.frac_coords, 
            self.site_element_indices, 
            ae_p, ps_p, ae_d, ps_d, r, l, m, occ,
            self.max_cutoffs
        )

    def calculate_onsite_ke_densities(
        self, grid_dims: list | tuple | np.ndarray, density_matrices: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Broadcasts localized core-sphere all-electron and pseudo onsite kinetic energy 
        grid fields separately to a pair of 3D arrays using cubic splines.
        
        Returns:
            tuple: (all_electron_ke_grid, pseudo_ke_grid)
        """
        ae_p, ps_p, ae_d, ps_d, r, l, m, occ = self._prepare_numba_containers(
            density_matrices, 
            use_kinetic=True
        )

        return reconstruct_onsite_densities(
            np.array(grid_dims, dtype=np.int64), 
            self.lattice_matrix, 
            self.structure.frac_coords, 
            self.site_element_indices, 
            ae_p, ps_p, ae_d, ps_d, r, l, m, occ,
            self.max_cutoffs
        )

    @classmethod
    def from_directory(
            cls, 
            directory: Path | str = Path("."), 
            fmt: str = "vasp", 
            **kwargs,
            ):
        directory = Path(directory)
        """Dynamic wf_reader factory routing file stream construction to selected code formats."""
        if fmt == "vasp": 
            from baderkit.post_wfc.pseudopotentials.vasp import parse_vasp_potcar
            paw_dataset = parse_vasp_potcar(directory / "POTCAR")
            structure = Structure.from_file(directory / "POSCAR")
        else: 
            raise ValueError(f"Unknown format profile template string keyword: {fmt}")
            
        return cls(structure=structure, paw_datasets=paw_dataset, **kwargs)