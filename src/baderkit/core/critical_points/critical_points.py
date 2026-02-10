# -*- coding: utf-8 -*-

import importlib
import logging
import time
import warnings
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from networkx import MultiDiGraph

from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.toolkit import Structure, Grid
from baderkit.core.bader.bader import Bader

from baderkit.core.utilities.basic import wrap_point_w_shift
from baderkit.core.utilities.interpolation import refine_critical_points

from .critical_points_numba import (
    get_manifold_labels,
    get_saddle_connections,
    get_canonical_saddle_connections,
    get_single_point_saddles,
    INT_TO_IMAGE,
    IMAGE_TO_INT
)

# This allows for Self typing and is compatible with python 3.10
Self = TypeVar("Self", bound="CriticalPoints")


class CriticalPoints(BaseAnalysis):
    """
    Class for running Bader analysis on a regular grid. For information on each
    method, see our [docs](https://sweav02.github.io/baderkit/)

    Parameters
    ----------
    charge_grid : Grid
        The Grid object with the charge density that will be integrated.
    total_charge_grid : Grid | None, optional
        The Grid object used for determining vacuum regions in the system. For
        pseudopotential codes this represents the total electron density and should
        be provided whenever possible. If None, defaults to the charge_grid.
    reference_grid : Grid | None, optional
        The Grid object whose values will be used to construct the basins. This
        should typically only be set when partitioning functions other than the
        charge density (e.g. ELI-D, ELF, etc.).If None, defaults to the
        total_charge_grid.
    vacuum_tol : float | bool, optional
        If a float is provided, this is the value below which a point will
        be considered part of the vacuum. If a bool is provided, no vacuum
        will be used on False, and the default tolerance (0.001) will be used on True.
    persistence_tol : float, optional
        It is common for false maxima to be found using only nearest neighbor
        points. To deal with this we combine pairs of basins that have low
        topological persistence.

        The persistence score is calculated as:

            score = dist * (lower_max - saddle_value) / higher_max

        where 'dist' is the cartesian distance between the maxima, lower_max
        is the value at the maximum with a lower value, saddle_value is the
        highest value at which there is a connection between the two maximas
        descending manifold (basin), and higher_max is the value at the maximum
        with a higher value. Any pairs that score below the tolerance will be
        combined. The default is 0.01.
    grid : Grid | Bader
        A Grid or Bader type object to perform the analysis on. If a Bader object
        is provided, the reference_grid will be used.

    """

    _reset_props = [
        "minima_frac",
        "maxima_frac",
        "saddle1_frac",
        "saddle2_frac",
        "minima_vox",
        "maxima_vox",
        "saddle1_vox",
        "saddle2_vox",
        "critical_vox",
        "critical_frac",
        "critical_values",
        "critical_types",
        "critical_voxel_groups",
        "minima_cart",
        "maxima_cart",
        "saddle1_cart",
        "saddle2_cart",
        "manifold_labels",
        "morse_graph",
        "bond_paths",
    ]

    def __init__(
        self,
        reference_grid: Grid | Bader,
        charge_grid: Grid | None = None,
        total_charge_grid: Grid | None = None,        
        persistence_tol: float = 0.01,
        **kwargs,
    ):
        if type(reference_grid) == Bader:
            bader = reference_grid
            reference_grid=bader.reference_grid
            charge_grid=bader.charge_grid
            total_charge_grid=bader.total_charge_grid
        elif type(reference_grid) == Grid:
            if charge_grid is None:
                charge_grid = reference_grid
            if total_charge_grid is None:
                total_charge_grid = total_charge_grid
            bader = Bader(
                charge_grid=charge_grid,
                total_charge_grid=total_charge_grid,
                reference_grid=reference_grid,
                persistence_tol=persistence_tol, 
                **kwargs,
                )
        self._bader = bader
            
        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs)

        self._persistence_tol = persistence_tol


    @property
    def bader(self) -> Bader:
        return self._bader

    @property
    def manifold_labels(self) -> NDArray[np.uint8]:
        """

        Returns
        -------
        NDArray[np.uint8]
            A mask representing various types of overlaps of the stable/unstable
            3-manifolds of maxima/minima. The labels correspond to:

                0: minima
                1: 1-saddle
                2: 2-saddle
                3: maxima
                4: meeting of 2 minima manifolds (saddle-1 unstable manifold)
                5: meeting of 2 maxima manifolds (saddle-2 stable manifold)
                6: meeting of 2 minima manifolds and 2 maxima basins (1D connections between critical points)
                7: meeting of at least 3 minima manifold borders (saddle-2 unstable manifold)
                8: meeting of at least 3 maxima manifold borders (saddle-1 stable manifold)

                255: overlapping single maxima/minima 3-manifolds

        """
        if self._manifold_labels is None:
            neighbor_transforms, neighbor_dists, _, _ = self.bader.reference_grid.point_neighbor_voronoi_transforms
            maxima_groups, minima_groups = self.bader.get_persistence_groups()

            self._manifold_labels = get_manifold_labels(
                maxima_labels=self.bader.maxima_basin_labels,
                maxima_images=self.bader.maxima_basin_images,
                maxima_groups=maxima_groups,
                minima_labels=self.bader.minima_basin_labels,
                minima_images=self.bader.minima_basin_images,
                minima_groups=minima_groups,
                neighbor_transforms=neighbor_transforms,
                vacuum_mask=self.bader.vacuum_mask,
                
            )

        return self._manifold_labels

    @property
    def maxima_vox(self) -> NDArray:
        return self.bader.maxima_vox
    
    @property
    def maxima_frac(self) -> NDArray:
        return self.bader.maxima_frac

    @property
    def minima_vox(self) -> NDArray:
        return self.bader.minima_vox
    
    @property
    def minima_frac(self) -> NDArray:
        return self.bader.minima_frac

    @property
    def saddle1_vox(self) -> NDArray:
        if self._saddle1_vox is None:
            self._saddle1_vox = self._get_saddle_coords(use_minima=True)
        return self._saddle1_vox
    
    @property
    def saddle1_frac(self) -> NDArray:
        if self._saddle1_frac is None:
            self._saddle1_frac = self._refine_saddles(self.saddle1_vox, 1)
        return self._saddle1_frac

    @property
    def saddle2_vox(self) -> NDArray:
        if self._saddle2_vox is None:
            self._saddle2_vox = self._get_saddle_coords(use_minima=False)
        return self._saddle2_vox
    
    @property
    def saddle2_frac(self) -> NDArray:
        if self._saddle2_frac is None:
            self._saddle2_frac = self._refine_saddles(self.saddle2_vox, 2)
        return self._saddle2_frac
    
    @property
    def critical_vox(self) -> NDArray:
        if self._critical_vox is None:
            # get all critical points
            critical_points = self.maxima_vox
            critical_frac = self.maxima_frac
            critical_types = [3 for i in critical_points]
            for crit_type, i, j in zip(
                    (2,1,0), 
                    (self.saddle2_vox, self.saddle1_vox, self.minima_vox),
                    (self.saddle2_frac, self.saddle1_frac, self.minima_frac),
                    ):
                critical_points = np.append(critical_points, i, axis=0)
                critical_frac = np.append(critical_frac, j, axis=0)
                critical_types.extend([crit_type for j in i])
            critical_types = np.uint8(critical_types)

            # get the groups of voxels within persistence for each critical point
            maxima_groups, minima_groups = self.bader.get_persistence_groups()
            for vox in self.saddle2_vox:
                maxima_groups.append([vox])
            for vox in self.saddle1_vox:
                maxima_groups.append([vox])
            maxima_groups.extend(minima_groups)
            
            # get the indices of each type of critical point in their respective
            # manifolds

            crit_indices = np.arange(len(self.maxima_vox))
            crit_indices = np.append(crit_indices, np.arange(len(self.saddle1_vox)+len(self.saddle2_vox)))
            crit_indices = np.append(crit_indices, np.arange(len(self.minima_vox)))
            
            # get the values at each critical point
            critical_values = self.bader.reference_grid.total[
                critical_points[:,0],
                critical_points[:,1],
                critical_points[:,2],
                ]
            # sort high to low
            ordered = np.flip(np.argsort(critical_values))
            self._critical_voxel_groups = [maxima_groups[i] for i in ordered]
            self._critical_types = critical_types[ordered]
            self._critical_values = critical_values[ordered]
            self._critical_vox = critical_points[ordered]
            self._critical_frac = critical_frac[ordered]
            
            # get dictionary mapping critical indices to ordered indices
            crit_indices = crit_indices[ordered]
            crit_map = {
                "minima": {},
                "maxima": {},
                "saddle": {},
                }
            for total_idx, (crit_idx, crit_type) in enumerate(zip(crit_indices, self._critical_types)):
                if crit_type == 0:
                    crit_map["minima"][crit_idx] = total_idx
                elif crit_type in (1,2):
                    crit_map["saddle"][crit_idx] = total_idx
                else:
                    crit_map["maxima"][crit_idx] = total_idx
            self._crit_map = crit_map
            
        return self._critical_vox
    
    @property
    def critical_frac(self) -> NDArray:
        if self._critical_frac is None:
            self.critical_vox
        return self._critical_frac
    
    @property
    def critical_values(self) -> NDArray:
        if self._critical_values is None:
            self.critical_vox
        return self._critical_values
    
    @property
    def critical_types(self) -> NDArray:
        if self._critical_types is None:
            self.critical_vox
        return self._critical_types
    
    @property
    def critical_voxel_groups(self) -> NDArray:
        if self._critical_voxel_groups is None:
            self.critical_vox
        return self._critical_voxel_groups
    
    @property
    def morse_graph(self) -> MultiDiGraph:
        if self._morse_graph is None:
            self._morse_graph = self._get_morse_graph()
        return self._morse_graph

    def _get_saddle_coords(self, use_minima=False):

        # get neighbors
        neighbor_transforms, neighbor_dists, _, _ = self.bader.reference_grid.point_neighbor_voronoi_transforms
        
        if use_minima:
            labels = self.bader.minima_basin_labels
            images = self.bader.minima_basin_images
        else:
            labels = self.bader.maxima_basin_labels
            images = self.bader.maxima_basin_images
        
        dir2car = np.linalg.inv(self.reference_grid.matrix).T
        # get canonical connection representations of saddles between basins
        possible_saddles, saddle_connections, connection_vals = get_canonical_saddle_connections(
            labels,
            images,
            self.bader.reference_grid.total,
            neighbor_transforms,
            self.manifold_labels,
            dir2car,
            use_minima=use_minima,
        )
        # get unique connections
        unique_connections, inverse = np.unique(
            saddle_connections[:, :3], axis=0, return_inverse=True)
        # get the highest connecting point between basins
        saddle_indices, saddle_vals = get_single_point_saddles(
            data=self.bader.reference_grid.total,
            connection_values=connection_vals,
            saddle_coords=possible_saddles,
            connection_indices=inverse,
            num_connections=len(unique_connections),
            use_minima=use_minima,
        )

        # sort highest to lowest
        sorted_indices = np.flip(np.argsort(saddle_vals))
        saddle_indices = saddle_indices[sorted_indices]
        

        return possible_saddles[saddle_indices]
    
    def _refine_saddles(self, saddle_vox, target_index):
        points = saddle_vox
        data = self.reference_grid.total
        refined_points, refined_status = refine_critical_points(
            points, 
            data,
            target_index,
            is_frac=False
            )
        # refined_points %= self.reference_grid.shape
        if not np.all(refined_status==0):
            logging.warning("Not all critical points successfully refined. Check results with care.")
        frac_points = np.round(refined_points / self.reference_grid.shape, 6)
        frac_points %= 1.0
        return frac_points

    def _get_morse_graph(self) -> MultiDiGraph:
        # get graph
        graph = MultiDiGraph()
        
        # add each critical point
        for crit_idx, (crit_type, crit_vox, crit_frac, crit_val, crit_group) in enumerate(zip(
                self.critical_types, 
                self.critical_vox,
                self.critical_frac,
                self.critical_values,
                self.critical_voxel_groups
                )):
            if crit_type == 0:
                name = "minima"
                # group = "minima"
            elif crit_type == 1:
                name = "saddle1"
                # group = "saddle"
            elif crit_type == 2:
                name = "saddle2"
                # group = "saddle"
            else:
                name = "maxima"
                # group = "maxima"
            graph.add_node(
                crit_idx, 
                type=name,
                vox_coords=crit_vox,
                frac_coords=crit_frac,
                value=crit_val,
                voxel_group=crit_group
                # to add:
                    # shape (point, ring, cage)
                    # color
                )
            
        
        # add connections from saddles to minima/maxima
        neighbor_transforms, neighbor_dists, _, _ = self.bader.reference_grid.point_neighbor_voronoi_transforms
        nx,ny,nz = self.bader.reference_grid.shape
        
        saddle_idx = 0
        for saddle_vox, labels, images, extrema in zip(
                (self.saddle2_vox, self.saddle1_vox),
                (self.bader.maxima_basin_labels, self.bader.minima_basin_labels),
                (self.bader.maxima_basin_images, self.bader.minima_basin_images),
                ("maxima", "minima")
                ):
            for i,j,k in saddle_vox:
                label = labels[i,j,k]
                image = images[i,j,k]
                # get differing neighbor
                label1 = -1; image1 = -1
                for si, sj, sk in neighbor_transforms:
                    ni, nj, nk, ssi, ssj, ssk= wrap_point_w_shift(i+si, j+sj, k+sk, nx, ny, nz)
                    neigh_label = labels[ni,nj,nk]
                    neigh_image = images[ni,nj,nk]
                    # adjust image
                    si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
                    sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
                    sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
                    neigh_image = IMAGE_TO_INT[si1, sj1, sk1]
    
                    if neigh_label != label or neigh_image != image:
                        label1=neigh_label
                        image1=neigh_image
                        break
            
                # get the corresponding minima node indices and images
                saddle_node = self._crit_map["saddle"][saddle_idx]
                node = self._crit_map[extrema][label]
                node1 = self._crit_map[extrema][label1]
                image = INT_TO_IMAGE[image]
                image1 = INT_TO_IMAGE[image1]
                
                # add forward and reverse direction edges to graph
                
                graph.add_edge(
                    saddle_node, 
                    node, 
                    image=image
                    )
                graph.add_edge(
                    saddle_node, 
                    node1, 
                    image=image1
                    )
                # reverse
                graph.add_edge(
                    node, 
                    saddle_node, 
                    image=-image
                    )
                graph.add_edge(
                    node1, 
                    saddle_node, 
                    image=-image1
                    )
                saddle_idx += 1
                
        # finally, we get the connections between saddles
        saddle_connections, saddle_conn_coords = get_saddle_connections(
            self.saddle1_vox,
            self.saddle2_vox,
            neighbor_transforms,
            self.manifold_labels,
        )

        saddle_connections = np.array(saddle_connections, dtype=np.uint16)
        saddle_conn_coords = np.array(saddle_conn_coords, dtype=np.uint16)

        unique_connections, indices = np.unique(saddle_connections[:, :3],axis=0, return_index=True)
        unique_connections = saddle_connections[indices]

        for idx1, idx2, image, is_reverse in unique_connections:
            node1 = self._crit_map["saddle"][idx1]
            node2 = self._crit_map["saddle"][idx2]
            image = INT_TO_IMAGE[image]
            if is_reverse:
                image = -image
            graph.add_edge(
                node1,
                node2,
                image=image
                )
            # reverse
            graph.add_edge(
                node2,
                node1,
                image=-image
                )
        return graph
    
    def to_dict(self):
        pass