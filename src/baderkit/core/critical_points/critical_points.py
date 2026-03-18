# -*- coding: utf-8 -*-

from typing import TypeVar

import numpy as np
from networkx import MultiDiGraph
from numpy.typing import NDArray

from baderkit.core.bader.bader import Bader
from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.toolkit import Grid
from baderkit.core.utilities.basic import merge_frac_coords_weighted
from baderkit.core.utilities.basins import get_manifold_labels
from baderkit.core.utilities.transforms import INT_TO_IMAGE

from .saddle_connections import (
    get_saddle_saddle_connections,
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
        "minima_vox",
        "minima_cart",
        "minima_values",
        "minima_persistence_groups",
        "minima_group_types",
        "saddle1_frac",
        "saddle1_vox",
        "saddle1_cart",
        "saddle1_values",
        "saddle1_persistence_groups",
        "saddle1_group_types",
        "saddle2_frac",
        "saddle2_vox",
        "saddle2_cart",
        "saddle2_values",
        "saddle2_persistence_groups",
        "saddle2_group_types",
        "maxima_frac",
        "maxima_vox",
        "maxima_cart",
        "maxima_values",
        "maxima_persistence_groups",
        "maxima_group_types",
        "saddle1_minima_connections",
        "saddle2_maxima_connections",
        "saddle_saddle_connections",
        "manifold_labels",
        "morse_graph",
        "bond_paths",
    ]

    def __init__(
            self,
            charge_grid: Grid,
            reference_grid: Grid | None = None,
            total_charge_grid: Grid | None = None,
            **kwargs,
            ):

        bader = Bader(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )
        self._bader = bader

        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )

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
            neighbor_transforms, neighbor_dists, _, _ = (
                self.bader.reference_grid.point_neighbor_voronoi_transforms
            )

            # neighbor_transforms, neighbor_dists = self.bader.reference_grid.point_neighbor_transforms
            self._manifold_labels = get_manifold_labels(
                maxima_labels=self.bader.maxima_basin_labels,
                maxima_images=self.bader.maxima_basin_images,
                maxima_groups=self.maxima_persistence_groups,
                minima_labels=self.bader.minima_basin_labels,
                minima_images=self.bader.minima_basin_images,
                minima_groups=self.minima_persistence_groups,
                neighbor_transforms=neighbor_transforms,
                vacuum_mask=self.bader.vacuum_mask,
            )

        return self._manifold_labels

    ####################################################################################################
    # Minima
    ####################################################################################################
    @property
    def minima_vox(self) -> NDArray:
        return self.bader.minima_vox

    @property
    def minima_frac(self) -> NDArray:
        return self.bader.minima_frac

    @property
    def minima_cart(self) -> NDArray:
        return self.bader.minima_cart

    @property
    def minima_values(self) -> NDArray:
        return self.bader.minima_ref_values

    @property
    def minima_group_types(self):
        if self._minima_group_types is None:
            self._get_extrema_persistence_groups()
        return self._minima_group_types

    @property
    def minima_persistence_groups(self):
        if self._minima_persistence_groups is None:
            self._get_extrema_persistence_groups()
        return self._minima_persistence_groups

    ####################################################################################################
    # 1-Saddles
    ####################################################################################################
    @property
    def saddle1_vox(self) -> NDArray:
        return self.bader.saddle1_vox

    @property
    def saddle1_frac(self) -> NDArray:
        return self.bader.saddle1_frac

    @property
    def saddle1_cart(self) -> NDArray:
        return self.bader.saddle1_cart

    @property
    def saddle1_values(self) -> NDArray:
        return self.bader.saddle1_ref_values

    @property
    def saddle1_group_types(self):
        if self._saddle1_group_types is None:
            self._saddle1_group_types = np.array(
                [0 for i in range(len(self.saddle1_frac))]
            )
        return self._saddle1_group_types

    @property
    def saddle1_persistence_groups(self):
        if self._saddle1_persistence_groups is None:
            self._saddle1_persistence_groups = [
                i.reshape((1, -1)) for i in self.saddle1_vox
            ]
        return self._saddle1_persistence_groups

    @property
    def saddle1_minima_connections(self):
        if self._saddle1_minima_connections is None:
            conns = self.bader.saddle1_connections
            saddle_indices = np.arange(len(conns))
            # get first part
            ext1 = conns[:, 0]
            images1 = conns[:, 2]

            ext2 = conns[:, 1]
            images2 = conns[:, 3]

            conns1 = np.column_stack((saddle_indices, ext1, images1))
            conns2 = np.column_stack((saddle_indices, ext2, images2))

            self._saddle1_minima_connections = np.append(conns1, conns2, axis=0)
        return self._saddle1_minima_connections

    ####################################################################################################
    # 2-Saddles
    ####################################################################################################

    @property
    def saddle2_vox(self) -> NDArray:
        return self.bader.saddle2_vox

    @property
    def saddle2_frac(self) -> NDArray:
        return self.bader.saddle2_frac

    @property
    def saddle2_cart(self) -> NDArray:
        return self.bader.saddle2_cart

    @property
    def saddle2_values(self) -> NDArray:
        return self.bader.saddle2_ref_values

    @property
    def saddle2_group_types(self):
        if self._saddle2_group_types is None:
            self._saddle2_group_types = np.array(
                [0 for i in range(len(self.saddle2_frac))]
            )
        return self._saddle2_group_types

    @property
    def saddle2_persistence_groups(self):
        if self._saddle2_persistence_groups is None:
            self._saddle2_persistence_groups = [
                i.reshape((1, -1)) for i in self.saddle2_vox
            ]
        return self._saddle2_persistence_groups

    @property
    def saddle2_maxima_connections(self):
        if self._saddle2_maxima_connections is None:
            conns = self.bader.saddle2_connections
            saddle_indices = np.arange(len(conns))
            # get first part
            ext1 = conns[:, 0]
            images1 = conns[:, 2]

            ext2 = conns[:, 1]
            images2 = conns[:, 3]

            conns1 = np.column_stack((saddle_indices, ext1, images1))
            conns2 = np.column_stack((saddle_indices, ext2, images2))

            self._saddle2_maxima_connections = np.append(conns1, conns2, axis=0)
        return self._saddle2_maxima_connections

    @property
    def saddle_saddle_connections(self):
        if self._saddle_saddle_connections is None:
            self._get_saddle_saddle_connections()
        return self._saddle_saddle_connections

    ####################################################################################################
    # 2-Saddles
    ####################################################################################################

    @property
    def maxima_vox(self) -> NDArray:
        return self.bader.maxima_vox

    @property
    def maxima_frac(self) -> NDArray:
        return self.bader.maxima_frac

    @property
    def maxima_cart(self) -> NDArray:
        return self.bader.maxima_cart

    @property
    def maxima_values(self) -> NDArray:
        return self.bader.maxima_ref_values

    @property
    def maxima_persistence_groups(self):
        if self._maxima_persistence_groups is None:
            self._get_extrema_persistence_groups()
        return self._maxima_persistence_groups

    @property
    def maxima_group_types(self):
        if self._maxima_group_types is None:
            self._get_extrema_persistence_groups()
        return self._maxima_group_types

    @property
    def morse_graph(self) -> MultiDiGraph:
        if self._morse_graph is None:
            self._morse_graph = self._get_morse_graph()
        return self._morse_graph

    def _update_persistence_groups(
        self,
        groups,
        betti_numbers,
        base_extrema,
    ):

        # classify each group's type
        new_groups = []
        group_types = []

        for group, base_coord, betti in zip(groups, base_extrema, betti_numbers):

            rings = betti[1]
            holes = betti[2]
            # check for points
            if rings == 0 and holes == 0:
                group_type = 0
            # check for rings
            elif rings == 1 and holes == 0:
                group_type = 1
            # check for hollow cage
            elif rings == 0 and holes == 1:
                group_type = 2
            # other shapes (multiple rings, holes and rings) we will treat similar
            # to our standard cages, so we mark everything else this way
            # else:
            #     group_type = 2

            # append our groups and types
            group_types.append(group_type)
            if group_type == 0:
                # only append the point
                new_groups.append(base_coord.reshape(1, -1))
            else:
                new_groups.append(group)

        return new_groups, group_types

    def _get_extrema_persistence_groups(self):

        maxima_betti, minima_betti, maxima_groups, minima_groups = (
            self.bader.get_betti_numbers(return_groups=True)
        )

        maxima_vox = self.bader.maxima_vox
        minima_vox = self.bader.minima_vox

        new_maxima_groups, maxima_group_types = self._update_persistence_groups(
            maxima_groups, maxima_betti, maxima_vox
        )
        new_minima_groups, minima_group_types = self._update_persistence_groups(
            minima_groups,
            minima_betti,
            minima_vox,
        )
        self._maxima_persistence_groups = new_maxima_groups
        self._minima_persistence_groups = new_minima_groups
        self._maxima_group_types = maxima_group_types
        self._minima_group_types = minima_group_types

    def _get_saddle_saddle_connections(self):
        neighbor_transforms, _ = self.reference_grid.point_neighbor_transforms
        saddle_connections, saddle_conn_coords = get_saddle_saddle_connections(
            saddle1_coords=self.saddle1_vox,
            saddle2_coords=self.saddle2_vox,
            neighbor_transforms=neighbor_transforms,
            edge_mask=self.manifold_labels,
        )

        saddle_connections = np.array(saddle_connections, dtype=np.uint16)
        saddle_conn_coords = np.array(saddle_conn_coords, dtype=np.uint16)

        unique_connections, indices = np.unique(
            saddle_connections, axis=0, return_index=True
        )
        unique_connections = saddle_connections[indices]

        self._saddle_saddle_connections = unique_connections

    def _get_morse_graph(self) -> MultiDiGraph:
        # get graph
        graph = MultiDiGraph()

        # add each critical point
        for crit_type, crit_name in enumerate(
            ["minima", "saddle1", "saddle2", "maxima"]
        ):
            voxels = getattr(self, f"{crit_name}_vox")
            fracs = getattr(self, f"{crit_name}_frac")
            values = getattr(self, f"{crit_name}_values")
            groups = getattr(self, f"{crit_name}_persistence_groups")
            group_types = getattr(self, f"{crit_name}_group_types")

            for crit_idx, (vox, frac, val, group, group_type) in enumerate(
                zip(
                    voxels,
                    fracs,
                    values,
                    groups,
                    group_types,
                )
            ):
                # if we have a ring/cage, we want to get the average coords
                # so edges display reasonably well.
                if group_type != 0:
                    group_frac = group / self.reference_grid.shape
                    # adjust to be as close to each other as possible
                    ref = vox / self.reference_grid.shape
                    group_frac = group_frac - np.round(group_frac - ref)
                    # convert back to voxel coords
                    group = (group_frac * self.reference_grid.shape).astype(np.int32)
                    nx, ny, nz = self.reference_grid.shape
                    # also get the average frac coord when this group is combined
                    frac = merge_frac_coords_weighted(
                        frac_coords=group_frac,
                        values=self.reference_grid.total[
                            group[:, 0] % nx,
                            group[:, 1] % ny,
                            group[:, 2] % nz,
                        ],
                        ref_coord=ref,
                        wrap=False,
                    )
                cart = frac @ self.reference_grid.matrix

                graph.add_node(
                    f"{crit_name}_{crit_idx}",
                    type_idx=crit_type,
                    vox_coords=vox,
                    frac_coords=frac,
                    cart_coords=cart,
                    value=val,
                    voxel_group=group,
                    group_type=group_type,
                )

        # add connections from saddles to extrema
        saddle1_conns = self.saddle1_minima_connections
        saddle2_conns = self.saddle2_maxima_connections

        # Add saddle1 to minima connections
        for saddle_idx, min_idx, image in saddle1_conns:
            saddle_node = f"saddle1_{saddle_idx}"
            min_node = f"minima_{min_idx}"

            # get saddle coord and minima group
            saddle_frac = graph.nodes[saddle_node]["frac_coords"]
            min_group_frac = (
                graph.nodes[min_node]["voxel_group"] / self.reference_grid.shape
            )
            min_group_frac += INT_TO_IMAGE[image]

            # get shortest distance
            d2 = np.sum(
                (min_group_frac - saddle_frac) ** 2, axis=1
            )  # squared distances
            idx = np.argmin(d2)
            closest_coord = min_group_frac[idx]

            # add forward and reverse direction edges to graph
            # saddle to minima
            graph.add_edge(
                saddle_node,
                min_node,
                p0=saddle_frac,
                p1=closest_coord,
            )

            # reverse
            # minima to saddle
            graph.add_edge(
                min_node,
                saddle_node,
                p1=saddle_frac,
                p0=closest_coord,
            )

        # Add saddle2 to maxima connections
        for saddle_idx, max_idx, image in saddle2_conns:
            saddle_node = f"saddle2_{saddle_idx}"
            max_node = f"maxima_{max_idx}"

            # get saddle coord and maxima group
            saddle_frac = graph.nodes[saddle_node]["frac_coords"]
            max_group_frac = (
                graph.nodes[max_node]["voxel_group"] / self.reference_grid.shape
            )
            max_group_frac += INT_TO_IMAGE[image]

            # get shortest distance
            d2 = np.sum(
                (max_group_frac - saddle_frac) ** 2, axis=1
            )  # squared distances
            idx = np.argmin(d2)
            closest_coord = max_group_frac[idx]

            # add forward and reverse direction edges to graph
            # saddle to maxima
            graph.add_edge(
                saddle_node,
                max_node,
                p0=saddle_frac,
                p1=closest_coord,
            )

            # reverse
            # maxima to saddle
            graph.add_edge(
                max_node,
                saddle_node,
                p1=saddle_frac,
                p0=closest_coord,
            )

        # add saddle1 to saddle2
        saddle_saddle_connections = self.saddle_saddle_connections
        for saddle1_idx, saddle2_idx, image in saddle_saddle_connections:
            saddle1_node = f"saddle1_{saddle1_idx}"
            saddle2_node = f"saddle2_{saddle2_idx}"

            saddle1_frac = graph.nodes[saddle1_node]["frac_coords"]
            saddle2_frac = (
                graph.nodes[saddle2_node]["frac_coords"] + INT_TO_IMAGE[image]
            )

            # saddle1 to saddle2
            graph.add_edge(saddle1_node, saddle2_node, p0=saddle1_frac, p1=saddle2_frac)

            # saddle2 to saddle1
            graph.add_edge(saddle2_node, saddle1_node, p1=saddle1_frac, p0=saddle2_frac)

        return graph

    def to_dict(self):
        pass
