# -*- coding: utf-8 -*-

import itertools
import json
import logging
import time

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from pymatgen.analysis.local_env import CrystalNN

from baderkit.core import Bader, Structure
from baderkit.core.bader.methods.shared_numba import get_edges

from .enum_and_styling import LINE_COLOR, DomainSubtype, FeatureType

# from elf_analyzer.core.utilities import IonicRadiiTools
from .graph_numba import (
    find_domain_bifurcations,
    find_domain_connections,
    find_potential_saddle_points,
    get_domains_surrounding_atoms,
)
from .nodes import IrreducibleNode, NodeBase, ReducibleNode


class BifurcationGraph:
    """
    A convenience class for storing the nodes of a bifurcation graph and gathering
    data on them.

    The nodes themselves contain the information on their connectivity.

    Parameters
    ----------
    structure : Structure
        The PyMatGen structure of the chemical system.
    basin_maxima_frac : NDArray[float]
        The fractional coordinates of the Bader basins in the system.
    basin_charges : NDArray[float]
        The integrated charges of the Bader basins in the system.
    basin_volumes : NDArray[float]
        The volumes of the Bader basins in the system.
    crystalnn_kwargs : dict
        The keyword arguments for the CrystalNN object used for finding neighbors.
        Particularly important when calculating charges assigned to each atom.
    atomic_radii : NDArray[float], optional
        The radii of each atom in the system used to calculate distance beyond
        the atom for each feature. The default is None.
    """

    def __init__(
        self,
        structure: Structure,
        basin_maxima_frac: NDArray[float],
        basin_charges: NDArray[float],
        basin_volumes: NDArray[float],
        crystalnn_kwargs: dict,
        atomic_radii: NDArray[float] = None,
        remove_cutoff: float = 0.05,
        reduce_cutoff: float = 0.10,
        **kwargs,
    ):

        self._root_nodes = []
        self._nodes = []
        self._node_keys = {}

        self.structure = structure
        self.basin_maxima_frac = basin_maxima_frac
        self.basin_charges = basin_charges
        self.basin_volumes = basin_volumes
        self.crystalnn_kwargs = crystalnn_kwargs
        self.cnn = CrystalNN(**crystalnn_kwargs)
        
        self._remove_cutoff = remove_cutoff
        self._reduce_cutoff = reduce_cutoff

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, key):
        return self.nodes[key]

    def __contains__(self, key):
        return key in self.nodes

    def __repr__(self):
        return f"BifurcationGraph(num_nodes={len(self._nodes)})"

    @property
    def root_nodes(self) -> list:
        """

        Returns
        -------
        list
            A list of nodes that have no parents

        """
        return self._root_nodes

    @property
    def nodes(self) -> list:
        """

        Returns
        -------
        list
            The list of nodes in the graph.

        """
        return self._nodes

    def node_from_key(self, key: int) -> list:
        """
        Returns a node in the graph given its key.

        Parameters
        ----------
        key : int
            The integer key of the node.

        Returns
        -------
        Node
            The node with the corresponding key.

        """
        return self._node_keys[key]

    @property
    def irreducible_nodes(self) -> list:
        """

        Returns
        -------
        list
            The list of irreducible nodes in the graph i.e. the nodes that do
            not have any children.

        """
        return [i for i in self if not i.is_reducible]

    @property
    def reducible_nodes(self) -> list:
        """

        Returns
        -------
        list
            The list of reducible nodes in the graph i.e. the nodes that have
            child nodes.

        """
        return [i for i in self if i.is_reducible]

    @property
    def sorted_reducible_nodes(self) -> list:
        """

        Returns
        -------
        list
            A list of reducible nodes sorted by minimum value.

        """
        nodes = self.reducible_nodes
        min_vals = [i.min_value for i in nodes]
        sorted_indices = np.argsort(min_vals)

        return [nodes[i] for i in sorted_indices]

    @property
    def unassigned_nodes(self) -> list:
        """

        Returns
        -------
        list
            The list of irreducible nodes that don't have an assigned feature type.

        """
        return [
            i
            for i in self
            if not i.is_reducible and i.feature_type in (None, FeatureType.unknown)
        ]

    @property
    def electride_structure(self) -> Structure:
        """

        Returns
        -------
        structure : Structure
            The PyMatGen Structure of the material with dummy atoms placed at
            highly localized off atom electrons that may behave like electride-atoms.
            (e.g. electrides)

        """
        structure = self.structure.copy()
        for node in self.get_feature_nodes(feature_types=FeatureType.bare_types):
            structure.append(node.feature_type.dummy_species, node.average_frac_coords)
        return structure

    @property
    def _electride_hatom_structure(self) -> Structure:
        """

        Returns
        -------
        structure : Structure
            The PyMatGen Structure of the material with H- dummy atoms placed at
            highly localized off atom electrons that may behave like electride-atoms.
            (e.g. electrides)

        """
        # same as above, but with H- dummy atoms for approximate radii in crystalNN
        structure = self.structure.copy()
        for node in self.get_feature_nodes(feature_types=FeatureType.bare_types):
            structure.append("H-", node.average_frac_coords)
        return structure

    @property
    def labeled_structure(self) -> Structure:
        """

        Returns
        -------
        structure : Structure
            The PyMatGen Structure of the material with dummy atoms placed at
            the maximum of each labeled feature.

        """
        structure = self.structure.copy()
        for node in self.irreducible_nodes:
            structure.append(node.feature_type.dummy_species, node.average_frac_coords)
        return structure

    def get_feature_nodes(self, feature_types: list[FeatureType]):
        """
        Gets a list of nodes from the requested list of FeatureType

        Parameters
        ----------
        feature_types : list[FeatureType]
            The list of FeatureTypes to get nodes from.

        Returns
        -------
        list
            The list of nodes of the requested type.

        """
        return [
            i for i in self if not i.is_reducible and i.feature_type in feature_types
        ]
###############################################################################
# To Methods
###############################################################################

    def to_dict(self) -> dict:
        """
        Gets a dictionary representation of the BifurcationGraph

        Returns
        -------
        dict
            The dictionary representation of the BifurcationGraph.

        """
        graph_dict = {
            "nodes": [i.to_dict() for i in self],
            "structure": self.structure.to_json(),
            "crystalnn_kwargs": self.crystalnn_kwargs,
        }
        # convert array props to python list/int for json
        for prop_str in [
            "basin_maxima_frac",
            "basin_charges",
            "basin_volumes",
            "atomic_radii",
        ]:
            prop = getattr(self, prop_str, None)
            if prop is not None:
                prop = prop.tolist()
            graph_dict[prop_str] = prop

        return graph_dict

    def to_json(self) -> str:
        """
        Creates a json string representation of the graph.

        Returns
        -------
        str
            A json string representation of the graph.

        """
        return json.dumps(self.to_dict())
    
###############################################################################
# From Methods
###############################################################################

    @classmethod
    def from_dict(cls, graph_dict: dict):
        """
        Creates a BifurcationGraph object from a dictionary representation.

        Parameters
        ----------
        graph_dict : dict
            A dictionary representation of the graph.

        Returns
        -------
        graph : BifurcationGraph
            The graph object created from the dictionary representation.

        """

        nodes = graph_dict.pop("nodes")
        graph_dict["structure"] = Structure.from_str(
            graph_dict["structure"], fmt="json"
        )
        for prop_str in [
            "basin_maxima_frac",
            "basin_charges",
            "basin_volumes",
            "atomic_radii",
        ]:
            prop = graph_dict.get(prop_str, None)
            if prop is not None:
                graph_dict[prop_str] = np.array(prop, dtype=np.float64)

        # create our initial graph object
        graph = cls(**graph_dict)

        # add nodes
        for node_dict in nodes:
            NodeBase.from_dict(graph, node_dict)

        return graph

    @classmethod
    def from_json(cls, json_str: str):
        """
        Creates a BifurcationGraph from a json string representation.

        Parameters
        ----------
        json_str : str
            A json string representation of a graph.

        Returns
        -------
        BifurcationGraph
            The graph object created from the json representation.

        """
        graph_dict = json.loads(json_str)
        return cls.from_dict(graph_dict)

    @classmethod
    def from_bader(cls, charge_bader: Bader, elf_bader: Bader, **kwargs):
        """
        Creates a BifurcationGraph from the Bader method results of the charge
        density and ELF (or ELI-D, LOL, etc.)

        Parameters
        ----------
        charge_bader : Bader
            A Bader object partitioning over the charge density
        elf_bader : Bader
            A bader object partitioning over the ELF (or ELI-D, LOL, etc.)

        Returns
        -------
        graph : BifurcationGraph
            The graph object created from the Bader results.

        """
        #######################################################################
        # Run Bader for clean logging
        #######################################################################
        _ = charge_bader.atom_labels
        _ = elf_bader.atom_labels
        
        # get convenience references
        elf_grid = elf_bader.reference_grid
        neighbor_transforms, _ = elf_grid.point_neighbor_transforms

        #######################################################################
        # Get Bifurcation Values and Corresponding domains
        #######################################################################

        logging.info("Locating Bifurcations")
        t0 = time.time()

        # get mask where potential saddle points connecting domains exist
        bif_mask = find_potential_saddle_points(
            data=elf_grid.total,
            edge_mask=elf_bader.basin_edges,
            greater=True,
            vacuum_mask=elf_grid.vacuum_mask,
        )

        # get the basins connected at these points
        lower_points, upper_points, connection_values = find_domain_connections(
            basin_labels=elf_bader.basin_labels,
            data=elf_grid.total,
            bif_mask=bif_mask,
            edge_mask=get_edges(
                elf_bader.basin_labels,
                neighbor_transforms,
                elf_bader.vacuum_mask,
            ),
            num_basins=len(elf_bader.basin_maxima_frac),
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=elf_bader.vacuum_mask,
        )

        # clear mask for memory
        bif_mask = None

        # add maxima values as the points each basin "connects" to itself
        basin_maxima = elf_bader.basin_maxima_ref_values
        basin_indices = np.arange(len(basin_maxima))
        lower_points = np.append(lower_points, basin_indices)
        upper_points = np.append(upper_points, basin_indices)
        connection_values = np.append(connection_values, basin_maxima)

        # group and get unique
        connection_array = np.column_stack(
            (lower_points, upper_points, connection_values)
        )
        unique_connections, unique_indices = np.unique(
            connection_array, return_index=True, axis=0
        )

        # get pairs of connections
        lower_points = lower_points[unique_indices]
        upper_points = upper_points[unique_indices]
        connection_pairs = np.column_stack((lower_points, upper_points))

        # get values of connections
        connection_values = connection_values[unique_indices]

        basin_maxima_grid = np.round(
            elf_grid.frac_to_grid(elf_bader.basin_maxima_frac)
        ).astype(np.int64)
        basin_maxima_grid %= elf_grid.shape

        basin_maxima_ref_values = elf_bader.basin_maxima_ref_values

        (
            domain_basins,
            domain_min_values,
            domain_max_values,
            domain_dims,
            domain_parents,
        ) = find_domain_bifurcations(
            connection_pairs,
            connection_values,
            basin_maxima_grid,
            basin_maxima_ref_values,
            elf_grid.total,
            neighbor_transforms,
            vacuum_mask=elf_bader.vacuum_mask,
        )
        # convert basins to numpy arrays to avoid Numba reflected list issue
        domain_basins = [np.array(i, dtype=np.int64) for i in domain_basins]

        t1 = time.time()
        logging.info(f"Time: {round(t1-t0, 2)}")

        # NOTE: It seems like my maxima merging is still not perfect. In the
        # neargrid method this means that in some cases a point with a higher
        # value than the basin's maximum can be assigned to it. For now I'm removing
        # this check unless major issues occur.
        # run a quick check ensuring that all basins appear as individual irreducible
        # nodes
        # all_basins = np.zeros(len(basin_maxima_grid), dtype=np.bool_)
        # for basins in domain_basins:
        #     if len(basins) != 1:
        #         continue
        #     all_basins[basins[0]] = True

        # assert np.all(
        #     all_basins
        # ), """Not all basins were assigned to irreducible domains. This is a bug!!! Please report to our github:
        #     https://github.com/SWeav02/baderkit"""

        #######################################################################
        # Get Atoms Surrounded by Each domain
        #######################################################################

        logging.info("Finding contained atoms")

        # possible saddle points where voids between domains first connect
        bif_mask = find_potential_saddle_points(
            data=elf_grid.total,
            edge_mask=elf_bader.basin_edges,
            greater=False,
            vacuum_mask=elf_bader.vacuum_mask,
        )

        # get the possible values and clear mask
        bif_values = elf_grid.total[bif_mask]
        bif_mask = None

        # add the values from the domain bifurcations and get only the
        # unique options
        bif_values = np.unique(np.append(bif_values, domain_min_values))

        # get atom grid coordinates
        atom_grid_coords = elf_grid.frac_to_grid(elf_bader.structure.frac_coords)
        atom_grid_coords = (
            np.round(atom_grid_coords).astype(np.int64) % elf_grid.shape
        )

        # get the atoms each domain contains
        (
            domain_basins,
            domain_min_values,
            domain_max_values,
            domain_dims,
            domain_parents,
            domain_atoms,
        ) = get_domains_surrounding_atoms(
            possible_values=bif_values,
            domain_basins=domain_basins,
            domain_min_values=domain_min_values,
            domain_max_values=domain_max_values,
            domain_dims=domain_dims,
            domain_parents=domain_parents,
            atom_grid_coords=atom_grid_coords,
            neighbor_transforms=neighbor_transforms,
            basin_labels=elf_bader.basin_labels,
            data=elf_grid.total,
            num_basins=len(elf_bader.basin_maxima_frac),
            vacuum_mask=elf_bader.vacuum_mask,
        )
        t2 = time.time()
        logging.info(f"Time: {round(t2-t1, 2)}")

        #######################################################################
        # Construct Graph
        #######################################################################
        graph = cls(
            structure=elf_bader.structure,
            basin_maxima_frac=elf_bader.basin_maxima_frac,
            basin_charges=elf_bader.basin_charges,
            basin_volumes=elf_bader.basin_volumes,
            **kwargs
        )
        node_keys = []
        for feat_idx in range(len(domain_basins)):
            if domain_parents[feat_idx] == -1:
                parent = None
            else:
                parent_key = node_keys[domain_parents[feat_idx]]
                parent = graph.node_from_key(parent_key)
            if feat_idx in domain_parents:
                # This is a reducible domain as it has children
                node = ReducibleNode(
                    bifurcation_graph=graph,
                    basins=domain_basins[feat_idx],
                    dimensionality=domain_dims[feat_idx],
                    contained_atoms=domain_atoms[feat_idx],
                    min_value=domain_min_values[feat_idx],
                    max_value=domain_max_values[feat_idx],
                    parent=parent,
                    domain_subtype=DomainSubtype.reducible,
                )

            else:
                # this is an irreducible domain
                node = IrreducibleNode(
                    bifurcation_graph=graph,
                    basins=domain_basins[feat_idx],
                    dimensionality=domain_dims[feat_idx],
                    contained_atoms=domain_atoms[feat_idx],
                    min_value=domain_min_values[feat_idx],
                    max_value=domain_max_values[feat_idx],
                    parent=parent,
                    domain_subtype=DomainSubtype.irreducible_point,
                )

            node_keys.append(node.key)

        # give each reducible node a subtype
        for node in graph.reducible_nodes:
            parent = node.parent
            if parent is None:
                node.domain_subtype = DomainSubtype.root
                continue
            # if the number of basins changes, this is a standard reducible domain
            elif len(parent.basins) != len(node.basins):
                node.domain_subtype = DomainSubtype.reducible_dom
            # check for dimension change
            elif parent.dimensionality != node.dimensionality:
                node.domain_subtype = DomainSubtype.reducible_dim
            # finally check for atom change
            elif len(parent.contained_atoms) != len(node.contained_atoms):
                node.domain_subtype = DomainSubtype.reducible_atom

        # It is common for there to be quite a few shallow reducible domains
        # seemingly due to voxelation. We remove those below a relative cutoff.
        cls._remove_shallow_reducible_nodes(graph, shallow_reducible_cutoff)

        # Now we check for reducible nodes that should really be considered
        # irreducible. These nodes are very deep but their children separate
        # at very low values
        cls._combine_shallow_irreducible_nodes(
            graph, shallow_irreducible_cutoff
        )
        
        # Next we calculate the overlap of each basin with the atomic regions
        # of the charge denisty

        return graph

    @staticmethod
    def _reduce_shallow_nodes(graph, remove_cutoff=0.05, reduce_cutoff=0.10):
        """
        Removes reducible nodes that are significantly more shallow than their
        parent nodes.

        Parameters
        ----------
        graph : BifurcationGraph
            The graph to remove shallow nodes from.
        cutoff : float, optional
            The cutoff ratio for a node to be considered shallow. The default is 0.05.

        """
        # First we remove shallow reducible nodes
        # iterate over nodes from low to high
        reducible_nodes = graph.sorted_reducible_nodes
        for node in reducible_nodes[1:]:
            parent = node.parent
            # check that this node is very shallow relative to its parent
            if (node.depth / parent.depth) > remove_cutoff:
                continue
            # This is a very shallow node. delete it
            node.remove()
            
        # Next we reduce very deep reducible nodes with very shallow children
        # to single irreducible nodes
        # TODO: Add check that nodes are at relatively similar values
        # iterate from highest to lowest
        reducible_nodes = graph.sorted_reducible_nodes.copy()
        reducible_nodes.reverse()
        for node in reducible_nodes:
            # skip infinite nodes and nodes we've already checked
            if node.is_infinite:  # or node.key in checked_nodes:
                continue
            # get this nodes depth
            depth = node.depth
            is_shallow = True
            # get all children
            for child in node.deep_children:
                # skip other reducible nodes
                if child.is_reducible:
                    continue
                # check if depth is more than the cutoff portion of the parent's depth. If so,
                # we don't consider this domain to be shallow
                if (child.depth / depth) > reduce_cutoff:
                    is_shallow = False
                    break

            if not is_shallow:
                continue

            # combine node
            node.make_irreducible()

###############################################################################
# Plotting Methods
###############################################################################

    def get_plot(self) -> go.Figure:
        """

        Returns
        -------
        fig : go.Figure
            Returns a plotly graph object representing the BifurcationGraph.

        """

        #######################################################################
        # Y Values
        #######################################################################

        indices = [i.key for i in self]

        def assign_y_positions(node, y_counter, y_positions):
            # This function iteratively loops starting from the root node and
            # places each parent node at the average position of its children.
            # children are placed when found. The iterative nature results in
            # connecting lines not overlapping.
            if not node.is_reducible:  # it's a leaf
                y_positions[node.key] = next(y_counter)
            else:  # its a branch
                children = node.children
                for child in children:
                    assign_y_positions(child, y_counter, y_positions)
                child_ys = [y_positions[child.key] for child in children]
                y_positions[node.key] = np.mean(child_ys)

        # Create a mapping from node ID to Y position
        y_positions = {}
        y_counter = itertools.count(0)  # This gives 0, 1, 2, ... for leaf placement

        # BUGFIX: We may have multiple roots (e.g. molecules separated by vacuum)
        # so we find the y values separately then adjust
        for root_node in self.root_nodes:
            assign_y_positions(root_node, y_counter, y_positions)

        # Then set Yn using our dict
        Yn = [y_positions[i] for i in indices]

        # Normalize Y scale
        max_y = 2
        Yn = np.array(Yn, dtype=float)
        Yn -= Yn.min()
        if Yn.max() > 0:
            Yn /= Yn.max()
            Yn *= max_y
        # Get the height of each irreducible node
        y_division = max_y / len(self.irreducible_nodes)

        #######################################################################
        # Lines
        #######################################################################
        # Now we need to get the lines that will be used for each edge. These will use
        # a nested lists where each edge has one entry and the sub-lists contain the
        # two x and y entries for each edge.
        Xn = [round(i.min_value, 4) for i in self]
        Xn1 = [round(i.max_value, 4) for i in self]
        Xe = []
        Ye = []
        for node in self.reducible_nodes:
            parent = node.key
            children = node.children
            for child_node in children:
                child = child_node.key
                px = Xn[indices.index(parent)]
                py = Yn[indices.index(parent)]
                cx = Xn[indices.index(child)]
                cy = Yn[indices.index(child)]

                # Vertical segment: (px, py) -> (px, cy)
                Xe.extend([px, px, None])
                Ye.extend([py, cy, None])

                # Horizontal segment: (px, cy) -> (cx, cy)
                Xe.extend([px, cx, None])
                Ye.extend([cy, cy, None])

        # create the figure and add the lines
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=Xe,
                y=Ye,
                mode="lines",
                name="connection",
                line=dict(color=LINE_COLOR, width=3),
                hoverinfo="none",
            )
        )

        #######################################################################
        # Nodes
        #######################################################################

        # tracker for legend
        already_added_types = set()
        # y positions for boxes
        Yn = np.array(Yn)
        Yn0 = Yn - y_division / 3
        Yn1 = Yn + y_division / 3
        for idx, node in enumerate(self):
            if node.is_reducible:
                showlegend = node.domain_subtype not in already_added_types
                already_added_types.add(node.domain_subtype)
                # add a circle
                fig.add_trace(
                    go.Scatter(
                        x=[Xn[idx]],
                        y=[Yn[idx]],
                        mode="markers",
                        name=f"{node.domain_subtype.value}",
                        marker=dict(
                            symbol="circle-dot",
                            size=18,
                            color=node.domain_subtype.plot_color,
                            line=dict(color="grey", width=1),
                        ),
                        text=node.plot_label,
                        hoverinfo="text",
                        showlegend=showlegend,
                    )
                )
            else:
                showlegend = node.feature_type not in already_added_types
                already_added_types.add(node.feature_type)
                # add a rectangle
                x0 = Xn[idx]
                x1 = Xn1[idx]
                # make sure x1 is at least a reasonable minimum
                x1 = max(x1, x0 + 0.01)
                y0 = Yn0[idx]
                y1 = Yn1[idx]
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, x1, x0, x0],
                        y=[y0, y0, y1, y1, y0],
                        fill="toself",
                        fillcolor=node.feature_type.plot_color,
                        line=dict(color=LINE_COLOR),
                        hoverinfo="text",
                        text=node.plot_label,
                        name=f"{node.feature_type.value}",
                        mode="lines",
                        opacity=0.8,
                        showlegend=showlegend,
                    )
                )

        #######################################################################
        # Layout
        #######################################################################

        min_x = min(Xn)
        max_x = max(Xn1)
        x_range = max_x - min_x
        buffer = x_range * 0.05
        # remove y axis label and add title
        fig.update_layout(
            title=dict(
                text=f"{self.structure.reduced_formula} Bifurcations",
                xanchor="center",
                x=0.5,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(
                range=[min_x - buffer, max_x + buffer],
                title="ELF",
            ),
            yaxis=dict(
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
            ),
        )
        return fig
