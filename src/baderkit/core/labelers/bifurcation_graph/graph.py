# -*- coding: utf-8 -*-

import json
import logging
import time
import itertools

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from pymatgen.analysis.local_env import CrystalNN

from baderkit.core import Bader, Structure
from baderkit.core.bader.methods.shared_numba import get_edges

from .nodes import IrreducibleNode, ReducibleNode, NodeBase
from .enum_and_styling import LINE_COLOR, DomainSubtype, FeatureType
# from elf_analyzer.core.utilities import IonicRadiiTools
from .graph_numba import (
    find_domain_bifurcations,
    find_domain_connections,
    find_potential_saddle_points,
    get_domains_surrounding_atoms,
    )


class BifurcationGraph:
    """
    A class for storing the nodes of a bifurcation graph. The nodes themselves
    contain the information on their connectivity.
    """
    
    def __init__(
            self, 
            structure: Structure,
            labeler_type: str,
            basin_maxima_frac: NDArray[float],
            basin_charges: NDArray[float],
            basin_volumes: NDArray[float],
            crystalnn_kwargs: dict,
            atomic_radii: NDArray[float] = None,
            ):
        self._root_nodes = []
        self._nodes = []
        self._node_keys = {}
        
        self.structure = structure
        self.labeler_type = labeler_type
        self.basin_maxima_frac = basin_maxima_frac
        self.basin_charges = basin_charges
        self.basin_volumes = basin_volumes
        self.crystalnn_kwargs = crystalnn_kwargs
        self.cnn = CrystalNN(**crystalnn_kwargs)

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
    def root_nodes(self):
        return self._root_nodes
    
    @property
    def nodes(self):
        return self._nodes
        
    def node_from_key(self, key):
        return self._node_keys[key]
    
    @property
    def irreducible_nodes(self):
        return [i for i in self if not i.is_reducible]
    
    @property
    def reducible_nodes(self):
        return [i for i in self if i.is_reducible]
    
    @property
    def sorted_reducible_nodes(self):
        nodes = self.reducible_nodes
        min_vals = [i.min_value for i in nodes]
        sorted_indices = np.argsort(min_vals)
        
        return [nodes[i] for i in sorted_indices]
    
    @property
    def unassigned_nodes(self):
        return [i for i in self if not i.is_reducible and i.feature_type in (None, FeatureType.unknown)]
    
    @property
    def quasi_atom_structure(self):
        structure = self.structure.copy()
        for node in self.get_feature_nodes(feature_types=FeatureType.bare_types):
            structure.append(node.feature_type.dummy_species, node.average_frac_coords)
        return structure
    
    @property
    def _quasi_hatom_structure(self):
        # same as above, but with H- dummy atoms for approximate radii in crystalNN
        structure = self.structure.copy()
        for node in self.get_feature_nodes(feature_types=FeatureType.bare_types):
            structure.append("H-", node.average_frac_coords)
        return structure
    
    @property
    def labeled_structure(self):
        structure = self.structure.copy()
        for node in self.irreducible_nodes:
            structure.append(node.feature_type.dummy_species, node.average_frac_coords)
        return structure
    
    
    def get_feature_nodes(self, feature_types: list[str]):
        return [i for i in self if not i.is_reducible and i.feature_type in feature_types]
    
    def to_dict(self) -> dict:
        graph_dict =  {
            "nodes": [i.to_dict() for i in self],
            "structure": self.structure.to_json(),
            "labeler_type": self.labeler_type,
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
    
    @classmethod
    def from_dict(cls, graph_dict: dict):
        nodes = graph_dict.pop("nodes")
        graph_dict["structure"] = Structure.from_str(graph_dict["structure"], fmt="json")
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

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str):
        graph_dict = json.loads(json_str)
        return cls.from_dict(graph_dict)
    
    @classmethod
    def from_labeler(cls, labeler: Bader):
        reference_grid = labeler.reference_grid
        neighbor_transforms, _ = reference_grid.point_neighbor_transforms
        
        
        #######################################################################
        # Get Bifurcation Values and Corresponding domains
        #######################################################################
        
        logging.info("Locating Bifurcations")
        t0 = time.time()
        
        # get mask where potential saddle points connecting domains exist
        bif_mask = find_potential_saddle_points(
            data=reference_grid.total,
            edge_mask=labeler.bader.basin_edges,
            greater=True
            )
        
        # get the basins connected at these points
        lower_points, upper_points, connection_values = find_domain_connections(
            basin_labels=labeler.bader.basin_labels,
            data=reference_grid.total,
            bif_mask=bif_mask,
            edge_mask=get_edges(labeler.bader.basin_labels,neighbor_transforms,labeler.bader.vacuum_mask),
            num_basins=len(labeler.bader.basin_maxima_frac),
            neighbor_transforms=neighbor_transforms,
            )

        # clear mask for memory
        bif_mask = None

        # add maxima values as the points each basin "connects" to itself
        basin_maxima = labeler.bader.basin_maxima_ref_values
        basin_indices = np.arange(len(basin_maxima))
        lower_points = np.append(lower_points, basin_indices)
        upper_points = np.append(upper_points, basin_indices)
        connection_values = np.append(connection_values, basin_maxima)
        
        # group and get unique
        connection_array = np.column_stack((lower_points, upper_points, connection_values))
        unique_connections, unique_indices = np.unique(connection_array, return_index=True, axis=0)
        
        # get pairs of connections
        lower_points = lower_points[unique_indices]
        upper_points = upper_points[unique_indices]
        connection_pairs=np.column_stack((lower_points, upper_points))
        
        # get values of connections
        connection_values = connection_values[unique_indices]
        
        basin_maxima_grid = np.round(labeler.reference_grid.frac_to_grid(labeler.bader.basin_maxima_frac)).astype(np.int64)
        basin_maxima_grid %= labeler.reference_grid.shape
        
        basin_maxima_ref_values=labeler.bader.basin_maxima_ref_values
        
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
            reference_grid.total,
            neighbor_transforms,
                )
        # convert basins to numpy arrays to avoid Numba reflected list issue
        domain_basins = [np.array(i, dtype=np.int64) for i in domain_basins]
        
        t1 = time.time()
        logging.info(f"Time: {round(t1-t0, 2)}")
        # breakpoint()
        # run a quick check ensuring that all basins appear as individual irreducible
        # nodes
        all_basins = np.zeros(len(basin_maxima_grid), dtype=np.bool_)
        for basins in domain_basins:
            if len(basins) != 1:
                continue
            all_basins[basins[0]] = True
        
        if not np.all(all_basins):
            breakpoint()
            
        assert np.all(all_basins), """Not all basins were assigned to irreducible domains. This is a bug!!! Please report to our github:
            https://github.com/SWeav02/baderkit"""

        #######################################################################
        # Get Atoms Surrounded by Each domain
        #######################################################################

        logging.info("Finding contained atoms")    
        
        # possible saddle points where voids between domains first connect
        bif_mask = find_potential_saddle_points(
            data=reference_grid.total,
            edge_mask=labeler.bader.basin_edges,
            greater=False
            )

        # get the possible values and clear mask
        bif_values = reference_grid.total[bif_mask]
        bif_mask = None

        # add the values from the domain bifurcations and get only the
        # unique options
        bif_values = np.unique(np.append(bif_values, domain_min_values))
        
        # get atom grid coordinates
        atom_grid_coords = reference_grid.frac_to_grid(labeler.structure.frac_coords)
        atom_grid_coords = np.round(atom_grid_coords).astype(np.int64) % reference_grid.shape
        # breakpoint()
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
                basin_labels=labeler.bader.basin_labels,
                data=reference_grid.total,
                num_basins=len(labeler.bader.basin_maxima_frac),
                )
        t2 = time.time()
        logging.info(f"Time: {round(t2-t1, 2)}")

        #######################################################################
        # Construct Graph
        #######################################################################    
        graph = cls(
            structure=labeler.structure,
            labeler_type=labeler.__class__.__name__,
            basin_maxima_frac=labeler.bader.basin_maxima_frac,
            basin_charges=labeler.bader.basin_charges,
            basin_volumes=labeler.bader.basin_volumes,
            crystalnn_kwargs=labeler.crystalnn_kwargs,
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
                    domain_subtype=DomainSubtype.reducible
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
                    domain_subtype=DomainSubtype.irreducible_point
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
        
        # TODO: It is common for there to be quite a few shallow reducible domains
        # seemingly due to voxelation. I need a better method for removing these
        cls._remove_shallow_reducible_nodes(graph)
            
        
        # Now we check for reducible nodes that should really be considered
        # irreducible. These nodes are very deep but their children separate
        # at very low values
        cls._combine_shallow_irreducible_nodes(graph)
        
        return graph
    
    @staticmethod
    def _remove_shallow_reducible_nodes(graph, cutoff=0.005):
        # iterate over nodes from low to high
        reducible_nodes = graph.sorted_reducible_nodes
        for node in reducible_nodes[1:]:
            parent = node.parent
            # check that this node is very shallow relative to its parent
            if (node.depth / parent.depth) > cutoff:
                continue
            # This is a very shallow node. delete it
            print("deleting")
            node.remove()
    
    @staticmethod
    def _combine_shallow_irreducible_nodes(graph, cutoff=0.1):
        # TODO: Add check that nodes are at relatively similar values
        # iterate from highest to lowest
        reducible_nodes = graph.sorted_reducible_nodes.copy()
        reducible_nodes.reverse()
        for node in reducible_nodes:
            # skip infinite nodes and nodes we've already checked
            if node.is_infinite:# or node.key in checked_nodes:
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
                if (child.depth / depth) > cutoff:
                    is_shallow = False
                    break
            
            if not is_shallow:
                continue
            
            # combine node
            node.make_irreducible()
            
        
    def get_plot(self) -> go.Figure:
        """
        Returns a plotly figure
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
            if not node.is_reducible: # it's a leaf
                y_positions[node.key] = next(y_counter)
            else: # its a branch
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
        Xn = [round(i.min_value,4) for i in self]
        Xn1 = [round(i.max_value,4) for i in self]
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
                x1 = max(x1, x0+0.01)
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
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(range=[min_x-buffer, max_x+buffer], title=f"{self.labeler_type} Bifurcations"),
            yaxis=dict(
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
            ),
        )
        return fig
        