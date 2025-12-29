# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, Type, TypeVar

import numpy as np
from numpy.typing import NDArray

from baderkit.core.utilities.basic import merge_frac_coords
from baderkit.core.utilities.coord_env import get_dists_to_atoms

from .enum_and_styling import DomainSubtype, FeatureType
from .graph_numba.bifurcation_base import get_pt_ring_cage

Node = TypeVar("Node", bound="NodeBase")


class NodeBase(ABC):

    _registry: Dict[str, Type["NodeBase"]] = {}
    is_reducible = False

    label_map = {
        "domain subtype": "domain_subtype",
        "basins": "basins",
        "dimensionality": "dimensionality",
        "contained atoms": "contained_atoms",
        "min value": "min_value",
        "max value": "max_value",
        "depth": "depth",
    }

    def __init__(
        self,
        bifurcation_graph,
        basins: NDArray[int],
        dimensionality: int,
        contained_atoms: list[int],
        min_value: float,
        max_value: float,
        key: int | None = None,
        parent: Node | int | None = None,
        domain_subtype: str = None,
    ):

        # set properties that all nodes have
        self.bifurcation_graph = bifurcation_graph
        self.basins = np.array(basins)
        self.dimensionality = int(dimensionality)
        self.contained_atoms = np.array(contained_atoms)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.domain_subtype = domain_subtype

        # convert integer parents to the corresponding Node object
        if type(parent) == int:
            parent = bifurcation_graph.node_from_key(parent)

        # check if our parent is None. If so, we add a new root node
        if parent is None:
            bifurcation_graph._root_nodes.append(self)
        else:
            parent._children.append(self)

        # set parent
        self._parent = parent

        # create a key for this node
        if key is None:
            if len(bifurcation_graph) > 0:
                key = max(bifurcation_graph._node_keys.keys()) + 1
            else:
                key = 0
        self.key = key

        # add this node to the corresponding graph
        bifurcation_graph._nodes.append(self)
        bifurcation_graph._node_keys[key] = self

    def __init_subclass__(cls, **kwargs):
        # automatically registers subclasses. Used for convenient from_dict method
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls  # Register subclass automatically
        # automatically add class name shortcut
        cls.domain_type = cls.__name__
        cls.__annotations__["domain_type"] = str

    def __eq__(
        self,
        other: Node,
    ):

        assert (
            self.bifurcation_graph is other.bifurcation_graph
        ), "Nodes must belong to the same BifurcationGraph to make equality check"

        # node indices are unique. Return the comparison between the two
        return self.key == other.key

    @property
    def domain_subtype(self) -> DomainSubtype | None:
        return self._domain_subtype

    @domain_subtype.setter
    def domain_subtype(self, value: DomainSubtype | None):
        if value is None:
            self._domain_subtype = None
        else:
            self._domain_subtype = DomainSubtype(value)

    ###########################################################################
    # Graph Related Properties
    ###########################################################################

    @property
    def parent(self) -> Node:
        return self._parent

    @parent.setter
    def parent(self, new_parent: int | Node):
        # make sure this isn't the parent node
        assert self._parent is not None, "Root node cannot be assigned a new parent"
        if type(new_parent) == int:
            new_parent = self.bifurcation_graph.get(new_parent, None)
        # remove this node from the current parent's children
        self._parent._children = [i for i in self._parent._children if i is not self]
        # update this node's parent
        self._parent = new_parent
        # add this node to the new parent's children
        new_parent._children.append(self)

    @property
    def siblings(self) -> list[Node]:
        if self.parent is not None:
            return [i for i in self.parent.children if i is not self]
        else:
            return []

    @property
    def ancestors(self) -> list[Node]:
        """

        Returns
        -------
        list[Node]
            The parent, grandparent, great grandparent, etc. of this node.

        """
        all_parents = []
        current_parent = self.parent
        while current_parent is not None:
            all_parents.append(current_parent)
            current_parent = current_parent.parent
        return all_parents

    ###########################################################################
    # Feature Related Properties
    ###########################################################################

    @cached_property
    def depth(self):
        return self.max_value - self.min_value

    @property
    def depth_to_infinite(self):
        if self.is_infinite:
            return 0.0  # this node itself is infinite

        # loop from most recent ancestor to oldest and find the first that is infinite
        for ancestor in self.ancestors:
            if ancestor.is_infinite:
                break
        return self.max_value - ancestor.max_value

    @cached_property
    def is_infinite(self):
        return self.dimensionality > 0

    ###########################################################################
    # Properties that must be set by children or require additional steps
    ###########################################################################
    @property
    def plot_label(self) -> None:
        lines = []
        for tag, attr in self.label_map.items():
            value = getattr(self, attr, None)
            if value is None:
                continue
            if "type" in tag:
                lines.append(f"{tag}: {value.value}".title())
                continue
            if isinstance(value, (float, np.floating)):
                value = round(value, 4)
            lines.append(f"{tag}: {value}".title())

        label = "<br>".join(lines)
        return label

    @abstractmethod
    def remove(self) -> None:
        pass

    def to_dict(self) -> dict:

        if self.parent is None:
            parent_key = None
        else:
            parent_key = self.parent.key

        return {
            "key": self.key,
            "basins": [
                int(i) for i in self.basins
            ],  # convert to python for json serialize
            "dimensionality": int(self.dimensionality),
            "contained_atoms": [int(i) for i in self.contained_atoms],
            "min_value": float(self.min_value),
            "max_value": float(self.max_value),
            "parent": parent_key,
            "domain_type": self.domain_type,
            "domain_subtype": self.domain_subtype,
        }

    @classmethod
    def from_dict(cls, bifurcation_graph, node_dict: dict) -> Node:
        # automatic node creation for all inheriting Nodes
        node_type = node_dict.pop("domain_type")
        subclass = cls._registry[node_type]
        return subclass(bifurcation_graph=bifurcation_graph, **node_dict)


class ReducibleNode(NodeBase):

    is_reducible = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._children = []

    @property
    def children(self):
        return self._children

    @property
    def deep_children(self):
        all_children = []
        current_children = self.children
        while True:
            new_children = []
            for child in current_children:
                all_children.append(child)
                if not child.is_reducible:
                    continue
                new_children.extend(child.children)

            current_children = new_children
            if len(current_children) == 0:
                break
        return all_children

    def remove(self) -> None:
        # if this is a root, each child will be a new root
        if self.parent is None:
            # remove this node as a root
            self.bifurcation_graph._root_nodes = [
                i for i in self.bifurcation_graph._root_nodes if i is not self
            ]
            # add each child as a root
            for child in self.children:
                self.bifurcation_graph._root_nodes.append(child)
                child._parent = None
        else:
            # assign all children to parent
            for child in self.children:
                child.parent = self.parent
            # remove this node from the current parent's children
            self.parent._children = [i for i in self.parent._children if i is not self]
            # recalculate parent's max value
            min_val = self.max_value
            for child in self.parent.children:
                if child.min_value < min_val:
                    min_val = child.min_value
            self.parent.max_value = min_val
        # delete this node
        graph = self.bifurcation_graph
        graph._nodes = [i for i in graph._nodes if i is not self]
        del graph._node_keys[self.key]

    def make_irreducible(self):
        # find the maximum value in children
        max_value = -1e300
        for child in self.deep_children:
            if child.is_reducible:
                continue
            if child.max_value > max_value:
                max_value = child.max_value
        # delete all nodes below this one
        nodes = self.deep_children.copy()
        nodes.reverse()
        for node in nodes:
            node.remove()
        # delete self
        self.remove()
        # check if the basins in this node form a point, ring, or cage
        frac_coords = self.bifurcation_graph.basin_maxima_frac[self.basins]
        irreducible_type = get_pt_ring_cage(frac_coords)
        if irreducible_type == 0:
            domain_subtype = DomainSubtype.irreducible_point
        elif irreducible_type == 1:
            domain_subtype = DomainSubtype.irreducible_ring
        else:
            domain_subtype = DomainSubtype.irreducible_cage

        # create a new irreducible node connected to parent
        node = IrreducibleNode(
            key=self.key,
            bifurcation_graph=self.bifurcation_graph,
            basins=self.basins,
            dimensionality=self.dimensionality,
            contained_atoms=self.contained_atoms,
            min_value=self.min_value,
            max_value=max_value,
            parent=self.parent,
            domain_subtype=domain_subtype,
        )

        return node


class IrreducibleNode(NodeBase):

    is_reducible = False

    label_map = NodeBase.label_map
    label_map.update(
        {
            "feature type": "feature_type",
            "charge": "charge",
            "volume": "volume",
            "depth to infinite feature": "depth_to_infinite",
            "atom distance": "atom_distance",
            "nearest atom index": "nearest_atom",
            "nearest atom species": "nearest_atom_species",
            "minimum surface dist": "min_surface_dist",
            "average surface dist": "avg_surface_dist",
            "distance beyond atom": "dist_beyond_atom",
            "coord number": "coord_num",
            "coord atom indices": "coord_atom_indices",
            "coord atom species": "coord_atom_species",
        }
    )

    def __init__(
        self,
        feature_type: FeatureType | str = FeatureType.unknown,
        nearest_atom_sphere: int = None,
        dist_beyond_atom: float = None,
        min_surface_dist: float = None,
        avg_surface_dist: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # these usually need to be labeled, but this allows them to be recreated
        # from json
        self.feature_type = feature_type
        self._dist_beyond_atom = dist_beyond_atom
        self._nearest_atom_sphere = nearest_atom_sphere
        self._min_surface_dist = min_surface_dist
        self._avg_surface_dist = avg_surface_dist

        self._coord_atom_indices = None
        self._coord_electride_indices = None

    @property
    def feature_type(self) -> FeatureType | None:
        return self._feature_type

    @feature_type.setter
    def feature_type(self, value: FeatureType | None):
        if value is None:
            self._feature_type = None
        else:
            self._feature_type = FeatureType(value)

    @cached_property
    def charge(self) -> float:
        return self.bifurcation_graph.basin_charges[self.basins].sum()

    @cached_property
    def volume(self) -> float:
        return self.bifurcation_graph.basin_volumes[self.basins].sum()

    @cached_property
    def frac_coords(self) -> NDArray:
        return self.bifurcation_graph.basin_maxima_frac[self.basins]

    @cached_property
    def average_frac_coords(self) -> NDArray:
        return merge_frac_coords(self.frac_coords)

    @cached_property
    def atom_dists(self) -> NDArray[int]:
        return get_dists_to_atoms(
            self.average_frac_coords,
            self.bifurcation_graph.structure.frac_coords,
            self.bifurcation_graph.structure.cart_coords,
            self.bifurcation_graph.structure.lattice.matrix,
        )

    @cached_property
    def electride_dists(self) -> NDArray[int]:
        return get_dists_to_atoms(
            self.average_frac_coords,
            self.bifurcation_graph.electride_structure.frac_coords,
            self.bifurcation_graph.electride_structure.cart_coords,
            self.bifurcation_graph.electride_structure.lattice.matrix,
        )

    @cached_property
    def atom_distance(self) -> float:
        return np.min(self.atom_dists)

    @cached_property
    def nearest_atom(self) -> int:
        return np.argmin(self.atom_dists)

    @cached_property
    def nearest_atom_species(self) -> str:
        return self.bifurcation_graph.structure[self.nearest_atom].species_string

    @property
    def nearest_atom_sphere(self) -> int:
        # the nearest atom if you measure the center of this feature to the
        # atoms surface
        return self._nearest_atom_sphere

    @property
    def dist_beyond_atom(self) -> float:
        return self._dist_beyond_atom

    @property
    def min_surface_dist(self) -> float:
        return self._min_surface_dist

    @property
    def avg_surface_dist(self) -> float:
        return self._avg_surface_dist

    @property
    def coord_number(self) -> int:
        return len(self.coord_atom_indices)

    @property
    def coord_atom_indices(self) -> list[int]:
        if self._coord_atom_indices is None:
            if self.feature_type in FeatureType.atomic_types:
                self._coord_atom_indices = [int(self.nearest_atom)]
            else:
                # TODO: I would really like a better method of doing this as
                # CrystalNN is very slow in this situation. I can't do them all
                # at once because I don't want features seeing each other as neighbors
                feature_structure = self.bifurcation_graph.structure.copy()
                feature_structure.append("H-", self.average_frac_coords)
                coordination = self.bifurcation_graph.cnn.get_nn_info(
                    feature_structure, -1
                )
                self._coord_atom_indices = [int(i["site_index"]) for i in coordination]

        return self._coord_atom_indices

    @property
    def coord_atom_species(self) -> list[str]:
        structure = self.bifurcation_graph.structure
        return [structure[i].species_string for i in self.coord_atom_indices]

    @property
    def coord_atom_dists(self) -> list[float]:
        return self.atom_dists[self.coord_atom_indices]

    @property
    def electride_coord_number(self) -> int:
        return len(self.coord_electride_indices)

    @property
    def coord_electride_indices(self) -> list[int]:
        if self._coord_electride_indices is None:
            if self.feature_type in FeatureType.atomic_types:
                self._coord_electride_indices = [int(self.nearest_atom)]
            else:
                feature_structure = (
                    self.bifurcation_graph._electride_hatom_structure.copy()
                )
                if self.feature_type in FeatureType.bare_types:
                    # this feature is already in our structure. just find the index
                    try:
                        feat_idx = self.bifurcation_graph.get_feature_nodes(
                            FeatureType.bare_types
                        ).index(self)
                    except:
                        breakpoint()
                    struc_idx = len(self.bifurcation_graph.structure) + feat_idx
                else:
                    # temporarily add this feature to our structure
                    feature_structure.append("H-", self.average_frac_coords)
                    struc_idx = -1
                coordination = self.bifurcation_graph.cnn.get_nn_info(
                    feature_structure, struc_idx
                )
                self._coord_electride_indices = [
                    int(i["site_index"]) for i in coordination
                ]

        return self._coord_electride_indices

    @property
    def coord_electride_species(self) -> list[str]:
        structure = self.bifurcation_graph.electride_structure
        return [structure[i].species_string for i in self.coord_electride_indices]

    @property
    def coord_electride_dists(self) -> list[float]:
        return self.electride_dists[self.coord_electride_indices]

    def remove(self) -> None:
        # remove this node from the current parent's children
        self.parent._children = [i for i in self.parent._children if i is not self]
        # delete this node
        graph = self.bifurcation_graph
        graph._nodes = [i for i in graph._nodes if i is not self]
        del graph._node_keys[self.key]

    def to_dict(self) -> dict:
        node_dict = super().to_dict()
        # get values that can't be calculated directly
        for attr in ["min_surface_dist", "avg_surface_dist"]:
            node_dict[attr] = getattr(self, attr, None)
        return node_dict
