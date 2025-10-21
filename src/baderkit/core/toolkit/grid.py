# -*- coding: utf-8 -*-

import itertools
import logging
import math
import time
from copy import deepcopy
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp import VolumetricData
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.ndimage import label, spline_filter, zoom
from scipy.spatial import Voronoi

from baderkit.core.toolkit.file_parsers import (
    Format,
    detect_format,
    infer_significant_figures,
    read_cube,
    read_vasp,
)
from baderkit.core.toolkit.file_parsers import write_cube as write_cube_file
from baderkit.core.toolkit.file_parsers import write_vasp as write_vasp_file
from baderkit.core.toolkit.grid_numba import Interpolator
from baderkit.core.toolkit.structure import Structure

# This allows for Self typing and is compatible with python versions before 3.11
Self = TypeVar("Self", bound="Grid")


class DataType(str, Enum):
    charge = "charge"
    elf = "elf"

    @property
    def prefix(self):
        return {
            DataType.charge: "CHGCAR",
            DataType.elf: "ELFCAR",
        }[self]


class Grid(VolumetricData):
    """
    A representation of the charge density, ELF, or other volumetric data.
    This class is a wraparound for Pymatgen's VolumetricData class with additional
    properties and methods.

    See Also
    --------
    :class:`~pymatgen.io.vasp.outputs.VolumetricData`
        The parent class that provides basic volumetric data handling.

    Parameters
    ----------
    structure : Structure
        The crystal structure associated with the volumetric data.
        Represents the lattice and atomic coordinates using the `Structure` class.
    data : (dict[str, NDArray[float]])
        A dictionary containing the volumetric data. Keys include:
        - `"total"`: A 3D NumPy array representing the total spin density. If the
            data is ELF, represents the spin up ELF for spin-polarized calculations
            and the total ELF otherwise.
        - `"diff"` (optional): A 3D NumPy array representing the spin-difference
          density (spin up - spin down). If the data is ELF, represents the
          spin down ELF.
    data_aug : NDArray[float], optional
        Any extra information associated with volumetric data
        (typically augmentation charges)
    source_format : Format, optional
        The file format this grid was created from, 'vasp', 'cube', 'hdf5', or None.
    data_type : DataType, optional
        The type of data stored in the Grid object, either 'charge' or 'elf'. If
        None, the data type will be guessed from the data range.
    distance_matrix : NDArray[float], optional
        A pre-computed distance matrix if available.
        Useful so pass distance_matrices between sums,
        short-circuiting an otherwise expensive operation.
    sig_figs : int, optional
        The number of sig figs the data has. If None, this will be guessed.
    """

    def __init__(
        self,
        structure: Structure,
        data: dict,
        data_aug: dict = None,
        source_format: Format = None,
        data_type: DataType = DataType.charge,
        distance_matrix: NDArray[float] = None,
        sig_figs: int = None,
        **kwargs,
    ):
        # The following is copied directly from pymatgen, but replaces their
        # creation of a RegularGridInterpolator to avoid some overhead
        self.structure = Structure.from_dict(
            structure.as_dict()
        )  # convert to baderkit structure
        self.is_spin_polarized = len(data) >= 2
        self.is_soc = len(data) >= 4
        # convert data to numpy arrays in case they were jsanitized as lists
        self.data = {k: np.array(v) for k, v in data.items()}
        self.dim = self.data["total"].shape
        self.data_aug = data_aug or {}
        self.ngridpts = self.dim[0] * self.dim[1] * self.dim[2]
        # lazy init the spin data since this is not always needed.
        self._spin_data: dict[Spin, float] = {}
        self._distance_matrix = distance_matrix or {}
        self.xpoints = np.linspace(0.0, 1.0, num=self.dim[0])
        self.ypoints = np.linspace(0.0, 1.0, num=self.dim[1])
        self.zpoints = np.linspace(0.0, 1.0, num=self.dim[2])
        self.name = "VolumetricData"

        # The rest of this is new for BaderKit methods
        if source_format is None:
            source_format = Format.vasp
        self.source_format = Format(source_format)

        if sig_figs is None:
            sig_figs = infer_significant_figures(self.data["total"])
        self.sig_figs = sig_figs

        # custom interpolator setting
        self._cubic_spline_coeffs = None

        if data_type is None:
            # attempt to guess data type from data range
            if self.total.max() <= 1 and self.total.min() >= 0:
                data_type = DataType.elf
            else:
                data_type = DataType.charge
            logging.info(f"Data type set as {data_type.value} from data range")
        self.data_type = data_type

        # assign cached properties
        self._reset_cache()

    def _reset_cache(self):
        self._grid_indices = None
        self._flat_grid_indices = None
        self._point_dists = None
        self._max_point_dist = None
        self._grid_neighbor_transforms = None
        self._symmetry_data = None
        self._maxima_mask = None
        self._minima_mask = None

    @property
    def total(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            For charge densities, returns the total charge (spin-up + spin-down).
            For ELF returns the spin-up or single spin ELF.

        """
        return self.data["total"]

    @total.setter
    def total(self, new_total: NDArray[float]):
        self.data["total"] = new_total
        # reset cache
        self._reset_cache()

    @property
    def diff(self) -> NDArray[float] | None:
        """

        Returns
        -------
        NDArray[float]
            For charge densities, returns the magnetized charge (spin-up - spin-down).
            For ELF returns the spin-down ELF. If the file was not from a spin
            polarized calculation, this will be None.

        """
        return self.data.get("diff")

    @diff.setter
    def diff(self, new_diff):
        self.data["diff"] = new_diff
        # reset cache
        self._reset_cache()

    @property
    def shape(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The number of points along each axis of the grid.

        """
        return np.array(self.total.shape)

    @property
    def matrix(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            A 3x3 matrix defining the a, b, and c sides of the unit cell. Each
            row is the corresponding lattice vector in cartesian space.

        """
        return self.structure.lattice.matrix

    @property
    def grid_indices(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The indices for all points on the grid. Uses 'C' ordering.

        """
        if self._grid_indices is None:
            self._grid_indices = np.indices(self.shape).reshape(3, -1).T
        return self._grid_indices

    @property
    def flat_grid_indices(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            An array of the same shape as the grid where each entry is the index
            of that voxel if you were to flatten/ravel the grid. Uses 'C' ordering.

        """
        if self._flat_grid_indices is None:
            self._flat_grid_indices = np.arange(
                np.prod(self.shape), dtype=np.int64
            ).reshape(self.shape)
        return self._flat_grid_indices

    @property
    def cubic_spline_coeffs(self) -> NDArray[float]:
        if self._cubic_spline_coeffs is None:
            self._cubic_spline_coeffs = spline_filter(
                self.total, order=3, mode="grid-wrap"
            )

        return self._cubic_spline_coeffs

    # TODO: Do this with numba to reduce memory and probably increase speed
    @property
    def point_dists(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each point to the origin in cartesian coordinates.

        """
        if self._point_dists is None:
            cart_coords = self.grid_to_cart(self.grid_indices)
            a, b, c = self.matrix
            corners = [
                np.array([0, 0, 0]),
                a,
                b,
                c,
                a + b,
                a + c,
                b + c,
                a + b + c,
            ]
            distances = []
            for corner in corners:
                voxel_distances = np.linalg.norm(cart_coords - corner, axis=1).round(6)
                distances.append(voxel_distances)
            min_distances = np.min(np.column_stack(distances), axis=1)
            self._point_dists = min_distances.reshape(self.shape)
        return self._point_dists

    @property
    def point_volume(self) -> float:
        """

        Returns
        -------
        float
            The volume of a single point in the grid.

        """
        volume = self.structure.volume
        return volume / self.ngridpts

    @property
    def max_point_dist(self) -> float:
        """

        Returns
        -------
        float
            The maximum distance from the center of a point to one of its corners. This
            assumes the voxel is the same shape as the lattice.

        """
        if self._max_point_dist is None:
            # We need to find the coordinates that make up a single voxel. This
            # is just the cartesian coordinates of the unit cell divided by
            # its grid size
            a, b, c = self.matrix
            end = [0, 0, 0]
            vox_a = [x / self.shape[0] for x in a]
            vox_b = [x / self.shape[1] for x in b]
            vox_c = [x / self.shape[2] for x in c]
            # We want the three other vertices on the other side of the voxel. These
            # can be found by adding the vectors in a cycle (e.g. a+b, b+c, c+a)
            vox_a1 = [x + x1 for x, x1 in zip(vox_a, vox_b)]
            vox_b1 = [x + x1 for x, x1 in zip(vox_b, vox_c)]
            vox_c1 = [x + x1 for x, x1 in zip(vox_c, vox_a)]
            # The final vertex can be found by adding the last unsummed vector to any
            # of these
            end1 = [x + x1 for x, x1 in zip(vox_a1, vox_c)]
            # The center of the voxel sits exactly between the two ends
            center = [(x + x1) / 2 for x, x1 in zip(end, end1)]
            # Shift each point here so that the origin is the center of the
            # voxel.
            voxel_vertices = []
            for vector in [
                center,
                end,
                vox_a,
                vox_b,
                vox_c,
                vox_a1,
                vox_b1,
                vox_c1,
                end,
            ]:
                new_vector = [(x - x1) for x, x1 in zip(vector, center)]
                voxel_vertices.append(new_vector)

            # Now we need to find the maximum distance from the center of the voxel
            # to one of its edges. This should be at one of the vertices.
            # We can't say for sure which one is the largest distance so we find all
            # of their distances and return the maximum
            self._max_point_dist = max(
                [np.linalg.norm(vector) for vector in voxel_vertices]
            )
        return self._max_point_dist

    @cached_property
    def point_neighbor_voronoi_transforms(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray]
            The transformations, neighbor distances, areas, and vertices of the
            voronoi surface between any point and its neighbors in the grid.
            This is used in the 'weight' method for Bader analysis.

        """
        # I go out to 2 voxels away here. I think 1 would probably be fine, but
        # this doesn't take much more time and I'm certain this will capture the
        # full voronoi cell.
        voxel_positions = np.array(list(itertools.product([-2, -1, 0, 1, 2], repeat=3)))
        center = math.floor(len(voxel_positions) / 2)
        cart_positions = self.grid_to_cart(voxel_positions)
        voronoi = Voronoi(cart_positions)
        site_neighbors = []
        facet_vertices = []
        facet_areas = []

        def facet_area(vertices):
            # You can use a 2D or 3D area formula for a polygon
            # Here we assume the vertices are in a 2D plane for simplicity
            # For 3D, a more complicated approach (e.g., convex hull or triangulation) is needed
            p0 = np.array(vertices[0])
            area = 0
            for i in range(1, len(vertices) - 1):
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[i + 1])
                area += np.linalg.norm(np.cross(p1 - p0, p2 - p0)) / 2.0
            return area

        for i, neighbor_pair in enumerate(voronoi.ridge_points):
            if center in neighbor_pair:
                neighbor = [i for i in neighbor_pair if i != center][0]
                vertex_indices = voronoi.ridge_vertices[i]
                vertices = voronoi.vertices[vertex_indices]
                area = facet_area(vertices)
                site_neighbors.append(neighbor)
                facet_vertices.append(vertices)
                facet_areas.append(area)
        transforms = voxel_positions[np.array(site_neighbors)]
        cart_transforms = cart_positions[np.array(site_neighbors)]
        transform_dists = np.linalg.norm(cart_transforms, axis=1)
        return transforms, transform_dists, np.array(facet_areas), facet_vertices

    @cached_property
    def point_neighbor_transforms(self) -> (NDArray[int], NDArray[float]):
        """

        Returns
        -------
        (NDArray[int], NDArray[float])
            A tuple where the first entry is a 26x3 array of transformations in
            from any point to its neighbors and the second is the
            distance to each of these neighbors in cartesian space.

        """
        neighbors = np.array(
            [i for i in itertools.product([-1, 0, 1], repeat=3) if i != (0, 0, 0)]
        ).astype(np.int64)
        cart_coords = self.grid_to_cart(neighbors)
        dists = np.linalg.norm(cart_coords, axis=1)

        return neighbors, dists

    @cached_property
    def point_neighbor_face_tranforms(self) -> (NDArray[int], NDArray[float]):
        """

        Returns
        -------
        (NDArray[int], NDArray[float])
            A tuple where the first entry is a 6x3 array of transformations in
            voxel space from any voxel to its face sharing neighbors and the
            second is the distance to each of these neighbors in cartesian space.

        """
        all_neighbors, all_dists = self.point_neighbor_transforms
        faces = []
        dists = []
        for i in range(len(all_neighbors)):
            if np.sum(np.abs(all_neighbors[i])) == 1:
                faces.append(all_neighbors[i])
                dists.append(all_dists[i])
        return np.array(faces).astype(int), np.array(dists)

    @property
    def grid_neighbor_transforms(self) -> list:
        """
        The transforms for translating a grid index to neighboring unit
        cells. This is necessary for the many voxels that will not be directly
        within an atoms partitioning.

        Returns
        -------
        list
            A list of voxel grid_neighbor_transforms unique to the grid dimensions.

        """
        if self._grid_neighbor_transforms is None:
            a, b, c = self.shape
            grid_neighbor_transforms = [
                (t, u, v)
                for t, u, v in itertools.product([-a, 0, a], [-b, 0, b], [-c, 0, c])
            ]
            # sort grid_neighbor_transforms. There may be a better way of sorting them. I
            # noticed that generally the correct site was found most commonly
            # for the original site and generally was found at grid_neighbor_transforms that
            # were either all negative/0 or positive/0
            grid_neighbor_transforms_sorted = []
            for item in grid_neighbor_transforms:
                if all(val <= 0 for val in item):
                    grid_neighbor_transforms_sorted.append(item)
                elif all(val >= 0 for val in item):
                    grid_neighbor_transforms_sorted.append(item)
            for item in grid_neighbor_transforms:
                if item not in grid_neighbor_transforms_sorted:
                    grid_neighbor_transforms_sorted.append(item)
            grid_neighbor_transforms_sorted.insert(
                0, grid_neighbor_transforms_sorted.pop(7)
            )
            self._grid_neighbor_transforms = grid_neighbor_transforms_sorted
        return self._grid_neighbor_transforms

    @property
    def grid_resolution(self) -> float:
        """

        Returns
        -------
        float
            The number of voxels per unit volume.

        """
        volume = self.structure.volume
        number_of_voxels = self.ngridpts
        return number_of_voxels / volume

    @property
    def symmetry_data(self):
        """

        Returns
        -------
        TYPE
            The pymatgen symmetry dataset for the Grid's Structure object

        """
        if self._symmetry_data is None:
            self._symmetry_data = SpacegroupAnalyzer(
                self.structure
            ).get_symmetry_dataset()
        return self._symmetry_data

    @property
    def equivalent_atoms(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The equivalent atoms in the Structure.

        """
        return self.symmetry_data.equivalent_atoms

    @property
    def maxima_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask with the same dimensions as the data that is True at local
            maxima. Adjacent points with the same value will both be labeled as
            True.
        """
        if self._maxima_mask is None:
            # avoid circular import
            from baderkit.core.methods.shared_numba import get_maxima

            self._maxima_mask = get_maxima(
                self.total,
                neighbor_transforms=self.point_neighbor_transforms[0],
                vacuum_mask=np.zeros_like(self.total, dtype=np.bool_),
            )
        return self._maxima_mask

    @property
    def minima_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask with the same dimensions as the data that is True at local
            minima. Adjacent points with the same value will both be labeled as
            True.
        """
        if self._minima_mask is None:
            # avoid circular import
            from baderkit.core.methods.shared_numba import get_maxima

            self._minima_mask = get_maxima(
                self.total,
                neighbor_transforms=self.point_neighbor_transforms[0],
                vacuum_mask=np.zeros_like(self.total, dtype=np.bool_),
                use_minima=True,
            )
        return self._minima_mask

    def value_at(
        self,
        x: float,
        y: float,
        z: float,
        method: str = "cubic",
    ):
        """Get a data value from self.data at a given point (x, y, z) in terms
        of fractional lattice parameters. Will be interpolated using the
        provided method.

        Parameters
        ----------
        x : float
            Fraction of lattice vector a.
        y: float
            Fraction of lattice vector b.
        z: float
            Fraction of lattice vector c.
        method : float
            The method to use for interpolation. nearest, linear, or cubic. The
            cubic method will calculate and store spline coefficients in an
            array the same size as the grid, which increases memory usage

        Returns
        -------
        float
            Value from self.data (potentially interpolated) corresponding to
            the point (x, y, z).
        """
        if method == "cubic":
            data = self.cubic_spline_coeffs
        else:
            data = self.total

        interpolator = Interpolator(
            data=data,
            method=method,
        )
        # interpolate value
        return interpolator([x, y, z])[0]

    def values_at(
        self,
        frac_coords: NDArray[float],
        method: str = "cubic",
    ) -> list[float]:
        """
        Interpolates the value of the data at each fractional coordinate in a
        given list or array.

        Parameters
        ----------
        frac_coords : NDArray
            The fractional coordinates to interpolate values at with shape
            N, 3.
        method : float
            The method to use for interpolation. nearest, linear, or cubic. The
            cubic method will calculate and store spline coefficients in an
            array the same size as the grid, which increases memory usage

        Returns
        -------
        list[float]
            The interpolated value at each fractional coordinate.

        """
        if method == "cubic":
            data = self.cubic_spline_coeffs
        else:
            data = self.total

        interpolator = Interpolator(data=data, method=method)
        # interpolate values
        return interpolator(frac_coords)

    def linear_slice(
        self, p1: NDArray[float], p2: NDArray[float], n: int = 100, method="cubic"
    ):
        """
        Interpolates the data between two fractional coordinates.

        Parameters
        ----------
        p1 : NDArray[float]
            The fractional coordinates of the first point
        p2 : NDArray[float]
            The fractional coordinates of the second point
        n : int, optional
            The number of points to collect along the line
        method : float
            The method to use for interpolation. nearest, linear, or cubic. The
            cubic method will calculate and store spline coefficients in an
            array the same size as the grid, which increases memory usage

        Returns:
            List of n data points (mostly interpolated) representing a linear slice of the
            data from point p1 to point p2.
        """
        if type(p1) not in {list, np.ndarray}:
            raise TypeError(
                f"type of p1 should be list or np.ndarray, got {type(p1).__name__}"
            )
        if len(p1) != 3:
            raise ValueError(f"length of p1 should be 3, got {len(p1)}")
        if type(p2) not in {list, np.ndarray}:
            raise TypeError(
                f"type of p2 should be list or np.ndarray, got {type(p2).__name__}"
            )
        if len(p2) != 3:
            raise ValueError(f"length of p2 should be 3, got {len(p2)}")

        x_pts = np.linspace(p1[0], p2[0], num=n)
        y_pts = np.linspace(p1[1], p2[1], num=n)
        z_pts = np.linspace(p1[2], p2[2], num=n)
        frac_coords = np.column_stack((x_pts, y_pts, z_pts))
        return self.values_at(frac_coords, method)

    def get_box_around_point(self, point: NDArray, neighbor_size: int = 1) -> NDArray:
        """
        Gets a box around a given point taking into account wrapping at cell
        boundaries.

        Parameters
        ----------
        point : NDArray
            The indices of the point to get a box around.
        neighbor_size : int, optional
            The size of the box on either side of the point. The default is 1.

        Returns
        -------
        NDArray
            A slice of the grid taken around the provided point.

        """

        slices = []
        for dim, c in zip(self.shape, point):
            idx = np.arange(c - neighbor_size, c + 2) % dim
            idx = idx.astype(int)
            slices.append(idx)
        return self.total[np.ix_(slices[0], slices[1], slices[2])]

    @staticmethod
    def get_2x_supercell(data: NDArray | None = None) -> NDArray:
        """
        Duplicates data to make a 2x2x2 supercell

        Parameters
        ----------
        data : NDArray | None, optional
            The data to duplicate. The default is None.

        Returns
        -------
        NDArray
            A new array with the data doubled in each direction
        """
        new_data = np.tile(data, (2, 2, 2))
        return new_data

    def get_points_in_radius(
        self,
        point: NDArray,
        radius: float,
    ) -> NDArray[int]:
        """
        Gets the indices of the points in a radius around a point

        Parameters
        ----------
        radius : float
            The radius in cartesian distance units to find indices around the
            point.
        point : NDArray
            The indices of the point to perform the operation on.

        Returns
        -------
        NDArray[int]
            The point indices in the sphere around the provided point.

        """
        point = np.array(point)
        # Get the distance from each point to the origin
        point_distances = self.point_dists

        # Get the indices that are within the radius
        sphere_indices = np.where(point_distances <= radius)
        sphere_indices = np.column_stack(sphere_indices)

        # Get indices relative to the point
        sphere_indices = sphere_indices + point
        # adjust points to wrap around grid
        # line = [[round(float(a % b), 12) for a, b in zip(position, grid_data.shape)]]
        new_x = (sphere_indices[:, 0] % self.shape[0]).astype(int)
        new_y = (sphere_indices[:, 1] % self.shape[1]).astype(int)
        new_z = (sphere_indices[:, 2] % self.shape[2]).astype(int)
        sphere_indices = np.column_stack([new_x, new_y, new_z])
        # return new_x, new_y, new_z
        return sphere_indices

    def get_transformation_in_radius(self, radius: float) -> NDArray[int]:
        """
        Gets the transformations required to move from a point to the points
        surrounding it within the provided radius

        Parameters
        ----------
        radius : float
            The radius in cartesian distance units around the voxel.

        Returns
        -------
        NDArray[int]
            An array of transformations to add to a point to get to each of the
            points within the radius surrounding it.

        """
        # Get voxels around origin
        voxel_distances = self.point_dists
        # sphere_grid = np.where(voxel_distances <= radius, True, False)
        # eroded_grid = binary_erosion(sphere_grid)
        # shell_indices = np.where(sphere_grid!=eroded_grid)
        shell_indices = np.where(voxel_distances <= radius)
        # Now we want to translate these indices to next to the corner so that
        # we can use them as transformations to move a voxel to the edge
        final_shell_indices = []
        for a, x in zip(self.shape, shell_indices):
            new_x = x - a
            abs_new_x = np.abs(new_x)
            new_x_filter = abs_new_x < x
            final_x = np.where(new_x_filter, new_x, x)
            final_shell_indices.append(final_x)

        return np.column_stack(final_shell_indices)

    def copy(self) -> Self:
        """
        Convenience method to get a copy of the current Grid.

        Returns
        -------
        Self
            A copy of the Grid.

        """
        return Grid(
            structure=self.structure.copy(),
            data=self.data.copy(),
            data_aug=self.data_aug.copy(),
            source_format=self.source_format,
            data_type=self.data_type,
            distance_matrix=self._distance_matrix.copy(),
        )

    def regrid(
        self,
        desired_resolution: int = 1200,
        new_shape: np.array = None,
        order: int = 3,
    ) -> Self:
        """
        Returns a new grid resized using scipy's ndimage.zoom method

        Parameters
        ----------
        desired_resolution : int, optional
            The desired resolution in voxels/A^3. The default is 1200.
        new_shape : np.array, optional
            The new array shape. Takes precedence over desired_resolution. The default is None.
        order : int, optional
            The order of spline interpolation to use. The default is 3.

        Returns
        -------
        Self
            A new Grid object near the desired resolution.
        """

        # get the original grid size and lattice volume.
        shape = self.shape
        volume = self.structure.volume

        if new_shape is None:
            # calculate how much the number of voxels along each unit cell must be
            # multiplied to reach the desired resolution.
            scale_factor = ((desired_resolution * volume) / shape.prod()) ** (1 / 3)

            # calculate the new grid shape. round up to the nearest integer for each
            # side
            new_shape = np.around(shape * scale_factor).astype(np.int32)

        # get the factor to zoom by
        zoom_factor = new_shape / shape

        # zoom each piece of data
        new_data = {}
        for key, data in self.data.items():
            new_data[key] = zoom(
                data, zoom_factor, order=order, mode="grid-wrap", grid_mode=True
            )

        # TODO: Add augment data?
        return Grid(structure=self.structure, data=new_data)

    def split_to_spin(self) -> tuple[Self, Self]:
        """
        Splits the grid to two Grid objects representing the spin up and spin down contributions

        Returns
        -------
        tuple[Self, Self]
            The spin-up and spin-down Grid objects.

        """

        # first check if the grid has spin parts
        assert (
            self.is_spin_polarized
        ), "Only one set of data detected. The grid cannot be split into spin up and spin down"
        assert not self.is_soc

        # Now we get the separate data parts. If the data is ELF, the parts are
        # stored as total=spin up and diff = spin down
        if self.data_type == "elf":
            logging.info(
                "Splitting Grid using ELFCAR conventions (spin-up in 'total', spin-down in 'diff')"
            )
            spin_up_data = self.total.copy()
            spin_down_data = self.diff.copy()
        elif self.data_type == "charge":
            logging.info(
                "Splitting Grid using CHGCAR conventions (spin-up + spin-down in 'total', spin-up - spin-down in 'diff')"
            )
            spin_data = self.spin_data
            # pymatgen uses some custom class as keys here
            for key in spin_data.keys():
                if key.value == 1:
                    spin_up_data = spin_data[key].copy()
                elif key.value == -1:
                    spin_down_data = spin_data[key].copy()

        # convert to dicts
        spin_up_data = {"total": spin_up_data}
        spin_down_data = {"total": spin_down_data}

        # get augment data
        aug_up_data = (
            {"total": self.data_aug["total"]} if "total" in self.data_aug else {}
        )
        aug_down_data = (
            {"total": self.data_aug["diff"]} if "diff" in self.data_aug else {}
        )

        spin_up_grid = Grid(
            structure=self.structure.copy(),
            data=spin_up_data,
            data_aug=aug_up_data,
            data_type=self.data_type,
            source_format=self.source_format,
        )
        spin_down_grid = Grid(
            structure=self.structure.copy(),
            data=spin_down_data,
            data_aug=aug_down_data,
            data_type=self.data_type,
            source_format=self.source_format,
        )

        return spin_up_grid, spin_down_grid

    @staticmethod
    def label(input: NDArray, structure: NDArray = np.ones([3, 3, 3])) -> NDArray[int]:
        """
        Uses scipy's ndimage package to label an array, and corrects for
        periodic boundaries

        Parameters
        ----------
        input : NDArray
            The array to label.
        structure : NDArray, optional
            The structureing elemetn defining feature connections.
            The default is np.ones([3, 3, 3]).

        Returns
        -------
        NDArray[int]
            An array of the same shape as the original with labels for each unique
            feature.

        """

        if structure is not None:
            labeled_array, _ = label(input, structure)
            if len(np.unique(labeled_array)) == 1:
                # there is one feature or no features
                return labeled_array
            # Features connected through opposite sides of the unit cell should
            # have the same label, but they don't currently. To handle this, we
            # pad our featured grid, re-label it, and check if the new labels
            # contain multiple of our previous labels.
            padded_featured_grid = np.pad(labeled_array, 1, "wrap")
            relabeled_array, label_num = label(padded_featured_grid, structure)
        else:
            labeled_array, _ = label(input)
            padded_featured_grid = np.pad(labeled_array, 1, "wrap")
            relabeled_array, label_num = label(padded_featured_grid)

        # We want to keep track of which features are connected to each other
        unique_connections = [[] for i in range(len(np.unique(labeled_array)))]

        for i in np.unique(relabeled_array):
            # for i in range(label_num):
            # Get the list of features that are in this super feature
            mask = relabeled_array == i
            connected_features = list(np.unique(padded_featured_grid[mask]))
            # Iterate over these features. If they exist in a connection that we
            # already have, we want to extend the connection to include any other
            # features in this super feature
            for j in connected_features:

                unique_connections[j].extend([k for k in connected_features if k != j])

                unique_connections[j] = list(np.unique(unique_connections[j]))

        # create set/list to keep track of which features have already been connected
        # to others and the full list of connections
        already_connected = set()
        reduced_connections = []

        # loop over each shared connection
        for i in range(len(unique_connections)):
            if i in already_connected:
                # we've already done these connections, so we skip
                continue
            # create sets of connections to compare with as we add more
            connections = set()
            new_connections = set(unique_connections[i])
            while connections != new_connections:
                # loop over the connections we've found so far. As we go, add
                # any features we encounter to our set.
                connections = new_connections.copy()
                for j in connections:
                    already_connected.add(j)
                    new_connections.update(unique_connections[j])

            # If we found any connections, append them to our list of reduced connections
            if connections:
                reduced_connections.append(sorted(new_connections))

        # For each set of connections in our reduced set, relabel all values to
        # the lowest one.
        for connections in reduced_connections:
            connected_features = np.unique(connections)
            lowest_idx = connected_features[0]
            for higher_idx in connected_features[1:]:
                labeled_array = np.where(
                    labeled_array == higher_idx, lowest_idx, labeled_array
                )

        # Now we reduce the feature labels so that they start at 0
        for i, j in enumerate(np.unique(labeled_array)):
            labeled_array = np.where(labeled_array == j, i, labeled_array)

        return labeled_array

    def linear_add(self, other: Self, scale_factor=1.0) -> Self:
        """
        Method to do a linear sum of volumetric objects. Used by + and -
        operators as well. Returns a VolumetricData object containing the
        linear sum.

        Parameters
        ----------
        other : Grid
            Another Grid object
        scale_factor : float
            Factor to scale the other data by

        Returns
        -------
            Grid corresponding to self + scale_factor * other.
        """
        if self.structure != other.structure:
            logging.warn(
                "Structures are different. Make sure you know what you are doing...",
                stacklevel=2,
            )
        if list(self.data) != list(other.data):
            raise ValueError(
                "Data have different keys! Maybe one is spin-polarized and the other is not?"
            )

        # To add checks
        data = {}
        for k in self.data:
            data[k] = self.data[k] + scale_factor * other.data[k]

        new = deepcopy(self)
        new.data = data.copy()
        new.data_aug = {}  # TODO: Can this be added somehow?
        return new

    ###########################################################################
    # The following is a series of methods that are useful for converting between
    # voxel coordinates, fractional coordinates, and cartesian coordinates.
    # Voxel coordinates go from 0 to grid_size-1. Fractional coordinates go
    # from 0 to 1. Cartesian coordinates convert to real space based on the
    # crystal lattice.
    ###########################################################################
    def get_voxel_coords_from_index(self, site: int) -> NDArray[int]:
        """
        Takes in an atom's site index and returns the equivalent voxel grid index.

        Parameters
        ----------
        site : int
            The index of the site to find the grid index for.

        Returns
        -------
        NDArray[int]
            A voxel grid index.

        """
        return self.frac_to_grid(self.structure[site].frac_coords)

    def get_voxel_coords_from_neigh_CrystalNN(self, neigh) -> NDArray[int]:
        """
        Gets the voxel grid index from a neighbor atom object from CrystalNN or
        VoronoiNN

        Parameters
        ----------
        neigh :
            A neighbor type object from pymatgen.

        Returns
        -------
        NDArray[int]
            A voxel grid index as an array.

        """
        grid_size = self.shape
        frac = neigh["site"].frac_coords
        voxel_coords = [a * b for a, b in zip(grid_size, frac)]
        # voxel positions go from 1 to (grid_size + 0.9999)
        return np.array(voxel_coords)

    def get_voxel_coords_from_neigh(self, neigh: dict) -> NDArray[int]:
        """
        Gets the voxel grid index from a neighbor atom object from the pymatgen
        structure.get_neighbors class.

        Parameters
        ----------
        neigh : dict
            A neighbor dictionary from pymatgens structure.get_neighbors
            method.

        Returns
        -------
        NDArray[int]
            A voxel grid index as an array.

        """

        grid_size = self.shape
        frac_coords = neigh.frac_coords
        voxel_coords = [a * b for a, b in zip(grid_size, frac_coords)]
        # voxel positions go from 1 to (grid_size + 0.9999)
        return np.array(voxel_coords)

    def cart_to_frac(self, cart_coords: NDArray | list) -> NDArray[float]:
        """
        Takes in a cartesian coordinate and returns the fractional coordinates.

        Parameters
        ----------
        cart_coords : NDArray | list
            An Nx3 Array or 1D array of length 3.

        Returns
        -------
        NDArray[float]
            Fractional coordinates as an Nx3 Array.

        """
        inverse_matrix = np.linalg.inv(self.matrix)

        return cart_coords @ inverse_matrix

    def cart_to_grid(self, cart_coords: NDArray | list) -> NDArray[int]:
        """
        Takes in a cartesian coordinate and returns the voxel coordinates.

        Parameters
        ----------
        cart_coords : NDArray | list
            An Nx3 Array or 1D array of length 3.

        Returns
        -------
        NDArray[int]
            Voxel coordinates as an Nx3 Array.

        """
        frac_coords = self.cart_to_frac(cart_coords)
        voxel_coords = self.frac_to_grid(frac_coords)
        return voxel_coords

    def frac_to_cart(self, frac_coords: NDArray) -> NDArray[float]:
        """
        Takes in a fractional coordinate and returns the cartesian coordinates.

        Parameters
        ----------
        frac_coords : NDArray | list
            An Nx3 Array or 1D array of length 3.

        Returns
        -------
        NDArray[float]
            Cartesian coordinates as an Nx3 Array.

        """

        return frac_coords @ self.matrix

    def grid_to_frac(self, vox_coords: NDArray) -> NDArray[float]:
        """
        Takes in a voxel coordinates and returns the fractional coordinates.

        Parameters
        ----------
        vox_coords : NDArray | list
            An Nx3 Array or 1D array of length 3.

        Returns
        -------
        NDArray[float]
            Fractional coordinates as an Nx3 Array.

        """

        return vox_coords / self.shape

    def frac_to_grid(self, frac_coords: NDArray) -> NDArray[int]:
        """
        Takes in a fractional coordinates and returns the voxel coordinates.

        Parameters
        ----------
        frac_coords : NDArray | list
            An Nx3 Array or 1D array of length 3.

        Returns
        -------
        NDArray[int]
            Voxel coordinates as an Nx3 Array.

        """
        return frac_coords * self.shape

    def grid_to_cart(self, vox_coords: NDArray) -> NDArray[float]:
        """
        Takes in a voxel coordinates and returns the cartesian coordinates.

        Parameters
        ----------
        vox_coords : NDArray | list
            An Nx3 Array or 1D array of length 3.

        Returns
        -------
        NDArray[float]
            Cartesian coordinates as an Nx3 Array.

        """
        frac_coords = self.grid_to_frac(vox_coords)
        return self.frac_to_cart(frac_coords)

    ###########################################################################
    # Functions for loading from files or strings
    ###########################################################################

    @classmethod
    def from_vasp(
        cls,
        grid_file: str | Path,
        data_type: str | DataType = None,
        total_only: bool = True,
        **kwargs,
    ) -> Self:
        """
        Create a grid instance using a CHGCAR or ELFCAR file.

        Parameters
        ----------
        grid_file : str | Path
            The file the instance should be made from. Should be a VASP
            CHGCAR or ELFCAR type file.
        data_type: str | DataType
            The type of data loaded from the file, either charge or elf. If
            None, the type will be guessed from the data range.
            Defaults to None.
        total_only: bool
            If true, only the first set of data in the file will be read. This
            increases speed and reduced memory usage for methods that do not
            use the spin data.
            Defaults to True.

        Returns
        -------
        Self
            Grid from the specified file.

        """
        logging.info(f"Loading {grid_file}")
        t0 = time.time()
        # get structure and data from file
        grid_file = Path(grid_file)
        # check that file exists
        assert grid_file.exists(), f"No file with name {grid_file} found in directory"
        structure, data, data_aug, sig_figs = read_vasp(
            grid_file, total_only=total_only
        )
        t1 = time.time()
        logging.info(f"Time: {round(t1-t0,2)}")
        return cls(
            structure=structure,
            data=data,
            data_aug=data_aug,
            data_type=data_type,
            source_format=Format.vasp,
            sig_figs=sig_figs,
            **kwargs,
        )

    @classmethod
    def from_cube(
        cls,
        grid_file: str | Path,
        data_type: str | DataType = None,
        **kwargs,
    ) -> Self:
        """
        Create a grid instance using a gaussian cube file.

        Parameters
        ----------
        grid_file : str | Path
            The file the instance should be made from. Should be a gaussian
            cube file.
        data_type: str | DataType
            The type of data loaded from the file, either charge or elf. If
            None, the type will be guessed from the data range.
            Defaults to None.

        Returns
        -------
        Self
            Grid from the specified file.

        """
        logging.info(f"Loading {grid_file}")
        t0 = time.time()
        # make sure path is a Path object
        grid_file = Path(grid_file)
        # check that file exists
        assert grid_file.exists(), f"No file with name {grid_file} found in directory"
        structure, data, ion_charges, origin, sig_figs = read_cube(grid_file)
        # TODO: Also save the ion charges/origin for writing later
        t1 = time.time()
        logging.info(f"Time: {round(t1-t0,2)}")
        return cls(
            structure=structure,
            data=data,
            data_type=data_type,
            source_format=Format.cube,
            sig_figs=sig_figs,
            **kwargs,
        )

    @classmethod
    def from_vasp_pymatgen(
        cls,
        grid_file: str | Path,
        data_type: str | DataType = None,
        **kwargs,
    ) -> Self:
        """
        Create a grid instance using a CHGCAR or ELFCAR file. Uses pymatgen's
        parse_file method which is often surprisingly slow.

        Parameters
        ----------
        grid_file : str | Path
            The file the instance should be made from. Should be a VASP
            CHGCAR or ELFCAR type file.
        data_type: str | DataType
            The type of data loaded from the file, either charge or elf. If
            None, the type will be guessed from the data range.
            Defaults to None.

        Returns
        -------
        Self
            Grid from the specified file.

        """
        logging.info(f"Loading {grid_file}")
        t0 = time.time()
        # make sure path is a Path object
        grid_file = Path(grid_file)
        # check that file exists
        assert grid_file.exists(), f"No file with name {grid_file} found in directory"
        # Create string to add structure to.
        poscar, data, data_aug = cls.parse_file(grid_file)
        t1 = time.time()
        logging.info(f"Time: {round(t1-t0,2)}")
        return cls(
            structure=poscar.structure,
            data=data,
            data_aug=data_aug,
            source_format=Format.vasp,
            data_type=data_type,
            **kwargs,
        )

    @classmethod
    def from_hdf5(
        cls,
        grid_file: str | Path,
        data_type: str | DataType = None,
        **kwargs,
    ) -> Self:
        """
        Create a grid instance using an hdf5 file.

        Parameters
        ----------
        grid_file : str | Path
            The file the instance should be made from. Should be a binary hdf5
            file.
        data_type: str | DataType
            The type of data loaded from the file, either charge or elf. If
            None, the type will be guessed from the data range.
            Defaults to None.

        Returns
        -------
        Self
            Grid from the specified file.

        """
        try:
            import h5py
        except:
            raise ImportError(
                """
                The `h5py` package is required to read/write to the hdf5 format.
                Please install with `conda install h5py` or `pip install h5py`.
                """
            )

        logging.info(f"Loading {grid_file}")
        t0 = time.time()
        # make sure path is a Path object
        grid_file = Path(grid_file)

        # check that file exists
        assert grid_file.exists(), f"No file with name {grid_file} found in directory"
        # load the file
        pymatgen_grid = super().from_hdf5(filename=grid_file)
        t1 = time.time()
        logging.info(f"Time: {round(t1-t0,2)}")
        return cls(
            structure=pymatgen_grid.structure,
            data=pymatgen_grid.data,
            data_aug=pymatgen_grid.data_aug,
            source_format=Format.hdf5,
            data_type=data_type,
            **kwargs,
        )

    @classmethod
    def from_dynamic(
        cls,
        grid_file: str | Path,
        format: str | Format = None,
        **kwargs,
    ) -> Self:
        """
        Create a grid instance using a VASP or .cube file. If no format is provided
        the format is guesed by the name of the file.

        Parameters
        ----------
        grid_file : str | Path
            The file the instance should be made from.
        format : Format, optional
            The format of the provided file. If None, a guess will be made based
            on the name of the file. Setting this is identical to calling the
            from methods for the corresponding file type. The default is None.

        Returns
        -------
        Self
            Grid from the specified file.

        """
        grid_file = Path(grid_file)

        # check that file exists
        assert grid_file.exists(), f"No file with name {grid_file} found in directory"

        if format is None:
            # guess format from file
            format = detect_format(grid_file)

        # make sure format is an available option
        assert (
            format in Format
        ), "Invalid provided format '{format}'. Options are: {[i.value for i in Format]}"

        # get the reading method corresponding to this output format
        method_name = format.reader

        # load from file
        return getattr(cls, method_name)(grid_file, **kwargs)

    def write_vasp(
        self,
        filename: Path | str,
        vasp4_compatible: bool = False,
    ):
        """
        Writes the Grid to a VASP-like file at the provided path.

        Parameters
        ----------
        filename : Path | str
            The name of the file to write to.

        Returns
        -------
        None.

        """
        filename = Path(filename)
        logging.info(f"Writing {filename.name}")
        write_vasp_file(filename=filename, grid=self, vasp4_compatible=vasp4_compatible)

    def write_cube(
        self,
        filename: Path | str,
        **kwargs,
    ):
        """
        Writes the Grid to a Gaussian cube-like file at the provided path.

        Parameters
        ----------
        filename : Path | str
            The name of the file to write to.

        Returns
        -------
        None.

        """
        filename = Path(filename)
        logging.info(f"Writing {filename.name}")
        write_cube_file(
            filename=filename,
            grid=self,
            **kwargs,
        )

    def to_hdf5(
        self,
        filename: Path | str,
        **kwargs,
    ):
        try:
            import h5py
        except:
            raise ImportError(
                """
                The `h5py` package is required to read/write to the hdf5 format.
                Please install with `conda install h5py` or `pip install h5py`.
                """
            )
        filename = Path(filename)
        logging.info(f"Writing {filename.name}")
        super().to_hdf5(filename)

    def write(
        self,
        filename: Path | str,
        output_format: Format | str = None,
        **kwargs,
    ):
        """
        Writes the Grid to the requested format file at the provided path. If no
        format is provided, uses this Grid objects stored format.

        Parameters
        ----------
        filename : Path | str
            The name of the file to write to.
        output_format : Format | str
            The format to write with. If None, writes to source format stored in
            this Grid objects metadata.
            Defaults to None.

        Returns
        -------
        None.

        """
        # If no provided format, get from metadata
        if output_format is None:
            output_format = self.source_format
        # Make sure format is a Format object not a string
        output_format = Format(output_format)
        # get the writing method corresponding to this output format
        method_name = output_format.writer
        # write the grid
        getattr(self, method_name)(filename, **kwargs)
