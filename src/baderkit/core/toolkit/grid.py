# -*- coding: utf-8 -*-

import itertools
import logging
import math
from copy import deepcopy
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from pymatgen.io.vasp import VolumetricData
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, label, zoom
from scipy.spatial import Voronoi

from baderkit.core.toolkit.file_parsers import (
    Format,
    detect_format,
    read_cube,
    read_vasp,
)
from baderkit.core.toolkit.file_parsers import write_cube as write_cube_file
from baderkit.core.toolkit.file_parsers import write_vasp as write_vasp_file
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
        The file format this grid was created from, either 'vasp' or 'cube'.
    data_type : DataType, optional
        The type of data stored in the Grid object, either 'charge' or 'elf'. If
        None, the data type will be guessed from the data range.
    distance_matrix : NDArray[float], optional
        A pre-computed distance matrix if available.
        Useful so pass distance_matrices between sums,
        short-circuiting an otherwise expensive operation.
    interpolation_method : str, optional
        The method of interpolation for methods that require it. See SciPy's
        [RegularGridInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html).

    """

    def __init__(
        self,
        structure: Structure,
        data: dict,
        data_aug: dict = None,
        source_format: Format = None,
        data_type: DataType = DataType.charge,
        distance_matrix: NDArray[float] = None,
        interpolation_method: str = "linear",
    ):
        super().__init__(
            structure=structure,
            data=data,
            data_aug=data_aug,
            distance_matrix=distance_matrix,
        )
        # convert structure to baderkit utility version
        self.structure = Structure.from_dict(self.structure.as_dict())
        self.format = source_format

        if data_type is None:
            # attempt to guess data type from data range
            if self.total.max() <= 1 and self.total.min() >= 0:
                data_type = DataType.elf
            else:
                data_type = DataType.charge
            logging.info(f"Data type set as {data_type.value} from data range")
        self.data_type = data_type

        # pymatgen always sets their RegularGridInterpolator with linear interpolation
        # but that isn't always what we want. Additionally, I have found some
        # padding of the grid is usually required to get accurate interpolation
        # near the edges.
        x, y, z = self.dim
        pad = 10
        padded_total = np.pad(self.data["total"], pad, mode="wrap")
        xpoints_pad = np.linspace(-pad, x + pad - 1, x + pad * 2) / x
        ypoints_pad = np.linspace(-pad, y + pad - 1, y + pad * 2) / y
        zpoints_pad = np.linspace(-pad, z + pad - 1, z + pad * 2) / z
        self.interpolator = RegularGridInterpolator(
            (xpoints_pad, ypoints_pad, zpoints_pad),
            padded_total,
            method=interpolation_method,
            bounds_error=True,
        )

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
    ):
        """Get a data value from self.data at a given point (x, y, z) in terms
        of fractional lattice parameters. Will be interpolated using a
        RegularGridInterpolator on self.data if (x, y, z) is not in the original
        set of data points.

        Parameters
        ----------
        x : float
            Fraction of lattice vector a.
        y: float
            Fraction of lattice vector b.
        z: float
            Fraction of lattice vector c.

        Returns
        -------
        float
            Value from self.data (potentially interpolated) corresponding to
            the point (x, y, z).
        """
        # wrap point
        x %= 1
        y %= 1
        z %= 1
        # interpolate value
        return self.interpolator([x, y, z])[0]

    def values_at(
        self,
        frac_coords: NDArray[float],
    ) -> list[float]:
        """
        Interpolates the value of the data at each fractional coordinate in a
        given list or array.

        Parameters
        ----------
        frac_coords : NDArray
            The fractional coordinates to interpolate values at with shape
            N, 3.

        Returns
        -------
        list[float]
            The interpolated value at each fractional coordinate.

        """
        # make sure we have an array
        frac_coords = np.array(frac_coords)

        # wrap coords
        frac_coords %= 1

        # interpolate values
        return self.interpolator(frac_coords)

    def linear_slice(self, p1: NDArray[float], p2: NDArray[float], n: int = 100):
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
        return self.values_at(frac_coords)

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

    def climb_to_max(self, frac_coords: NDArray) -> NDArray[float]:
        """
        Hill climbs to a maximum from the provided fractional coordinate.

        Parameters
        ----------
        frac_coords : NDArray
            The starting coordinate for hill climbing.

        Returns
        -------
        NDArray[float]
            The final fractional coordinates after hill climbing.
        float
            The data value at the found maximum

        """
        # Convert to voxel coords and round
        coords = np.round(self.frac_to_grid(frac_coords)).astype(int)
        # wrap around edges of cell
        coords %= self.shape
        i, j, k = coords

        # import numba function to avoid circular import
        from baderkit.core.methods.shared_numba import climb_to_max

        # get neighbors and dists
        neighbor_transforms, neighbor_dists = self.point_neighbor_transforms
        # get max
        mi, mj, mk = climb_to_max(
            self.total, i, j, k, neighbor_transforms, neighbor_dists
        )
        # get value at max
        max_val = self.total[mi, mj, mk]
        # Now we check if this point borders other points with the same value
        box = self.get_box_around_point((mi, mj, mk))
        all_max = np.argwhere(box == max_val)
        avg_pos = all_max.mean(axis=1)
        local_offset = avg_pos - 1  # shift from subset center
        current_coords = np.array((mi, mj, mk)) + local_offset
        current_coords %= self.shape

        new_frac_coords = self.grid_to_frac(current_coords)
        x, y, z = new_frac_coords

        return new_frac_coords, self.value_at(x, y, z)

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

    # def get_padded_grid_axes(
    #     self, padding: int = 0
    # ) -> tuple[NDArray, NDArray, NDArray]:
    #     """
    #     Gets the the possible indices for each dimension of a padded grid.
    #     e.g. if the original charge density grid is 20x20x20, and is padded
    #     with one extra layer on each side, this function will return three
    #     arrays with integers from 0 to 21.

    #     Parameters
    #     ----------
    #     padding : int, optional
    #         The amount the grid has been padded. The default is 0.

    #     Returns
    #     -------
    #     tuple[NDArray, NDArray, NDArray]
    #         Three arrays with lengths the same as the grids shape.

    #     """

    #     grid = self.total
    #     a = np.linspace(
    #         0,
    #         grid.shape[0] + (padding - 1) * 2 + 1,
    #         grid.shape[0] + padding * 2,
    #     )
    #     b = np.linspace(
    #         0,
    #         grid.shape[1] + (padding - 1) * 2 + 1,
    #         grid.shape[1] + padding * 2,
    #     )
    #     c = np.linspace(
    #         0,
    #         grid.shape[2] + (padding - 1) * 2 + 1,
    #         grid.shape[2] + padding * 2,
    #     )
    #     return a, b, c

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
            data_aug=self.data_aug,
        )

    def get_atoms_in_volume(self, volume_mask: NDArray[bool]) -> NDArray[int]:
        """
        Checks if an atom is within the provided volume. This only checks the
        point write where the atom is located, so a shell around the atom will
        not be caught

        Parameters
        ----------
        volume_mask : NDArray[bool]
            A mask of the same shape as the current grid.

        Returns
        -------
        NDArray[int]
            A list of atoms in the provided mask.

        """
        # Make sure the shape of the mask is the same as the grid
        assert np.all(
            np.equal(self.shape, volume_mask.shape)
        ), "Mask and Grid must be the same shape"
        # Get the voxel coordinates for each atom
        site_voxel_coords = self.frac_to_grid(self.structure.frac_coords).astype(int)
        # Return the indices of the atoms that are in the mask
        atoms_in_volume = volume_mask[
            site_voxel_coords[:, 0], site_voxel_coords[:, 1], site_voxel_coords[:, 2]
        ]
        return np.argwhere(atoms_in_volume)

    def get_atoms_surrounded_by_volume(
        self, volume_mask: NDArray[bool], return_type: bool = False
    ) -> NDArray[int]:
        """
        Checks if a mask completely surrounds any of the atoms
        in the structure. This method uses scipy's ndimage package to
        label features in the grid combined with a supercell to check
        if atoms identical through translation are connected.

        Parameters
        ----------
        volume_mask : NDArray[bool]
            A mask of the same shape as the current grid.
        return_type : bool, optional
            Whether or not to return the type of surrounding. 0 indicates that
            the atom sits exactly in the volume. 1 indicates that it is surrounded
            but not directly in it. The default is False.

        Returns
        -------
        NDArray[int]
            The atoms that are surrounded by this mask.

        """
        # Make sure the shape of the mask is the same as the grid
        assert np.all(
            np.equal(self.shape, volume_mask.shape)
        ), "Mask and Grid must be the same shape"
        # first we get any atoms that are within the mask itself. These won't be
        # found otherwise because they will always sit in unlabeled regions.
        structure = np.ones([3, 3, 3])
        dilated_mask = binary_dilation(volume_mask, structure)
        init_atoms = self.get_atoms_in_volume(dilated_mask)
        # check if we've surrounded all of our atoms. If so, we can return and
        # skip the rest
        if len(init_atoms) == len(self.structure):
            return init_atoms, np.zeros(len(init_atoms))
        # Now we create a supercell of the mask so we can check connections to
        # neighboring cells. This will be used to check if the feature connects
        # to itself in each direction
        dilated_supercell_mask = self.get_2x_supercell(dilated_mask)
        # We also get an inversion of this mask. This will be used to check if
        # the mask surrounds each atom. To do this, we use the dilated supercell
        # We do this to avoid thin walls being considered connections
        # in the inverted mask
        inverted_mask = dilated_supercell_mask == False
        # Now we use use scipy to label unique features in our masks

        inverted_feature_supercell = self.label(inverted_mask, structure)

        # if an atom was fully surrounded, it should sit inside one of our labels.
        # The same atom in an adjacent unit cell should have a different label.
        # To check this, we need to look at the atom in each section of the supercell
        # and see if it has a different label in each.
        # Similarly, if the feature is disconnected from itself in each unit cell
        # any voxel in the feature should have different labels in each section.
        # If not, the feature is connected to itself in multiple directions and
        # must surround many atoms.
        transformations = np.array(list(itertools.product([0, 1], repeat=3)))
        transformations = self.frac_to_grid(transformations)
        # Check each atom to determine how many atoms it surrounds
        surrounded_sites = []
        for i, site in enumerate(self.structure):
            # Get the voxel coords of each atom in their equivalent spots in each
            # quadrant of the supercell
            frac_coords = site.frac_coords
            voxel_coords = self.frac_to_grid(frac_coords)
            transformed_coords = (transformations + voxel_coords).astype(int)
            # Get the feature label at each transformation. If the atom is not surrounded
            # by this basin, at least some of these feature labels will be the same
            features = inverted_feature_supercell[
                transformed_coords[:, 0],
                transformed_coords[:, 1],
                transformed_coords[:, 2],
            ]
            if len(np.unique(features)) == 8:
                # The atom is completely surrounded by this basin and the basin belongs
                # to this atom
                surrounded_sites.append(i)
        surrounded_sites.extend(init_atoms)
        surrounded_sites = np.unique(surrounded_sites)
        types = []
        for site in surrounded_sites:
            if site in init_atoms:
                types.append(0)
            else:
                types.append(1)
        if return_type:
            return surrounded_sites, types
        return surrounded_sites

    def check_if_infinite_feature(self, volume_mask: NDArray[bool]) -> bool:
        """
        Checks if a mask extends infinitely in at least one direction.
        This method uses scipy's ndimage package to label features in the mask
        combined with a supercell to check if the label matches between unit cells.

        Parameters
        ----------
        volume_mask : NDArray[bool]
            A mask of the same shape as the current grid.

        Returns
        -------
        bool
            Whether or not this is an infinite feature.

        """
        # First we check that there is at least one feature in the mask. If not
        # we return False as there is no feature.
        if (~volume_mask).all():
            return False

        structure = np.ones([3, 3, 3])
        # Now we create a supercell of the mask so we can check connections to
        # neighboring cells. This will be used to check if the feature connects
        # to itself in each direction
        supercell_mask = self.get_2x_supercell(volume_mask)
        # Now we use use scipy to label unique features in our masks
        feature_supercell = self.label(supercell_mask, structure)
        # Now we check if we have the same label in any of the adjacent unit
        # cells. If yes we have an infinite feature.
        transformations = np.array(list(itertools.product([0, 1], repeat=3)))
        transformations = self.frac_to_grid(transformations)
        initial_coord = np.argwhere(volume_mask)[0]
        transformed_coords = (transformations + initial_coord).astype(int)

        # Get the feature label at each transformation. If the atom is not surrounded
        # by this basin, at least some of these feature labels will be the same
        features = feature_supercell[
            transformed_coords[:, 0], transformed_coords[:, 1], transformed_coords[:, 2]
        ]

        inf_feature = False
        # If any of the transformed coords have the same feature value, this
        # feature extends between unit cells in at least 1 direction and is
        # infinite. This corresponds to the list of unique features being below
        # 8
        if len(np.unique(features)) < 8:
            inf_feature = True

        return inf_feature

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
        )
        spin_down_grid = Grid(
            structure=self.structure.copy(),
            data=spin_down_data,
            data_aug=aug_down_data,
            data_type=self.data_type,
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
        new.data = data
        new.data_aug = {}
        return new

    # @staticmethod
    # def periodic_center_of_mass(
    #     labels: NDArray[int], label_vals: NDArray[int] = None
    # ) -> NDArray:
    #     """
    #     Computes center of mass for each label in a 3D periodic array.

    #     Parameters
    #     ----------
    #     labels : NDArray[int]
    #         3D array of integer labels.
    #     label_vals : NDArray[int], optional
    #         list/array of unique labels to compute. None will return all.

    #     Returns
    #     -------
    #     NDArray
    #         A 3xN array of centers of mass in voxel index coordinates.
    #     """

    #     shape = labels.shape
    #     if label_vals is None:
    #         label_vals = np.unique(labels)
    #         label_vals = label_vals[label_vals != 0]

    #     centers = []
    #     for val in label_vals:
    #         # get the voxel coords for each voxel in this label
    #         coords = np.array(np.where(labels == val)).T  # shape (N, 3)
    #         # If we have no coords for this label, we skip
    #         if coords.shape[0] == 0:
    #             continue

    #         # From chap-gpt: Get center of mass using spherical distance
    #         center = []
    #         for i, size in enumerate(shape):
    #             angles = coords[:, i] * 2 * np.pi / size
    #             x = np.cos(angles).mean()
    #             y = np.sin(angles).mean()
    #             mean_angle = np.arctan2(y, x)
    #             mean_pos = (mean_angle % (2 * np.pi)) * size / (2 * np.pi)
    #             center.append(mean_pos)
    #         centers.append(center)
    #     centers = np.array(centers)
    #     centers = centers.round(6)

    #     return centers

    # The following method finds critical points using the gradient. However, this
    # assumes an orthogonal unit cell and should be improved.
    # @staticmethod
    # def get_critical_points(
    #     array: NDArray, threshold: float = 5e-03, return_hessian_s: bool = True
    # ) -> tuple[NDArray, NDArray, NDArray]:
    #     """
    #     Finds the critical points in the grid. If return_hessians is true,
    #     the hessian matrices for each critical point will be returned along
    #     with their type index.
    #     NOTE: This method is VERY dependent on grid resolution and the provided
    #     threshold.

    #     Parameters
    #     ----------
    #     array : NDArray
    #         The array to find critical points in.
    #     threshold : float, optional
    #         The threshold below which the hessian will be considered 0.
    #         The default is 5e-03.
    #     return_hessian_s : bool, optional
    #         Whether or not to return the hessian signs. The default is True.

    #     Returns
    #     -------
    #     tuple[NDArray, NDArray, NDArray]
    #         The critical points and values.

    #     """

    #     # get gradient using a padded grid to handle periodicity
    #     padding = 2
    #     # a = np.linspace(
    #     #     0,
    #     #     array.shape[0] + (padding - 1) * 2 + 1,
    #     #     array.shape[0] + padding * 2,
    #     # )
    #     # b = np.linspace(
    #     #     0,
    #     #     array.shape[1] + (padding - 1) * 2 + 1,
    #     #     array.shape[1] + padding * 2,
    #     # )
    #     # c = np.linspace(
    #     #     0,
    #     #     array.shape[2] + (padding - 1) * 2 + 1,
    #     #     array.shape[2] + padding * 2,
    #     # )
    #     padded_array = np.pad(array, padding, mode="wrap")
    #     dx, dy, dz = np.gradient(padded_array)

    #     # get magnitude of the gradient
    #     magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

    #     # unpad the magnitude
    #     slicer = tuple(slice(padding, -padding) for _ in range(3))
    #     magnitude = magnitude[slicer]

    #     # now we want to get where the magnitude is close to 0. To do this, we
    #     # will create a mask where the magnitude is below a threshold. We will
    #     # then label the regions where this is true using scipy, then combine
    #     # the regions into one
    #     magnitude_mask = magnitude < threshold
    #     # critical_points = np.where(magnitude<threshold)
    #     # padded_critical_points = np.array(critical_points).T + padding

    #     label_structure = np.ones((3, 3, 3), dtype=int)
    #     labeled_magnitude_mask = Grid.label(magnitude_mask, label_structure)
    #     min_indices = []
    #     for idx in np.unique(labeled_magnitude_mask):
    #         label_mask = labeled_magnitude_mask == idx
    #         label_indices = np.where(label_mask)
    #         min_mag = magnitude[label_indices].min()
    #         min_indices.append(np.argwhere((magnitude == min_mag) & label_mask)[0])
    #     min_indices = np.array(min_indices)

    #     critical_points = min_indices[:, 0], min_indices[:, 1], min_indices[:, 2]

    #     # critical_points = self.periodic_center_of_mass(labeled_magnitude_mask)
    #     padded_critical_points = tuple([i + padding for i in critical_points])
    #     values = array[critical_points]
    #     # # get the value at each of these critical points
    #     # fn_values = RegularGridInterpolator((a, b, c), padded_array , method="linear")
    #     # values = fn_values(padded_critical_points)

    #     if not return_hessian_s:
    #         return critical_points, values

    #     # now we want to get the hessian eigenvalues around each of these points
    #     # using interpolation. First, we get the second derivatives
    #     d2f_dx2 = np.gradient(dx, axis=0)
    #     d2f_dy2 = np.gradient(dy, axis=1)
    #     d2f_dz2 = np.gradient(dz, axis=2)
    #     # # now create interpolation functions for each
    #     # fn_dx2 = RegularGridInterpolator((a, b, c), d2f_dx2, method="linear")
    #     # fn_dy2 = RegularGridInterpolator((a, b, c), d2f_dy2, method="linear")
    #     # fn_dz2 = RegularGridInterpolator((a, b, c), d2f_dz2, method="linear")
    #     # and calculate the hessian eigenvalues for each point
    #     # H00 = fn_dx2(padded_critical_points)
    #     # H11 = fn_dy2(padded_critical_points)
    #     # H22 = fn_dz2(padded_critical_points)
    #     H00 = d2f_dx2[padded_critical_points]
    #     H11 = d2f_dy2[padded_critical_points]
    #     H22 = d2f_dz2[padded_critical_points]
    #     # summarize the hessian eigenvalues by getting the sum of their signs
    #     hessian_eigs = np.array([H00, H11, H22])
    #     hessian_eigs = np.moveaxis(hessian_eigs, 1, 0)
    #     hessian_eigs_signs = np.where(hessian_eigs > 0, 1, hessian_eigs)
    #     hessian_eigs_signs = np.where(hessian_eigs < 0, -1, hessian_eigs_signs)
    #     # Now we get the sum of signs for each set of hessian eigenvalues
    #     s = np.sum(hessian_eigs_signs, axis=1)

    #     return critical_points, values, s

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
    @staticmethod
    def _guess_file_format(
        filename: str,
        data: NDArray[np.float64],
    ):
        # guess from filename
        data_type = None
        if "elf" in filename.lower():
            data_type = DataType.elf
        elif any(i in filename.lower() for i in ["chg", "charge"]):
            data_type = DataType.charge
        if data_type is not None:
            logging.info(f"Data type set as {data_type.value} from file name")
        return data_type

    @classmethod
    def from_vasp(
        cls,
        grid_file: str | Path,
        data_type: str | DataType = None,
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
            None, the type will be guessed from the filename then the data range.
            Defaults to None.

        Returns
        -------
        Self
            Grid from the specified file.

        """
        logging.info(f"Loading {grid_file} from file")
        # get structure and data from file
        grid_file = Path(grid_file)
        structure, data, data_aug = read_vasp(grid_file)
        # guess data type
        if data_type is None:
            data_type = cls._guess_file_format(grid_file.name, data["total"])

        return cls(
            structure=structure,
            data=data,
            data_aug=data_aug,
            data_type=data_type,
            source_format=Format.vasp,
        )

    @classmethod
    def from_cube(
        cls,
        grid_file: str | Path,
        data_type: str | DataType = None,
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
            None, the type will be guessed from the filename then the data range.
            Defaults to None.

        Returns
        -------
        Self
            Grid from the specified file.

        """
        logging.info(f"Loading {grid_file} from file")
        structure, data, ion_charges, origin = read_cube(grid_file)
        # TODO: Also save the ion charges/origin for writing later

        # guess data type
        if data_type is None:
            data_type = cls._guess_file_format(grid_file.name, data["total"])
        return cls(
            structure=structure,
            data=data,
            data_type=data_type,
            source_format=Format.cube,
        )

    @classmethod
    def from_vasp_pymatgen(
        cls,
        grid_file: str | Path,
        data_type: str | DataType = None,
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
            None, the type will be guessed from the filename then the data range.
            Defaults to None.

        Returns
        -------
        Self
            Grid from the specified file.

        """
        logging.info(f"Loading {grid_file} from file")
        # Create string to add structure to.
        poscar, data, data_aug = cls.parse_file(grid_file)
        # guess data type
        if data_type is None:
            data_type = cls._guess_file_format(grid_file.name, data["total"])

        return cls(structure=poscar.structure, data=data, data_aug=data_aug)

    @classmethod
    def from_dynamic(
        cls,
        grid_file: str | Path,
        format: str | Format = None,
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
        if format is None:
            # guess format from file
            format = detect_format(grid_file)

        if format == Format.cube:
            return cls.from_cube(grid_file)
        elif format == Format.vasp:
            return cls.from_vasp(grid_file)
        else:
            raise ValueError(
                "Provided format '{format}'. Options are: {[i.value for i in Format]}"
            )

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
