# -*- coding: utf-8 -*-

import io
import multiprocessing as mp
import sys
from itertools import product
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from pyvistaqt import QtInteractor, BackgroundPlotter

from baderkit.core import Structure

# BUG-FIX We use multiprocessing to export html because on linux/mac an error
# will throw if this is not done as a main process. We also force fork as our
# start method to avoid pickling issues.
if sys.platform != "win32":
    mp.set_start_method("fork", force=True)


def _export_html(queue: Queue, plotter: pv.Plotter):
    queue.put(plotter.export_html(filename=None))


class VtkPlotter:

    def __init__(
        self,
        structure: Structure,
        off_screen: bool = False,
        qt_plotter: bool = False,
        qt_frame=None,
        show_lattice=True,
        lattice_thickness: bool = 0.1,
        background_color: str = "#FFFFFF",
        view_hkl: list[int] = [1, 0, 0],
        show_axes = True,
        parallel_projection = True,
        pbr = False,
        **kwargs,
    ):
        """
        The base class that all other plotter classes inherit from

        Parameters
        ----------
        off_screen : bool, optional
            Whether or not the plotter should be in offline mode. The default is False.
        qt_plotter : bool, optional
            Whether or not the plotter will use pyvistaqt for qt applications
        qt_frame
            If using pyvistaqt, the QFrame to link the plotter to.

        """
        # sort and relabel structure for consistency
        structure.relabel_sites()
        # create initial class variables
        self.structure = structure
        self.off_screen = off_screen
        self.qt_plotter = qt_plotter
        self.qt_frame = qt_frame
        self._show_lattice = show_lattice

        self._lattice_thickness = lattice_thickness
        self._background_color = background_color
        self._show_axes = show_axes
        self._parallel_projection = parallel_projection
        self._pbr = pbr

        # generate initial plotter
        self._suppressing = False
        self.plotter = None
        self.soft_rebuild()
        h,k,l=view_hkl
        self.set_camera_to_hkl(h, k, l)


    def __setattr__(self, name, value):
        
        # get plotter object if it exists and suppress rendering
        plotter = getattr(self, "plotter", None)
        suppressing = getattr(self, "_suppressing", False)
        if plotter is not None and not suppressing:
            plotter.suppress_rendering = True
            
        super().__setattr__(name, value)
        
        if plotter is not None and not suppressing:
            plotter.suppress_rendering = False

    ###########################################################################
    # Properties and Setters
    ###########################################################################
    
    @property
    def show_lattice(self) -> bool:
        """

        Returns
        -------
        bool
            Whether or not to display the outline of the unit cell.

        """
        return self._show_lattice

    @show_lattice.setter
    def show_lattice(self, show_lattice: bool):
        actor = self.plotter.actors["lattice"]
        actor.visibility = show_lattice
        self._show_lattice = show_lattice

    @property
    def lattice_thickness(self) -> float:
        """

        Returns
        -------
        float
            The thickness of the lines outlining the unit cell.

        """
        return self._lattice_thickness

    @lattice_thickness.setter
    def lattice_thickness(self, lattice_thickness: float):
        actor = self.plotter.actors["lattice"]
        actor.prop.line_width = lattice_thickness
        self._lattice_thickness = lattice_thickness

    @property
    def background_color(self) -> str:
        """

        Returns
        -------
        str
            The color of the plot background as a hex code, rgb array, or color
            string.

        """
        return self._background_color

    @background_color.setter
    def background_color(self, background_color: str):
        self.plotter.set_background(background_color)
        self._background_color = background_color

    @property
    def show_axes(self) -> bool:
        """

        Returns
        -------
        bool
            Whether or not to show the axis widget.

        """
        return self._show_axes

    @show_axes.setter
    def show_axes(self, show_axes: bool):
        if not show_axes:
            self.plotter.hide_axes()
        else:
            self.plotter.show_axes_all()
        self._show_axes = show_axes

    @property
    def parallel_projection(self) -> bool:
        """

        Returns
        -------
        bool
            If True, a parallel projection scheme will be used rather than
            perspective.

        """
        return self._parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, parallel_projection: bool):
        if parallel_projection:
            self.plotter.renderer.enable_parallel_projection()
        else:
            self.plotter.renderer.disable_parallel_projection()
        self._parallel_projection = parallel_projection
        
    @property
    def pbr(self) -> bool:
        """

        Returns
        -------
        bool
            If True, physically based rendering will be used

        """
        return self._pbr

    @pbr.setter
    def pbr(self, pbr: bool):
        self._pbr = pbr
        # pbr is set when adding a mesh, so we need to rebuild
        self.soft_rebuild()

    def set_camera_to_hkl(self, h, k, l, rotation=0.0):
        normal = self.structure.get_cart_from_miller(h, k, l)
        origin = np.mean(self.structure.cart_coords, axis=0)
        
        # Define distance from object
        distance = 5
        camera_pos = origin + normal * distance
        
        # get an appropriate view up
        z_axis = np.array([0, 0, 1])
        view_up = z_axis - np.dot(z_axis, normal) * normal
        norm_proj_z = np.linalg.norm(view_up)
        if norm_proj_z < 1e-14:
            # fallback to y-axis if view direction is exactly perpendicular to
            # the z direction
            y_axis = np.array([0, 1, 0])
            view_up = y_axis - np.dot(y_axis, normal) * normal

        self.plotter.camera_position = [camera_pos, origin, view_up] 
        # reset camera to fit well
        self.plotter.reset_camera()
        
    def set_camera_to_vector(self, normal, origin=None, rotation=0.0):
        
        if origin is None:
            origin = np.mean(self.structure.cart_coords, axis=0)
        # Define distance from object
        distance = 5
        camera_pos = origin + normal * distance
        
        # get an appropriate view up
        z_axis = np.array([0, 0, 1])
        view_up = z_axis - np.dot(z_axis, normal) * normal
        norm_proj_z = np.linalg.norm(view_up)
        if norm_proj_z < 1e-14:
            # fallback to y-axis if view direction is exactly perpendicular to
            # the z direction
            y_axis = np.array([0, 1, 0])
            view_up = y_axis - np.dot(y_axis, normal) * normal

        self.plotter.camera_position = [camera_pos, origin, view_up] 
        # reset camera to fit well
        self.plotter.reset_camera()
    
    ###########################################################################
    # Mesh Generation
    ###########################################################################
        
    @staticmethod
    def _wrap_near_edge(frac_coord: NDArray, tol: float = 0.01) -> NDArray:
        """
        Generates translationally equivalent atoms if coords are exactly on an edge
        of the lattice

        Parameters
        ----------
        frac_coord : NDArray
            The fractiona coordinates of a single atom to wrap.
        tol : float, optional
            The tolerance in fractional coords to consider an atom on an edge
            of the unit cell. The default is 1e-08.

        Returns
        -------
        NDArray
            The fractional coordinates of the atom wrapped at edges.

        """
        transforms = [
            [0, 1] if abs(x) < tol else [0, -1] if abs(x - 1) < tol else [0]
            for x in frac_coord
        ]

        shifts = set(product(*transforms))
        return [np.array(frac_coord) + np.array(shift) for shift in shifts], shifts

    @staticmethod
    def _wrap_group_near_edge_shifts(frac_coords, tol=0.01):
        transforms = []
    
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    shift = np.array((i, j, k), dtype=float)
                    try:
                        coords = frac_coords + shift
                    except:
                        breakpoint()
    
                    # point-wise interior test
                    interior = np.all(
                        (coords > tol) & (coords < 1 - tol),
                        axis=1
                    )
    
                    if np.any(interior):
                        transforms.append(shift)
    
        return np.array(transforms)

    def _add_lattice_mesh(self) -> pv.PolyData:
        """
        Generates the mesh representing the outline of the unit cell.

        Returns
        -------
        pv.PolyData
            A pyvista mesh representing the outline of the unit cell.

        """
        # get the lattice matrix
        a, b, c = self.structure.lattice.matrix
        # get the corners of the matrix
        corners = [np.array([0, 0, 0]), a, b, c, a + b, a + c, b + c, a + b + c]
        # get the indices indicating edges of the lattice
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]
        # generate lines with pv
        lines = []
        for i, j in edges:
            line = pv.Line(corners[i], corners[j])
            lines.append(line)
        # combine and return
        lattice_mesh = pv.merge(lines)
        self.plotter.add_mesh(
            lattice_mesh,
            line_width=self.lattice_thickness,
            color="k",
            name="lattice",
        )
        
    def _add_axes_widget(self, plotter, show_axes: bool):
        # lattice[0] = a-vector, lattice[1] = b-vector, lattice[2] = c-vector
        lattice = self.structure.lattice.matrix 
    
        # 2. Construct the 4x4 transformation matrix
        # The columns of the upper-left 3x3 represent where the unit X, Y, Z go.
        # We transpose the lattice matrix to put vectors in columns.
        matrix = np.eye(4)
        matrix[:3, :3] = lattice.T 
        
        scale = [1/np.linalg.norm(a) for a in lattice]
    
        # 3. Create the AxesAssembly
        # We pass our calculated matrix to 'user_matrix'
        axes = pv.AxesAssembly(
            user_matrix=matrix,
            scale=scale,
            labels=["a", "b", "c"],
            # label_color="black",
            label_size=24,
        )

        plotter.add_orientation_widget(
            axes,
            viewport=(0.0, 0.0, 0.25, 0.25),
            interactive=False,
        )
        plotter.show_axes = self.show_axes

    def _create_plotter(self) -> pv.Plotter:
        if self.plotter is not None:
            return self.plotter
        if self.qt_plotter:
            assert self.qt_frame is not None, "A frame must be set to use qt"
            plotter = QtInteractor(self.qt_frame)
        elif self.off_screen:
            plotter = pv.Plotter(off_screen=True)
        else:
            plotter = BackgroundPlotter()
        self.plotter = plotter
        return plotter

    def _create_plot(self) -> pv.Plotter:
        """
        Generates a pyvista.Plotter object from the current class variables.
        This is called when the class is first instanced and generally shouldn't
        be called again.

        Returns
        -------
        plotter : pv.Plotter
            A pyvista Plotter object with the base lattice required for viewing

        """
        # if we already have a plotter, just return it
        plotter = self._create_plotter()
        
        # set background
        plotter.set_background(self.background_color)

        # add lattice
        self._add_lattice_mesh()
        self.show_lattice = self.show_lattice
        
        # create axes widget
        self._add_axes_widget(plotter, self.show_axes)
        self.show_axes = self.show_axes

        # set camera perspective type
        if self.parallel_projection:
            plotter.renderer.enable_parallel_projection()

        return plotter

    ###########################################################################
    # Helper Functions
    ###########################################################################

    def show(self):
        """
        Renders the plot to a window. After closing the window, a new instance
        must be created to plot again. Pressing q pauses the rendering allowing
        changes to be made without fully exiting.

        Returns
        -------
        None.

        """
        
        self.plotter.show()

    def rebuild(self) -> pv.Plotter:
        """
        Builds a new pyvista plotter object representing the current state of
        the Plotter class.

        Returns
        -------
        pv.Plotter
            A pyvista Plotter object representing the current state of the
            StructurePlotter class.

        """
        self.plotter = None
        plotter = self._create_plotter()
        plotter.suppress_rendering=True
        self._suppressing = True
        self._create_plot()
        plotter.suppress_rendering=False
        self._suppressing = True
        
    def soft_rebuild(self) -> pv.Plotter:
        """
        reuilds the current pyvista plotter object with current settings.

        Returns
        -------
        pv.Plotter
            A pyvista Plotter object representing the current state of the
            StructurePlotter class.

        """
        plotter = self.plotter
        if plotter is None:
            plotter = self._create_plotter()
        plotter.suppress_rendering=True
        self._suppressing = True
        self._create_plot()
        plotter.suppress_rendering=False
        self._suppressing = True
        

    def get_plot_html(self) -> str:
        """
        Creates an html string representing the current state of the StructurePlotter
        class.

        Returns
        -------
        str
            The html string representing the current StructurePlotter class.

        """
        if sys.platform == "win32":
            # We can return the html directly without opening a subprocess. And
            # we need to because the "fork" start method doesn't work
            html_plotter = self.plotter.export_html(filename=None)
            return html_plotter.read()
        # BUG-FIX: On Linux and maybe MacOS, pyvista's export_html must be run
        # as a main process. To do this within our streamlit apps, we use python's
        # multiprocess to run the process as is done in [stpyvista](https://github.com/edsaac/stpyvista/blob/main/src/stpyvista/trame_backend.py)
        queue = Queue(maxsize=1)
        process = Process(target=_export_html, args=(queue, self.plotter))
        process.start()
        html_plotter = queue.get().read()
        process.join()
        return html_plotter

    def get_plot_screenshot(
        self,
        filename: str | Path | io.BytesIO = None,
        transparent_background: bool = None,
        return_img: bool = True,
        window_size: tuple[int, int] = None,
        scale: int = None,
    ) -> NDArray[float]:
        """
        Creates a screenshot of the current state of the StructurePlotter class.
        This is a wraparound of pyvista's screenshot method

        Parameters
        ----------
        filename: str | Path | io.BytesIO
            Location to write image to. If None, no image is written.

        transparent_background: bool
            Whether to make the background transparent.
            The default is looked up on the plotter’s theme.

        return_img: bool
            If True, a numpy.ndarray of the image will be returned. Defaults to
            True.

        window_size: tuple[int, int]
            Set the plotter’s size to this (width, height) before taking the
            screenshot.

        scale: int
            Set the factor to scale the window size to make a higher resolution image. If None this will use the image_scale property on this plotter which defaults to one.

        Returns
        -------
        NDArray[float]
            Array containing pixel RGB and alpha. Sized:

            [Window height x Window width x 3] if transparent_background is set to False.

            [Window height x Window width x 4] if transparent_background is set to True.

        """


        plotter = self.plotter
        
        # if our plotter is not currently rendered, we want to temporarily set
        # it to be off screen to take the screenshot, then set it back

        screenshot = plotter.screenshot(
            filename=filename,
            transparent_background=transparent_background,
            return_img=return_img,
            window_size=window_size,
            scale=scale,
        )

        return screenshot

    def _set_camera_tight(self, padding=0.0, adjust_render_window=True):
        """
        Adjust the camera parallel_scale to fit the actors tightly,
        without changing the camera position, focal point, or view direction.
        """
        camera = self.plotter.camera
        ren = camera._renderer
        x0, x1, y0, y1, z0, z1 = ren.bounds

        # Compute aspect ratio
        ren.ComputeAspect()
        aspect = ren.GetAspect()

        # Bounding box size
        bbox_size = np.array([x1 - x0, y1 - y0, z1 - z0])

        # Use current camera view up and direction
        viewup = np.array(camera.GetViewUp())
        direction = np.array(camera.GetFocalPoint()) - np.array(camera.GetPosition())
        direction /= np.linalg.norm(direction)
        horizontal = np.cross(direction, viewup)

        # Project bounding box onto camera plane axes
        vert_dist = abs(bbox_size @ viewup)
        horiz_dist = abs(bbox_size @ horizontal)

        # Set parallel scale
        ps = max(horiz_dist / aspect[0], vert_dist) / 2
        camera.parallel_scale = ps * (1 + padding)

        # Reset clipping planes
        camera._renderer.ResetCameraClippingRange(x0, x1, y0, y1, z0, z1)

        if adjust_render_window:
            ren_win = ren.GetRenderWindow()
            size = list(ren_win.GetSize())
            size_ratio = size[0] / size[1]
            tight_ratio = horiz_dist / vert_dist
            resize_ratio = tight_ratio / size_ratio
            if resize_ratio < 1:
                size[0] = round(size[0] * resize_ratio)
            else:
                size[1] = round(size[1] / resize_ratio)
            ren_win.SetSize(size)
