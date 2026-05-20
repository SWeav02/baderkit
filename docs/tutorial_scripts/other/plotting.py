# -*- coding: utf-8 -*-

from baderkit.elf_analysis import Badelf

# create badelf instance
badelf = Badelf.from_vasp()

# create badelf plot
plotter = badelf.to_plotter()

# use physical rendering
plotter.pbr = True
plotter.light_intensity = 1.0
plotter.atom_metallicness = 0.25

# show electride basin and hide electride dummy atom
plotter.visible_atom_basins = [3]
plotter.atom_opacity = [1, 1, 1, 0]


# Make isosurface solid
plotter.use_solid_cap_color = True
plotter.use_solid_surface_color = True
plotter.cap_opacity = 1.0
plotter.surface_opacity = 1.0

# set camera angle
plotter.set_camera_to_hkl(1, 1, 0)

# export image
plotter.get_plot_screenshot(filename="Ca2N_electride.png", transparent_background=True)
