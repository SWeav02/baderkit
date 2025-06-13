# -*- coding: utf-8 -*-

import inspect
import os
import time

import panel as pn
import pyvista as pv

from baderkit.core import Bader
from baderkit.panel.sections import (
    get_atom_widgets,
    get_bader_widgets,
    get_grid_widgets,
    get_view_widgets,
)
from baderkit.plotting import BaderPlotter

# Get bader results first
env = os.environ.copy()
charge_filename = env["CHARGE_FILE"]
method = env["BADER_METHOD"]
if "REFERENCE_FILE" in env.keys():
    reference_filename = env["REFERENCE_FILE"]
else:
    reference_filename = None
bader = Bader.from_vasp(
    charge_filename=charge_filename,
    reference_filename=reference_filename,
    method=method,
)

pn.extension("vtk", "tabulator", design="material")

# Always set PyVista to plot off screen with Panel
pv.OFF_SCREEN = True

# get initial plotter
plotter = BaderPlotter(bader, off_screen=True)
# plotter.plotter.suppress_rendering = True
# Update camera angle
plotter.plotter.camera.tight()
plotter.camera_position = [1, 0, 0]

pane = pn.pane.vtk.vtk.VTKRenderWindowSynchronized(
    plotter.plotter.ren_win,
    sizing_mode="stretch_both",
    enable_keybindings=True,
)


# define an update funciton for general options
plotter_properties = [
    name
    for name, value in inspect.getmembers(BaderPlotter)
    if isinstance(value, property)
]


# IMPORTANT: Use the watch method rather than bind. Returning a new pane
# caused immense lag, likely due to background storage of previous panes.
def update_plotter(*events):
    for event in events:
        property_name = event.obj.tags[0]
        print(property_name)
        if not event.name == "value":
            continue
        if property_name not in plotter_properties:
            continue
        # for key, value in kwargs.items():
        #     if not key in plotter_properties:
        #         continue
        # get current value
        current_val = getattr(plotter, property_name)
        if event.new != current_val:
            setattr(plotter, property_name, event.new)
            # time.sleep(0.01)
            # plotter.plotter.close()
            # camera_position = plotter.camera_position
            # plotter.rebuild()
            # plotter.camera_position = camera_position
            pane.object = plotter.plotter.ren_win
            # pane.synchronize()
            # return pane


# The get widgets functions return the widgets to bind and the column to
# add to the tab as a tuple.
(
    (atom_widgets, atoms_column),
    (bader_widgets, bader_column),
    (grid_widgets, grid_column),
    (view_widgets, view_column),
) = (
    get_atom_widgets(plotter, pane),
    get_bader_widgets(plotter, pane),
    get_grid_widgets(plotter),
    get_view_widgets(plotter, pane),
)
all_widgets = []
# # We don't bind the bader widgets because they are bound separately
for widgets_list in [atom_widgets, grid_widgets, view_widgets]:
    all_widgets.extend(widgets_list)
# bound_plot = pn.bind(update_plotter, **kwargs)
for widget in all_widgets:
    widget.param.watch(update_plotter, ["value"])

Tabs = pn.Tabs(
    ("Basins", bader_column),
    ("Atoms", atoms_column),
    ("Grid", grid_column),
    ("View", view_column),
)

pn.template.MaterialTemplate(
    site="BaderKit",
    title="Plotter",
    sidebar=[Tabs],
    main=[pane],
).servable()
