# -*- coding: utf-8 -*-
"""
Defines the widget and layout for the grid tab
"""
import matplotlib.pyplot as plt
import panel as pn
from bokeh.models.formatters import PrintfTickFormatter

from baderkit.plotting import BaderPlotter


def get_grid_widgets(plotter: BaderPlotter):
    # General Selection
    iso_val = pn.widgets.EditableFloatSlider(
        name="Iso Value",
        tags=["iso_val"],
        start=plotter.min_val,
        end=plotter.max_val,
        step=round(plotter.max_val - plotter.min_val) / 1000,
        value=plotter.iso_val,
        value_throttled=float(plotter.iso_val),
        format=PrintfTickFormatter(format="%.2f"),
    )
    colormap = pn.widgets.Select(
        name="Colormap", tags=["colormap"], value="viridis", options=plt.colormaps()
    )

    # Surface selection
    show_surface = pn.widgets.Checkbox(
        name="Show Surface",
        tags=["show_surface"],
        value=True,
    )
    use_solid_surface_color = pn.widgets.Checkbox(
        name="Solid Color",
        tags=["use_solid_surface_color"],
        value=False,
        align="center",
    )
    surface_color = pn.widgets.ColorPicker(
        value=plotter.surface_color,
        tags=["surface_color"],
        align="center",
    )
    surface_color_row = pn.Row(use_solid_surface_color, surface_color)
    surface_opacity = pn.widgets.EditableFloatSlider(
        name="Surface Opacity",
        tags=["surface_opacity"],
        start=0.0,
        end=1.0,
        step=0.01,
        value=plotter.surface_opacity,
        format=PrintfTickFormatter(format="%.2f"),
    )

    # Cap selection
    show_caps = pn.widgets.Checkbox(
        name="Show Caps",
        tags=["show_caps"],
        value=True,
    )
    use_solid_cap_color = pn.widgets.Checkbox(
        name="Solid Color",
        tags=["use_solid_cap_color"],
        value=False,
        align="center",
    )
    cap_color = pn.widgets.ColorPicker(
        value=plotter.cap_color,
        tags=["cap_color"],
        align="center",
    )
    cap_color_row = pn.Row(use_solid_cap_color, cap_color)
    cap_opacity = pn.widgets.EditableFloatSlider(
        name="Cap Opacity",
        tags=["cap_opacity"],
        start=0.0,
        end=1.0,
        step=0.01,
        value=plotter.surface_opacity,
        format=PrintfTickFormatter(format="%.2f"),
    )

    # create dict of widgets that can be automatically mapped
    widgets_list = [
        iso_val,
        colormap,
        use_solid_surface_color,
        show_surface,
        surface_opacity,
        surface_color,
        use_solid_cap_color,
        show_caps,
        cap_opacity,
        cap_color,
    ]  # create column to show in the tab
    grid_column = pn.WidgetBox(
        iso_val,
        colormap,
        pn.layout.Divider(),
        show_surface,
        surface_color_row,
        surface_opacity,
        pn.layout.Divider(),
        show_caps,
        cap_color_row,
        cap_opacity,
        sizing_mode="stretch_width",
    )

    return widgets_list, grid_column
