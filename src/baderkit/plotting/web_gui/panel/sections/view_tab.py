# -*- coding: utf-8 -*-
"""
Defines the widget and layout for the view tab
"""
import pandas as pd
import panel as pn
from bokeh.models.formatters import PrintfTickFormatter

from baderkit.plotting import BaderPlotter


def get_view_widgets(plotter: BaderPlotter, pane):
    # Lattice Widgets
    show_lattice = pn.widgets.Checkbox(
        name="Show Lattice",
        tags=["show_lattice"],
        value=True,
    )
    lattice_thickness = pn.widgets.FloatInput(
        value=plotter.lattice_thickness,
        tags=["lattice_thickness"],
        name="Lattice Thickness",
        step=0.01,
        start=0.00,
        align=("center", "end"),
        sizing_mode="stretch_width",
    )
    background = pn.widgets.ColorPicker(
        name="Background",
        tags=["background"],
        value=plotter.background,
    )
    # View Widgets

    # Create a dataframe for input of camera angle in terms of miller indices
    init_view_df = pd.DataFrame({"h": [1], "k": [0], "l": [0]})
    # Create editors forcing step size to be 1
    editors = {
        "h": {"type": "number", "step": 1},
        "k": {"type": "number", "step": 1},
        "l": {"type": "number", "step": 1},
    }
    # Create tabulator widget
    view_df = pn.widgets.Tabulator(
        init_view_df,
        name="View Angle (miller indices)",
        selectable=False,  # don't allow selection
        show_index=False,
        widths=80,  # set column widths
        editors=editors,
        header_align="center",
        theme="modern",
        configuration={
            "columnDefaults": {
                "headerSort": False,
            }
        },
    )
    # Create button to apply view
    apply_view = pn.widgets.Button(name="Apply", align="center")

    def update_view(event):
        if not event:
            return
        # pool view angle from df
        row = view_df.value.iloc[0]
        h, k, l = row["h"], row["k"], row["l"]
        angle = [h, k, l]
        # TODO: This doesn't work despite matching the docs
        plotter.camera_position = angle
        pane.synchronize()

    pn.bind(update_view, apply_view, watch=True)
    view_row = pn.Row(view_df, apply_view)

    widgets_list = [
        show_lattice,
        lattice_thickness,
        background,
    ]
    # create column to show in the tab
    view_column = pn.WidgetBox(
        show_lattice,
        lattice_thickness,
        background,
        pn.layout.Divider(),
        view_row,
        sizing_mode="stretch_width",
    )
    return widgets_list, view_column
