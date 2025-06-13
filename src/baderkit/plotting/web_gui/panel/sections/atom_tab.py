# -*- coding: utf-8 -*-
"""
Defines the widgets and layout for the Atom's tab
"""
import pandas as pd
import panel as pn

from baderkit.plotting import BaderPlotter


def get_atom_widgets(plotter: BaderPlotter, pane):
    # TODO: Make a widget grid for each atom type and place in drop down
    # widgets. Add ability to change color of each atom type all at once.
    # get the initial atoms df
    init_atom_df = plotter.atom_df
    rows = [
        (
            pn.pane.Markdown(
                "Label",
                align=("center", "end"),
            ),
            pn.pane.Markdown(
                "Visible",
                align=("center", "end"),
            ),
            pn.pane.Markdown(
                "Color", align=("center", "end"), sizing_mode="stretch_width"
            ),
            pn.pane.Markdown(
                "Radius", align=("center", "end"), sizing_mode="stretch_width"
            ),
        )
    ]
    # Create widgets for each row
    for i, row in init_atom_df.iterrows():
        label = pn.pane.Markdown(f'{row["Label"]}', align=("center", "end"))
        visible = pn.widgets.Checkbox(
            value=row["Visible"],
            align=("center", "end"),
        )
        color = pn.widgets.ColorPicker(
            value=row["Color"], align=("center", "end"), sizing_mode="stretch_width"
        )
        radius = pn.widgets.FloatInput(
            value=row["Radius"],
            step=0.01,
            start=0.01,
            align=("center", "end"),
            sizing_mode="stretch_width",
        )
        rows.append((label, visible, color, radius))
    # Build widget grid
    atoms_column = pn.WidgetBox()
    for row in rows:
        row = pn.Row(*row, height=50)
        atoms_column.append(pn.Row(*row))

    def update_df(*events):
        # it shouldn't matter what the even it, we are going to pool all the
        # values anyways
        data = {
            "Visible": [chk.value for _, chk, _, _ in rows[1:]],
            "Color": [clr.value for _, _, clr, _ in rows[1:]],
            "Radius": [flt.value for _, _, _, flt in rows[1:]],
        }
        df = pd.DataFrame(data)
        plotter.atom_df = df
        pane.synchronize()
        # camera_position = plotter.camera_position
        # plotter.rebuild()
        # plotter.camera_position = camera_position
        # pane.object = plotter.plotter.ren_win

    # get all of the widgets and sync them to th update
    all_widgets = [w for row in rows[1:] for w in row[1:]]
    for widget in all_widgets:
        widget.param.watch(update_df, "value")

    # Metallicness doesn't seem to work in panel. Or I'm doing something wrong
    # somewhere
    # atom_metallicness = pn.widgets.FloatSlider(
    #     name='Atom Metallicness',
    #     start=0.0,
    #     end=1.0,
    #     step=0.01,
    #     value=0.0,
    #     value_throttled=True,
    #     )

    # Create dict of widgets that can be automatically updated
    atoms_list = []

    return atoms_list, atoms_column
