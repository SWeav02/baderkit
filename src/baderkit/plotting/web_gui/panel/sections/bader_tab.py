# -*- coding: utf-8 -*-
"""
Defines the widget and layout for the bader tab
"""
import panel as pn

from baderkit.plotting import BaderPlotter


def get_bader_widgets(plotter: BaderPlotter, pane):

    # Get initial results table
    init_bader_results = plotter.bader.get_basin_results_dataframe().copy()
    init_atom_results = plotter.bader.get_atom_results_dataframe().copy()
    # Get dataframe for atoms
    visible_atom_basins_df = pn.widgets.Tabulator(
        init_atom_results,
        show_index=False,
        selectable="checkbox",
        disabled=True,
        theme="modern",
        hidden_columns=["x", "y", "z"],
    )

    visible_bader_basins_df = pn.widgets.Tabulator(
        init_bader_results,
        selectable="checkbox",
        disabled=True,
        theme="modern",
        hidden_columns=["x", "y", "z"],
    )

    # Define function for hiding atoms
    def visible_atom_basins(*events):
        for event in events:
            if event.name == "selection":
                selection = event.new
                # get atoms not in selection
                plotter.visible_atom_basins = selection
                # update basin dataframe
                pane.synchronize()
                # plotter.rebuild()
                # # plotter.camera_position = camera_position
                # pane.object = plotter.plotter.ren_win

    # Define function for hiding basins
    def visible_bader_basins(*events):
        for event in events:
            if event.name == "selection":
                selection = event.new
                # get basins not in selection
                plotter.visible_bader_basins = selection
                pane.synchronize()
                # plotter.rebuild()
                # # plotter.camera_position = camera_position
                # pane.object = plotter.plotter.ren_win

    # link functions
    visible_atom_basins_df.param.watch(visible_atom_basins, "selection")
    visible_bader_basins_df.param.watch(visible_bader_basins, "selection")

    # create dict of items that can be automatically updated
    bader_list = []
    bader_column = pn.WidgetBox(
        visible_atom_basins_df,
        visible_bader_basins_df,
    )
    return bader_list, bader_column
