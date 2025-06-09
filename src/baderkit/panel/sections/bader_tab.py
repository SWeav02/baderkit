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
    hidden_atom_basins_df = pn.widgets.Tabulator(
        init_atom_results,
        show_index=False,
        selectable="checkbox",
        disabled=True,
        theme='modern',
        hidden_columns = ["x", "y", "z"]
        )
    
    hidden_bader_basins_df = pn.widgets.Tabulator(
        init_bader_results,
        selectable="checkbox",
        disabled=True,
        theme='modern',
        hidden_columns = ["x", "y", "z"],
        )
    
    # Define function for hiding atoms
    def hidden_atom_basins(*events):
        for event in events:
            if event.name == "selection":
                selection = event.new
                # get atoms not in selection
                inverse_selection = [i for i in range(len(plotter.structure)) if i not in selection]
                plotter.hidden_atom_basins = inverse_selection
                # update basin dataframe
                hidden_bader_basins_df.selection = [i for i in range(len(plotter.bader.basin_atoms)) if i not in plotter.hidden_bader_basins]
                pane.synchronize()
    # Define function for hiding basins
    def hidden_bader_basins(*events):
        for event in events:
            if event.name == "selection":
                selection = event.new
                # get basins not in selection
                inverse_selection = [i for i in range(len(plotter.bader.basin_atoms)) if i not in selection]
                plotter.hidden_bader_basins = inverse_selection
                # update basin dataframe
                hidden_atom_basins_df.selection = [i for i in range(len(plotter.structure)) if i not in plotter.hidden_atom_basins]
                pane.synchronize()
    # link functions
    hidden_atom_basins_df.param.watch(hidden_atom_basins, 'selection')
    hidden_bader_basins_df.param.watch(hidden_bader_basins, 'selection')
    
    # create dict of items that can be automatically updated
    bader_dict = {}
    bader_column = pn.WidgetBox(
        hidden_atom_basins_df,
        hidden_bader_basins_df,
        )
    return bader_dict, bader_column
