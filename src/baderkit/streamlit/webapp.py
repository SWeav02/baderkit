# -*- coding: utf-8 -*-

from io import StringIO

import streamlit as st
import streamlit.components.v1 as components

from baderkit.plotting import GridPlotter
from baderkit.plotting.defaults import ATOM_COLORS
from baderkit.utilities import Grid

###############################################################################
# Setup
###############################################################################
# set page to wide
st.set_page_config(layout="wide")
# define session variables. Note that these must match the plotter's class variable
# names
session_variables = {
    "charge_grid_name": None,
    "charge_grid": None,
    "structure": None,
    "plotter": None,
    "plot_html": None,
    "plot_height": 800,
}
for variable, value in session_variables.items():
    if variable not in st.session_state:
        st.session_state[variable] = value

# create columns
col_settings, col_plot = st.columns([1, 3])

###############################################################################
# Settings Panel
###############################################################################
# create tag for if reset is rerun is required
need_reset = False

with col_settings:
    with st.container(border=True, height=st.session_state.plot_height):
        # create tabs
        tabs = st.tabs(["Upload", "Atoms", "Grid", "View"])
        tab_upload, tab_atoms, tab_grid, tab_view = tabs
        #######################################################################
        # Upload
        #######################################################################
        with tab_upload:
            grid_file = st.file_uploader("Upload Charge Grid")

            if grid_file is not None:
                if (
                    st.session_state.charge_grid is None
                    or st.session_state.charge_grid_name != grid_file.name
                ):
                    # update charge grid name
                    st.session_state.charge_grid_name = grid_file.name
                    # get string and load grid
                    stringio = StringIO(grid_file.getvalue().decode("utf-8"))
                    grid = Grid.from_vasp_string(stringio.read())
                    # update grid and plotter
                    st.session_state.charge_grid = grid
                    st.session_state.structure = grid.structure
                    st.session_state.plotter = GridPlotter(grid)
            else:
                for tab in tabs[1:]:
                    tab.markdown("Upload Grid to Start")
                st.stop()

        # create dict for saving settings. Note these must have the same keys as
        # in the session state
        settings = st.session_state.get("settings", {})

        #######################################################################
        # Atoms tab
        #######################################################################
        with tab_atoms:
            # TODO: Find way to remove gaps. Probably with CSS
            # TODO: Add options for selection for full atom group
            # Get current settings
            colors = settings.get("colors", st.session_state.plotter.colors)
            hidden_atoms = settings.get("hidden_atoms", [])
            radii = settings.get("radii", st.session_state.plotter.radii)
            
            # create lists for new radii
            new_colors = []
            new_hidden_atoms = []
            new_radii = []
            # Create column headers
            label_col, color_col, active_col, radius_col = st.columns(
                [1, 1, 1, 1], vertical_alignment="bottom"
            )
            label_col.markdown("*Label*")
            color_col.markdown("*Color*")
            active_col.markdown("*Show*")
            radius_col.markdown("*Radius*")
            for i, site in enumerate(st.session_state.structure):
                # create columns for this row
                label_col, color_col, active_col, radius_col = st.columns(
                    [1, 1, 1, 1], vertical_alignment="bottom"
                )
                # Add label
                label_col.markdown(site.label)
                # add color picker. Use current color in settings as default
                color = color_col.color_picker(
                    label=site.label,
                    value=colors[i],
                    label_visibility="hidden",
                )
                new_colors.append(color)
                # add show atom check
                active = active_col.checkbox(
                    site.label,
                    value=False if i in hidden_atoms else True,
                    label_visibility="hidden",
                )
                if not active:
                    new_hidden_atoms.append(i)
                # add radius selection
                radius = radius_col.number_input(
                    site.label,
                    value=radii[i],
                    min_value=0.01,
                    max_value=10.00,
                    label_visibility="hidden"
                    )
                new_radii.append(radius)
                    
                "---"

            settings["colors"] = new_colors
            settings["hidden_atoms"] = new_hidden_atoms
            settings["radii"] = new_radii
            # reset color button
            if st.button("Reset"):
                # reset colors
                settings["colors"] = [
                    ATOM_COLORS.get(s.specie.symbol, "#FFFFFF")
                    for s in st.session_state.structure
                ]
                # reset hidden atoms
                settings["hidden_atoms"] = []
                # reset radii
                settings["radii"] = [s.specie.atomic_radius for s in st.session_state.structure]
                st.rerun()

        #######################################################################
        # Grid tab
        #######################################################################
        with tab_grid:
            # Surface Settings
            st.markdown("### Surface Settings")
            settings["show_surface"] = st.toggle("Show Surface", True)
            if settings["show_surface"]:
                # isosurface value
                settings["iso_val"] = st.slider(
                    "Isosurface value",
                    st.session_state.plotter.min_val,
                    st.session_state.plotter.max_val,
                    value=st.session_state.plotter.iso_val,
                )
                # isosurface opacity
                settings["surface_opacity"] = st.slider("Surface Opacity", 0.0, 1.0, 1.0)
                # use solid surface color
                settings["use_solid_surface_color"] = st.toggle(
                    "Solid Color", False, key="solid0"
                )
                if settings["use_solid_surface_color"]:
                    # surface color
                    settings["surface_color"] = st.color_picker("Surface Color")

            # Caps settings
            st.markdown("### Cap Settings")
            # show caps
            settings["show_caps"] = st.toggle("Show Caps", True)
            if settings["show_caps"]:
                # cap opacity
                settings["cap_opacity"] = st.slider("Cap Opacity", 0.0, 1.0, 1.0)
                # use solid cap color
                settings["use_solid_cap_color"] = st.toggle(
                    "Solid Color", False, key="solid1"
                )
                if settings["use_solid_cap_color"]:
                    # cap color
                    settings["surface_color"] = st.color_picker("Surface Color")
            
            # Other
            st.markdown("### Other Settings")
            # colormap
            settings["colormap"] = st.text_input(
                "Color Map. Any values from [Matplotlib](https://matplotlib.org/stable/users/explain/colors/colormaps.html) are accepted.",
                value="viridis",
            )
        
        #######################################################################
        # View Tab
        #######################################################################
        with tab_view:
            # add setting for screen size
            height_str = st.selectbox(
                "Window Height", 
                ["extra small", "small", "medium", "large", "extra large"],
                index=3,
                )
            height_map = {
                "extra small": 200,
                "small": 400,
                "medium": 600,
                "large": 800,
                "extra large": 1000
                }
            # setting for background color
            settings["background"] = st.color_picker("Background Color", st.session_state.plotter.background)
            settings["show_lattice"] = st.toggle("Show Lattice", True)
            settings["lattice_thickness"] = st.slider(
                "Lattice Thickness", 0.0, 1.0, 0.3
            )
            # settings["show_axes"] = st.toggle("Show Axes", True)
            # settings["atom_metallicness"] = st.slider("Metallicness",0.0,1.0,0.0)
            settings["wrap_atoms"] = st.toggle("Wrap Atoms", True)
            # settings for view angle
            # settings["parallel_projection"] = st.toggle("Use Parallel Projection", True)
            st.markdown("View Angle (Miller Indices)")
            h_col, k_col, l_col = st.columns(3)
            h = h_col.number_input("h", value=1, label_visibility="hidden")
            k = k_col.number_input("k", value=0, label_visibility="hidden")
            l = l_col.number_input("l", value=0, label_visibility="hidden")
            settings["view_indices"] = [h,k,l]
            
    st.session_state.settings = settings
    # create apply button
    if st.button("Apply"):
        # update view
        st.session_state.plot_height = height_map[height_str]
        # check if any plotter values are different and if so update them
        for key, value in settings.items():
            plotter_val = getattr(st.session_state.plotter, key, None)
            if value != plotter_val:
                # update plotter
                setattr(st.session_state.plotter, key, value)
                # note we need reset
                need_reset = True

###############################################################################
# Display plot
###############################################################################
with col_plot:
    if st.session_state.plot_html is None or need_reset:
        st.session_state.plot_html = (
            st.session_state.plotter.get_grid_plot_html(height=st.session_state.plot_height, width=st.session_state.plot_height)
        )
        # reset
        st.rerun()
    components.html(st.session_state.plot_html, height=st.session_state.plot_height)
