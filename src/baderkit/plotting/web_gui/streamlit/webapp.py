# -*- coding: utf-8 -*-

import os

import numpy as np
import streamlit as st
from PIL import Image

from baderkit.core import Bader, Grid
from baderkit.plotting import BaderPlotter
from baderkit.plotting.core.defaults import COLORMAPS

st.markdown(
    """
    <style>
        .stAppDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)


# load and run bader into plotter
def load_plotter():
    env = os.environ.copy()
    charge_filename = env["CHARGE_FILE"]
    method = env["BADER_METHOD"]
    vacuum_tol = float(env["VACUUM_TOL"])
    normalize_vac = bool(env["NORMALIZE_VAC"])
    basin_tol = float(env["BASIN_TOL"])
    if "REFERENCE_FILE" in env.keys():
        reference_filename = env["REFERENCE_FILE"]
    else:
        reference_filename = None
    bader = Bader.from_vasp(
        charge_filename=charge_filename,
        reference_filename=reference_filename,
        method=method,
        vacuum_tol=vacuum_tol,
        normalize_vacuum=normalize_vac,
        basin_tol=basin_tol,
    )
    # check how many atom basins there are. We can probably only handle up to
    # 50
    if len(bader.structure) > 50:
        st.session_state["too_many_atoms"] = True
    else:
        st.session_state["too_many_atoms"] = False

    plotter = BaderPlotter(bader, off_screen=True)
    # set plotter camera
    plotter.view_indices = plotter._view_indices
    st.session_state.bader = bader
    st.session_state.plotter = plotter
    st.session_state.html_string = plotter.get_plot_html()
    st.session_state.page_height = 400


if any(k not in st.session_state for k in ("plotter", "bader")):
    load_plotter()

if st.session_state.too_many_atoms:
    st.markdown(
        """
        Too many atoms in the structure. The BaderKit webapp can currently only
        handle up to ~50 atoms. If you still need visualization, basins can be written
        out with the `--print` tag and visualized using tools such as VESTA or OVITO.
        """
    )
    st.stop()

# get bader/plotter for this session so we don't have to pull from session state constantly
bader = st.session_state.bader
plotter = st.session_state.plotter

settings = {}
with st.sidebar:
    with st.container(height=st.session_state.page_height):
        basins_tab, grid_tab, atoms_tab, view_tab, export_tab = st.tabs(
            ["Basins", "Grid", "Atoms", "View", "Export"]
        )
        #######################################################################
        # Basin settings
        #######################################################################
        with basins_tab:
            st.markdown(
                "Select basins to show on the plot. The display will be union of the selection."
            )
            # get selection
            settings["visible_atom_basins"] = st.segmented_control(
                "Atom Basins",
                [i for i in range(len(bader.atom_charges))],
                selection_mode="multi",
                key="vis_atoms",
            )
            # don't display bader selection if there are too many of them.
            if len(bader.basin_charges) <= 40:
                settings["visible_bader_basins"] = st.segmented_control(
                    "Bader Basins",
                    [i for i in range(len(bader.basin_charges))],
                    selection_mode="multi",
                    key="vis_basins",
                )
            else:
                st.markdown("Too many basins were found in the structure to display.")

        #######################################################################
        # Surface and Cap settings
        #######################################################################
        with grid_tab:
            # general settings
            settings["iso_val"] = st.number_input(
                f"Iso Value: {round(plotter.min_val,2)} - {round(plotter.max_val,2)}",
                value=plotter.iso_val,
            )
            settings["colormap"] = st.selectbox(
                "Colormap",
                options=COLORMAPS,
                index=3,  # this is viridis
            )
            st.divider()
            # Surface settings
            col1, col2, _ = st.columns([2, 1, 1], vertical_alignment="center")
            col1.markdown("Show Surface")
            settings["show_surface"] = col2.toggle(
                "Show Surface", value=True, label_visibility="collapsed"
            )
            if settings["show_surface"]:
                col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="center")
                # use solid color
                col1.markdown("Solid Color")
                settings["use_solid_surface_color"] = col2.toggle(
                    "Solid Color", key="surface_solid", label_visibility="collapsed"
                )
                if settings["use_solid_surface_color"]:
                    settings["surface_color"] = col3.color_picker(
                        "Color",
                        value=plotter.surface_color,
                        label_visibility="collapsed",
                        key="surface_color",
                    )
                settings["surface_opacity"] = st.number_input(
                    "Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=plotter.surface_opacity,
                    key="surface_opacity",
                )
            st.divider()
            # Cap settings
            col1, col2, _ = st.columns([2, 1, 1], vertical_alignment="center")
            col1.markdown("Show Caps")
            settings["show_caps"] = col2.toggle(
                "Show Caps", value=True, label_visibility="collapsed"
            )
            if settings["show_caps"]:
                col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="center")
                # use solid color
                col1.markdown("Solid Color")
                settings["use_solid_cap_color"] = col2.toggle(
                    "Solid Color", key="cap_solid", label_visibility="collapsed"
                )
                if settings["use_solid_cap_color"]:
                    settings["cap_color"] = col3.color_picker(
                        "Color",
                        value=plotter.cap_color,
                        label_visibility="collapsed",
                        key="cap_color",
                    )
                settings["cap_opacity"] = st.number_input(
                    "Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=plotter.cap_opacity,
                    key="cap_opacity",
                )
        #######################################################################
        # Atom settings
        #######################################################################
        with atoms_tab:
            # create settings
            settings["visible_atoms"] = []
            settings["colors"] = [None for i in range(len(plotter.structure))]
            settings["radii"] = [None for i in range(len(plotter.structure))]
            # create column headers
            label_col, vis_col, col_col, radii_col = st.columns(
                4, vertical_alignment="bottom"
            )
            label_col.markdown("Species")
            vis_col.markdown("Visible")
            col_col.markdown("Color")
            radii_col.markdown("Radius")

            for species in plotter.structure.symbol_set:
                atom_indices = plotter.structure.indices_from_symbol(species)
                init_color = plotter.colors[atom_indices[0]]
                init_radius = plotter.radii[atom_indices[0]]
                # create widgets
                label_col, vis_col, col_col, radii_col = st.columns(
                    4, vertical_alignment="center"
                )
                label = label_col.markdown(species)
                vis = vis_col.checkbox(
                    species, value=True, label_visibility="collapsed"
                )
                color = col_col.color_picker(
                    species, value=init_color, label_visibility="collapsed"
                )
                radius = radii_col.number_input(
                    species, value=init_radius, label_visibility="collapsed"
                )
                # update settings
                for i in atom_indices:
                    settings["colors"][i] = color
                    settings["radii"][i] = radius
                    if vis:
                        settings["visible_atoms"].append(i)
        #######################################################################
        # View Settings
        #######################################################################
        with view_tab:
            col1, col2, col3 = st.columns([2, 1, 2], vertical_alignment="center")
            col1.markdown("Show Lattice")
            settings["show_lattice"] = col2.toggle(
                "Show Lattice", True, label_visibility="collapsed"
            )
            settings["lattice_thickness"] = col3.number_input(
                "Lattice Thickness",
                value=plotter.lattice_thickness,
                min_value=0.01,
                label_visibility="collapsed",
            )
            settings["background"] = st.color_picker("Background Color", "#FFFFFF")
            st.divider()
            # view setting
            settings["view_indices"] = []
            col1, col2 = st.columns(2)
            st.markdown(
                "The view direction will be the vector perpendicular to the family of lattice planes defined by the miller indices."
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown("h")
            col2.markdown("k")
            col3.markdown("l")
            col4.markdown("rot")
            for i, (x, col) in enumerate(zip(["h", "k", "l"], [col1, col2, col3])):
                settings["view_indices"].append(
                    col.number_input(
                        label=x,
                        value=plotter.view_indices[i],
                        step=1,
                        label_visibility="collapsed",
                    )
                )
            settings["camera_rotation"] = col4.number_input(
                label="rot",
                value=0.0,
                label_visibility="collapsed",
            )

        #######################################################################
        # Export settings
        #######################################################################
        with export_tab:
            st.markdown("### Export Current Basins")
            st.markdown(
                "All currently selected basins will be exported as VASP type files"
            )
            if st.button("Export Basins", icon=":material/download:"):
                atom_labels = bader.atom_labels
                basin_labels = bader.basin_labels
                data_mask = np.isin(atom_labels, plotter.visible_atom_basins) & np.isin(
                    basin_labels, plotter.visible_bader_basins
                )
                total = np.where(data_mask, bader.charge_grid.total, 0)
                temp_grid = Grid(
                    structure=plotter.structure.copy(), data={"total": total}
                )
                temp_grid.write_file("CHGCAR_absum")
            # TODO: I can either make this export directly to the users computer
            # wherever they opened this webapp or try and make it a download.

            st.markdown("### Export as Image")
            st.markdown(
                "Rendered image will match the state of the window when you hit apply. It does not take into account any movements you have made since. Parallel perspective and matallicness can be selected for export, but do not show up in the viewer due to limitations in pyvista's html export."
            )
            metallic = st.number_input(
                "Atom Metallicness", min_value=0.0, max_value=1.0, value=0.0
            )
            col1, col2 = st.columns(2)
            parallel = col1.checkbox("Parallel Perspective", True)
            transparent = col2.checkbox("Transparent Background", False)
            col1, col2, col3 = st.columns(3)
            width = col1.number_input("Width", min_value=1, value=400)
            height = col2.number_input("Height", min_value=1, value=400)
            scale = col3.number_input("Scale", min_value=1, value=1)
            filetype_options = Image.registered_extensions()
            filetype = st.selectbox(
                "File Format",
                options=[i for i in filetype_options.keys()],
                index=0,
            )
            filename = st.text_input(
                "Filename", value=f"{plotter.structure.composition.reduced_formula}"
            )
            # TODO: Find a better way of caching
            if st.button("Export Image", icon=":material/download:"):
                plotter.parallel_projection = parallel
                plotter.atom_metallicness = metallic
                img_array = plotter.get_plot_screenshot(
                    transparent_background=transparent,
                    window_size=(width, height),
                    scale=scale,
                )
                pil_img = Image.fromarray(img_array)
                pil_img.save(f"{filename}{filetype}", format=filetype_options[filetype])

    if st.button("Apply"):
        st.session_state.apply_clicked = True

    # Apply update after clicking apply
    if st.session_state.get("apply_clicked", False):
        # apply settings
        for key, value in settings.items():
            # check if the value has changed. If so set it
            current_value = getattr(plotter, key)
            if current_value != value:
                setattr(plotter, key, value)
        # save html and rerun
        st.session_state.html_string = plotter.get_plot_html()
        st.session_state.apply_clicked = False

# Display plot
st.components.v1.html(st.session_state.html_string, height=st.session_state.page_height)
height = st.selectbox(
    "Page Height",
    options=[
        100,
        200,
        300,
        400,
        500,
        600,
        800,
        1000,
        1200,
        1600,
        2000,
        3000,
    ],
    index=3,
)
if height != st.session_state.page_height:
    st.session_state.page_height = height
    st.rerun()
# if st.button("Apply"):
#     # apply settings
#     for key, value in settings.items():
#         setattr(plotter, key, value)
#     # save html and rerun
#     st.session_state.html_string = plotter.get_plot_html()
#     st.rerun()
