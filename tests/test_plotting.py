# -*- coding: utf-8 -*-

import numpy as np
from pymatgen.core import Lattice

from baderkit.plotting.toolkit import GridPlotter, StructurePlotter
from baderkit.toolkit import Grid, Structure


def _make_ethene_structure() -> Structure:
    lattice = Lattice.orthorhombic(10.0, 10.0, 10.0)
    species = ["C", "C", "H", "H", "H", "H"]
    frac_coords = np.array(
        [
            [0.433, 0.500, 0.500],
            [0.567, 0.500, 0.500],
            [0.376, 0.593, 0.500],
            [0.376, 0.407, 0.500],
            [0.624, 0.593, 0.500],
            [0.624, 0.407, 0.500],
        ],
        dtype=np.float64,
    )
    return Structure(lattice, species, frac_coords)


def _make_signed_grid(structure: Structure, shape=(16, 16, 16)) -> Grid:
    x = np.linspace(0, 2 * np.pi, shape[0], endpoint=False)
    y = np.linspace(0, 2 * np.pi, shape[1], endpoint=False)
    z = np.linspace(0, 2 * np.pi, shape[2], endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    data = np.sin(X) * np.cos(Y) * np.cos(Z)
    return Grid(structure=structure, data={"total": data})


def test_structure_plotter_defaults_and_bonds():
    structure = _make_ethene_structure()
    sp = StructurePlotter(
        structure=structure,
        off_screen=True,
        show_lattice=False,
        show_axes=False,
        wrap_atoms=False,
    )
    assert sp.pbr is True
    assert sp.anti_aliasing == "ssaa"
    assert sp.depth_peeling is True
    assert sp.sphere_resolution == 40
    assert sp.show_bonds is False
    assert "bonds" not in sp.plotter.actors

    # Enable automatic covalent bonds (1 C-C + 4 C-H = 5 bonds)
    sp.show_bonds = True
    assert len(sp._find_bond_segments()) == 5
    assert "bonds" in sp.plotter.actors

    # Switch to solid bond color and explicit pair override
    sp.bond_color = "#888888"
    sp.bond_pairs = [(0, 1)]
    assert len(sp._find_bond_segments()) == 1
    assert "bonds" in sp.plotter.actors

    # Disable bonds cleanly
    sp.show_bonds = False
    assert "bonds" not in sp.plotter.actors
    sp.plotter.close()


def test_grid_plotter_high_quality_signed_isosurfaces():
    structure = _make_ethene_structure()
    grid = _make_signed_grid(structure)

    gp = GridPlotter(
        grid=grid,
        iso_value=0.3,
        show_negative_surface=True,
        show_caps=False,
        use_solid_surface_color=True,
        surface_color="#d62728",
        negative_surface_color="#1f77b4",
        surface_opacity=0.6,
        upsample_factor=2.0,
        show_bonds=True,
        off_screen=True,
        show_lattice=False,
        show_axes=False,
        wrap_atoms=False,
    )
    assert gp.pbr is True
    assert gp.smooth_surface is True
    assert gp.smooth_shading is True
    assert gp.negative_iso_value == -0.3
    assert "iso" in gp.plotter.actors
    assert "iso_neg" in gp.plotter.actors
    assert "bonds" in gp.plotter.actors
    # Upsampled grid should have (16*2 + 1) = 33 points per axis
    assert gp._structured_grid.dimensions == (33, 33, 33)

    img = gp.get_plot_screenshot(window_size=(200, 200), return_img=True)
    assert img.shape == (200, 200, 3)
    gp.plotter.close()
