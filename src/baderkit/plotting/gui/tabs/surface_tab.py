# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from qtpy import QtWidgets as qw
from qtpy.QtCore import Qt

from baderkit.plotting.gui.widgets import ColorPicker, DoubleSpinBox


class SurfaceTab(qw.QWidget):

    def __init__(self, main, parent=None):
        super().__init__(parent)

        self.main = main
        self.name = "Surface"

        # Create a stacked layout at the base
        self.layout = qw.QStackedLayout()
        self.setLayout(self.layout)  # attach it to this QWidget

        # add a label for when there is no Bader result
        empty_label = qw.QLabel("Bader has not yet run")
        empty_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(empty_label)

        # Create a VBox that will hold the settings
        settings = qw.QWidget()
        self.settings_layout = qw.QVBoxLayout(settings)
        self.layout.addWidget(settings)

        #######################################################################
        # Shared Settings
        #######################################################################
        shared_widget = qw.QWidget()
        shared_layout = qw.QFormLayout()
        shared_widget.setLayout(shared_layout)
        self.settings_layout.addWidget(shared_widget)

        # Add widget to indicate range
        self.iso_range = qw.QLabel("")
        shared_layout.addRow("Value Range: ", self.iso_range)

        # Add iso value widget
        self.iso_val = DoubleSpinBox(
            decimals=4, step_size=0.01, main=main, plot_prop="iso_val"
        )
        shared_layout.addRow("Isosurface Value", self.iso_val)

        # Add widget to indicate range (updated later)
        self.iso_range = qw.QLabel("")
        shared_layout.addRow("", self.iso_range)

        # add colormap widget
        colormap = qw.QComboBox()
        for color in plt.colormaps():
            colormap.addItem(color)
        colormap.setCurrentText("viridis")
        colormap.plot_prop = "colormap"
        colormap.currentTextChanged.connect(main.set_property)
        shared_layout.addRow("Colormap", colormap)

        # -----------------------
        # surfaces
        # -----------------------
        surface_group = qw.QGroupBox("Show surface")
        surface_group.plot_prop = "show_surface"
        surface_group.setCheckable(True)
        surface_group.clicked.connect(main.set_property)
        self.settings_layout.addWidget(surface_group)

        # create layout for iso surface settings
        surface_layout = qw.QFormLayout()
        surface_group.setLayout(surface_layout)

        # Create HBox for color options (wrapper widget)
        surface_color_box = qw.QWidget()
        surface_color_layout = qw.QHBoxLayout(surface_color_box)
        surface_color_layout.setContentsMargins(0, 0, 0, 0)
        surface_color_layout.setAlignment(Qt.AlignVCenter)  # center vertically

        # add the wrapper to the form
        surface_layout.addRow("Solid Color", surface_color_box)
        surface_layout.setAlignment(
            surface_color_box, Qt.AlignVCenter
        )  # center vertically

        # checkbox
        use_solid_surface_color = qw.QCheckBox()
        use_solid_surface_color.plot_prop = "use_solid_surface_color"
        use_solid_surface_color.clicked.connect(main.set_property)

        # force checkbox to its natural (fixed/minimum) width so it doesn't stretch
        use_solid_surface_color.setSizePolicy(
            qw.QSizePolicy.Minimum, qw.QSizePolicy.Preferred
        )
        surface_color_layout.addWidget(use_solid_surface_color, 0)  # stretch=0

        # color picker (give it expanding horizontal policy so it fills remaining space)
        self.surface_color = ColorPicker(main=main, plot_prop="surface_color")
        self.surface_color.setEnabled(False)
        use_solid_surface_color.clicked.connect(self.surface_color.setEnabled)
        surface_color_layout.addWidget(self.surface_color, 1)  # stretch=1 -> expands

        # surface opacity (separate row)
        self.surface_opacity = DoubleSpinBox(
            min_value=0.0,
            max_value=1.0,
            plot_prop="surface_opacity",
            main=main,
        )
        surface_layout.addRow("Opacity", self.surface_opacity)

        # -----------------------
        # Caps
        # -----------------------
        cap_group = qw.QGroupBox("Show Caps")
        cap_group.plot_prop = "show_caps"
        cap_group.setCheckable(True)
        cap_group.clicked.connect(main.set_property)
        self.settings_layout.addWidget(cap_group)

        # create layout for iso cap settings
        cap_layout = qw.QFormLayout()
        cap_group.setLayout(cap_layout)

        # Create HBox for color options (wrapper widget)
        cap_color_box = qw.QWidget()
        cap_color_layout = qw.QHBoxLayout(cap_color_box)
        cap_color_layout.setContentsMargins(0, 0, 0, 0)
        cap_color_layout.setAlignment(Qt.AlignVCenter)  # center vertically

        # add the wrapper to the form
        cap_layout.addRow("Solid Color", cap_color_box)
        cap_layout.setAlignment(cap_color_box, Qt.AlignVCenter)  # center vertically

        # checkbox
        use_solid_cap_color = qw.QCheckBox()
        use_solid_cap_color.plot_prop = "use_solid_cap_color"
        use_solid_cap_color.clicked.connect(main.set_property)

        # force checkbox to its natural (fixed/minimum) width so it doesn't stretch
        use_solid_cap_color.setSizePolicy(
            qw.QSizePolicy.Minimum, qw.QSizePolicy.Preferred
        )
        cap_color_layout.addWidget(use_solid_cap_color, 0)  # stretch=0

        # color picker (give it expanding horizontal policy so it fills remaining space)
        self.cap_color = ColorPicker(main=main, plot_prop="cap_color")
        self.cap_color.setEnabled(False)
        use_solid_cap_color.clicked.connect(self.cap_color.setEnabled)
        cap_color_layout.addWidget(self.cap_color, 1)  # stretch=1 -> expands

        # cap opacity (separate row)
        self.cap_opacity = DoubleSpinBox(
            min_value=0.0,
            max_value=1.0,
            plot_prop="cap_opacity",
            main=main,
        )
        cap_layout.addRow("Opacity", self.cap_opacity)

        # push everything to the top
        self.settings_layout.addStretch()

    def set_bader(self):
        bader_plotter = self.main.bader_plotter

        # update iso value range/value
        self.iso_range.setText(
            f"({round(bader_plotter.min_val, 4)} - {round(bader_plotter.max_val, 4)})"
        )
        self.iso_val.setMinimum(bader_plotter.min_val)
        self.iso_val.setMaximum(bader_plotter.max_val)
        self.iso_val.setValue(bader_plotter._iso_val)

        # update opacity value
        self.surface_opacity.setValue(bader_plotter.surface_opacity)
        self.cap_opacity.setValue(bader_plotter.cap_opacity)
        # update color
        self.cap_color.set_color(bader_plotter.cap_color)

        # Make options visible
        self.layout.setCurrentIndex(1)
