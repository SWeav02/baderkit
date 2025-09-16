# -*- coding: utf-8 -*-

from qtpy import QtWidgets as qw
from qtpy.QtCore import Qt

from baderkit.plotting.gui.widgets import ColorPicker, DoubleSpinBox, SpinBox


class ViewTab(qw.QWidget):

    def __init__(self, main, parent=None):
        super().__init__(parent)

        self.main = main
        self.name = "View"

        # Create a stacked layout at the base
        self.layout = qw.QStackedLayout()
        self.setLayout(self.layout)  # attach it to this QWidget

        # add a label for when there is no Bader result
        empty_label = qw.QLabel("Bader has not yet run")
        empty_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(empty_label)

        ###################
        # Settings
        ###################
        # create vbox layout for settings
        settings = qw.QWidget()
        settings_layout = qw.QVBoxLayout()
        settings.setLayout(settings_layout)
        self.layout.addWidget(settings)

        # create a form layout for basic settings
        basic = qw.QWidget()
        self.basic_layout = qw.QFormLayout()
        basic.setLayout(self.basic_layout)
        settings_layout.addWidget(basic)

        # show lattice
        lattice_checkbox = qw.QCheckBox()
        lattice_checkbox.setChecked(True)
        lattice_checkbox.setSizePolicy(qw.QSizePolicy.Minimum, qw.QSizePolicy.Preferred)
        lattice_checkbox.plot_prop = "show_lattice"
        lattice_checkbox.toggled.connect(main.set_property)
        self.basic_layout.addRow("Show Lattice", lattice_checkbox)

        # lattice thickness
        self.lattice_thickness = DoubleSpinBox(
            min_value=0.01, max_value=10.00, plot_prop="lattice_thickness", main=main
        )
        self.basic_layout.addRow("Lattice Thickness", self.lattice_thickness)

        # background color
        background_color = ColorPicker("#FFFFFF", plot_prop="background", main=main)
        self.basic_layout.addRow("Background Color", background_color)

        # parallel perspective
        parallel_checkbox = qw.QCheckBox()
        parallel_checkbox.setChecked(True)
        parallel_checkbox.setSizePolicy(
            qw.QSizePolicy.Minimum, qw.QSizePolicy.Preferred
        )
        parallel_checkbox.plot_prop = "parallel_projection"
        parallel_checkbox.toggled.connect(main.set_property)
        self.basic_layout.addRow("Parallel Perspective", parallel_checkbox)

        # create a visual box for the view
        view_group = qw.QGroupBox("Camera")
        view_group_layout = qw.QVBoxLayout(view_group)
        settings_layout.addWidget(view_group)
        # create a form layout for view
        view = qw.QWidget()
        self.view_layout = qw.QFormLayout()
        view.setLayout(self.view_layout)
        view_group_layout.addWidget(view)

        # view indices
        self.view_indices = []
        # create HBox to store 3 SpinBoxes
        view_box = qw.QWidget()
        view_layout = qw.QHBoxLayout()
        view_box.setLayout(view_layout)
        for i in range(3):
            spinner = SpinBox(min_value=int(-1e9), max_value=int(1e9))
            # Make spinners expand horizontally but stay centered vertically
            spinner.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Fixed)
            view_layout.addWidget(spinner, stretch=1, alignment=Qt.AlignVCenter)
            self.view_indices.append(spinner)
        self.view_layout.addRow("View Vector (miller)", view_box)

        # rotation
        self.rotation = DoubleSpinBox(
            min_value=0.0,
            max_value=360.0,
            decimals=1,
            current_value=0.0,
        )
        self.view_layout.addRow("Camera Rotation", self.rotation)

        # view button
        view_button = qw.QPushButton("Set View")
        view_button.pressed.connect(self.set_view)
        view_group_layout.addWidget(view_button)

        # push everything to the top
        settings_layout.addStretch()

    def set_bader(self):
        bader_plotter = self.main.bader_plotter

        # set thickness
        self.lattice_thickness.setValue(bader_plotter.lattice_thickness)

        # set view
        view = bader_plotter.view_indices
        for i, spinner in zip(view, self.view_indices):
            spinner.setValue(i)

        # Make options visible
        self.layout.setCurrentIndex(1)

    def set_view(self):
        bader_plotter = self.main.bader_plotter

        # get miller indices
        i, j, k = (int(view.value()) for view in self.view_indices)

        # get rotation
        rotation = self.rotation.value()

        # get camera position
        camera_position = bader_plotter.get_camera_position_from_miller(
            i, j, k, rotation
        )

        self.main.set_property(camera_position, "camera_position")
