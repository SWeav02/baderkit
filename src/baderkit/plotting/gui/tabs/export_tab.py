# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from PIL import Image
from qtpy import QtWidgets as qw
from qtpy.QtCore import Qt

from baderkit.core import Grid
from baderkit.plotting.gui.widgets import SpinBox


class ExportTab(qw.QWidget):

    def __init__(self, main, parent=None):
        super().__init__(parent)

        self.main = main
        self.name = "Export"

        # Create a stacked layout at the base
        self.layout = qw.QStackedLayout()
        self.setLayout(self.layout)  # attach it to this QWidget

        # add a label for when there is no Bader result
        empty_label = qw.QLabel("Bader has not yet run")
        empty_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(empty_label)

        # Create a VBox that will hold the settings
        settings = qw.QWidget()
        settings_layout = qw.QVBoxLayout(settings)
        self.layout.addWidget(settings)

        ###############
        # Basin Export
        ###############
        # create box to surround basin export info
        basin_group = qw.QGroupBox("Image Export")
        basin_group_layout = qw.QVBoxLayout(basin_group)
        settings_layout.addWidget(basin_group)

        # create a form layout for settings
        basin_form = qw.QWidget()
        basin_form_layout = qw.QFormLayout()
        basin_form.setLayout(basin_form_layout)
        basin_group_layout.addWidget(basin_form)

        # select data source
        self.data_source = qw.QComboBox()
        for source in ["Charge", "Reference"]:
            self.data_source.addItem(source)
        self.data_source.setCurrentText("Charge")
        basin_form_layout.addRow("Grid", self.data_source)

        # export button
        export_button = qw.QPushButton("Export Basins")
        export_button.clicked.connect(self.export_basins)
        basin_group_layout.addWidget(export_button)

        ###############
        # Image Export
        ###############
        # create box to surround image export info
        image_group = qw.QGroupBox("Image Export")
        image_group_layout = qw.QVBoxLayout(image_group)
        settings_layout.addWidget(image_group)
        # create a form layout for settings
        image_form = qw.QWidget()
        image_form_layout = qw.QFormLayout()
        image_form.setLayout(image_form_layout)
        image_group_layout.addWidget(image_form)

        # width
        self.width = SpinBox(min_value=100, max_value=int(1e9), current_value=1000)
        image_form_layout.addRow("Width", self.width)
        # height
        self.height = SpinBox(min_value=100, max_value=int(1e9), current_value=1000)
        image_form_layout.addRow("Height", self.height)
        # scale
        self.scale = SpinBox(min_value=1, max_value=int(1e9), current_value=1)
        image_form_layout.addRow("Scale", self.scale)
        # transparent background
        self.transparent = qw.QCheckBox()
        image_form_layout.addRow("Transparent Background", self.transparent)
        # export button
        export_button = qw.QPushButton("Export Image")
        export_button.clicked.connect(self.export_image)
        image_group_layout.addWidget(export_button)

        # push everything to top
        settings_layout.addStretch()

    def set_bader(self):
        # Make options visible
        self.layout.setCurrentIndex(1)

    def export_basins(self):
        format_filters = "VASP (*.vasp);;CUBE (*.cube);;HDF5 (*.hdf5)"

        # open dialog
        filename, selected_filter = qw.QFileDialog.getSaveFileName(
            self, "Export Basins", "", format_filters, options=qw.QFileDialog.Options()
        )
        if filename:
            # Detect extension from filter if missing
            if "." not in filename:
                ext = (
                    selected_filter.split("(")[1]
                    .split(")")[0]
                    .split()[0]
                    .replace("*", "")
                )
                filename += ext
            else:
                ext = Path(filename).suffix

        # get bader and plotter objects
        bader = self.main.bader
        plotter = self.main.bader_plotter

        # get mask where current selected basins are
        basin_labels = bader.basin_labels
        data_mask = np.isin(basin_labels, list(plotter.visible_bader_basins))

        # get a copy of the grid
        if self.data_source.currentText == "Charge":
            total = bader.charge_grid.total.copy()
        else:
            total = bader.reference_grid.total.copy()
        total[~data_mask] = 0.0
        temp_grid = Grid(bader.structure, data={"total": total})

        # write to file
        temp_grid.write(filename, output_format=ext.strip("."))

    def export_image(self):
        # get possible extensions
        image_filters = ""
        for ext, name in Image.registered_extensions().items():
            image_filters += f"{name} (*{ext});;"

        # open dialog
        filename, selected_filter = qw.QFileDialog.getSaveFileName(
            self, "Export Image", "", image_filters, options=qw.QFileDialog.Options()
        )
        if filename:
            # Detect extension from filter if missing
            if "." not in filename:
                ext = (
                    selected_filter.split("(")[1]
                    .split(")")[0]
                    .split()[0]
                    .replace("*", "")
                )
                filename += ext
            else:
                ext = Path(filename).suffix

        # Call write method
        self.write_image(filename, ext)

    def write_image(self, path: str, filetype: str):
        img_array = self.main.bader_plotter.get_plot_screenshot(
            transparent_background=self.transparent.isChecked(),
            window_size=(self.width.value(), self.height.value()),
            scale=self.scale.value(),
        )
        pil_img = Image.fromarray(img_array)
        pil_img.save(path, format=Image.registered_extensions()[filetype])
