# -*- coding: utf-8 -*-
from pathlib import Path

from qtpy import QtCore as qc
from qtpy import QtWidgets as qw

from baderkit.core import Bader, Grid
from baderkit.plotting.gui.widgets import DoubleSpinBox, ErrorWindow, FilePicker


class BaderTab(qw.QWidget):

    def __init__(
        self,
        main,
        parent=None,
    ):
        super().__init__(parent)

        # link to main application
        self.main = main

        # set tab name
        self.name = "Run"

        # create layout to hold widgets
        layout = qw.QVBoxLayout(self)

        # create layout for basic settings
        basic_box = qw.QWidget()
        basic_layout = qw.QFormLayout()
        basic_box.setLayout(basic_layout)
        layout.addWidget(basic_box)

        # add a file picker for the charge density
        self.charge_filepicker = FilePicker()
        self.charge_filepicker.line_edit.textChanged.connect(self.check_paths)
        basic_layout.addRow("Charge File", self.charge_filepicker)
        basic_layout.setAlignment(self.charge_filepicker, qc.Qt.AlignVCenter)

        # add a file picker for the reference
        self.reference_filepicker = FilePicker()
        self.reference_filepicker.line_edit.textChanged.connect(self.check_paths)
        basic_layout.addRow("Reference File (Optional)", self.reference_filepicker)
        basic_layout.setAlignment(self.reference_filepicker, qc.Qt.AlignVCenter)

        # Add method dropdown
        self.method_select = qw.QComboBox()
        for method in Bader.all_methods():
            self.method_select.addItem(method)
        self.method_select.setCurrentText("weight")
        basic_layout.addRow("Method", self.method_select)

        # Add advanced options box
        advanced_box = qw.QGroupBox("Advanced Options")
        advanced_box.setCheckable(True)  # makes it collapsible
        advanced_box.setChecked(False)  # start collapsed

        adv_layout = qw.QFormLayout()
        self.vacuum_tol = DoubleSpinBox(
            min_value=-1.0e12,
            max_value=1.0e12,
            current_value=1.0e-3,
            step_size=0.1,
            decimals=3,
        )
        self.basin_tol = DoubleSpinBox(
            min_value=-1.0e12,
            max_value=1.0e12,
            current_value=1.0e-3,
            step_size=0.1,
            decimals=3,
        )
        adv_layout.addRow("Vacuum Tolerance", self.vacuum_tol)
        adv_layout.addRow("Basin Tolerance", self.basin_tol)
        advanced_box.setLayout(adv_layout)
        layout.addWidget(advanced_box)

        # Add run button
        self.run_button = qw.QPushButton("Run Bader")
        self.run_button.pressed.connect(self.run_bader)
        self.run_button.setEnabled(False)  # disable at start
        layout.addWidget(self.run_button)

        # push everything to top
        layout.addStretch()
        self.layout = layout

    def check_paths(self):
        # If paths are valid, enable bader run button
        charge_path = Path(self.charge_filepicker.file_path())
        reference_path = Path(self.reference_filepicker.file_path())
        valid_charge = charge_path.exists()
        valid_reference = reference_path.exists() or not reference_path
        if valid_charge and valid_reference:
            self.run_button.setEnabled(True)
        else:
            self.run_button.setEnabled(False)

    def run_bader(self):
        # disable button
        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")

        self.thread = qc.QThread()
        self.worker = BaderWorker(
            charge_path=Path(self.charge_filepicker.file_path()),
            reference_path=Path(self.reference_filepicker.file_path()),
            method=self.method_select.currentText(),
            vacuum_tol=self.vacuum_tol.value(),
            basin_tol=self.basin_tol.value(),
        )

        # connect worker signals
        self.worker.finished.connect(self.on_bader_finished)
        self.worker.error.connect(self.on_bader_error)

        # move worker to thread
        self.worker.moveToThread(self.thread)

        # start worker when thread starts
        self.thread.started.connect(self.worker.run)

        # cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # run
        self.thread.start()

    def on_bader_finished(self, bader):
        self.main.set_bader(bader)
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Bader")

    def on_bader_error(self, message):
        ErrorWindow(self.main, message)
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Bader")

    def set_bader(self):
        # Must be set for method in main
        pass


class BaderWorker(qc.QObject):
    finished = qc.Signal(object)  # bader
    error = qc.Signal(str)

    def __init__(
        self,
        charge_path,
        reference_path,
        method,
        vacuum_tol,
        basin_tol,
    ):
        super().__init__()
        self.charge_path = charge_path
        self.reference_path = reference_path
        self.method = method
        self.vacuum_tol = vacuum_tol
        self.basin_tol = basin_tol

    @qc.Slot()
    def run(self):
        try:
            # get grids
            charge_path = self.charge_path
            charge_grid = Grid.from_dynamic(charge_path)
            reference_path = self.reference_path
            if reference_path.name:
                reference_grid = Grid.from_dynamic(reference_path)
            else:
                reference_grid = None
        except Exception as e:
            self.error.emit(f"Grid failed to load with the following error:\n {e}")
            return

        # create bader object
        bader = Bader(
            charge_grid,
            reference_grid,
            method=self.method,
            vacuum_tol=self.vacuum_tol,
            basin_tol=self.basin_tol,
        )

        try:
            _ = bader.results_summary  # force evaluation
        except Exception as e:
            self.error.emit(f"Bader algorithm failed with the following error:\n {e}")
            return

        # success
        self.finished.emit(bader)

    # def add_widget(self):
    #     # add set isosurface test
    #     min_val = self.main.bader_plotter.min_val
    #     max_val = self.main.bader_plotter.max_val
    #     print(min_val)
    #     print(max_val)
    #     current_val = self.main.bader_plotter._iso_val
    #     iso_value = qw.QDoubleSpinBox()
    #     iso_value.plot_prop = "iso_val"
    #     iso_value.setRange(min_val, max_val)
    #     iso_value.setSingleStep(0.1)     # increment step
    #     iso_value.setValue(current_val)         # default value
    #     iso_value.valueChanged.connect(self.main.set_property)
    #     self.layout.addWidget(iso_value)
