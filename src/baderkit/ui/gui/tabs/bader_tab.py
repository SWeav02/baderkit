# -*- coding: utf-8 -*-
from pathlib import Path

from qtpy import QtCore as qc
from qtpy import QtWidgets as qw

from baderkit import Grid, Bader
from baderkit.ui.gui.widgets import (
    DoubleSpinBox,
    ErrorWindow,
    FilePicker,
)


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
        self.basic_layout = basic_layout
        
        # select calculation type
        self.analysis_type = qw.QComboBox()
        for source in ["Bader", "BadELF", "ElfLabeler"]:
            self.analysis_type.addItem(source)
        self.analysis_type.setCurrentText("Bader")
        basic_layout.addRow("Analysis Type", self.analysis_type)
        self.main.analysis_type = self.analysis_type

        # select data source
        self.data_source = qw.QComboBox()
        for source in ["charge_grid", "total_charge_grid", "reference_grid"]:
            self.data_source.addItem(source)
        self.data_source.setCurrentText("reference_grid")
        basic_layout.addRow("Isosurface Grid", self.data_source)
        self.main.data_source = self.data_source

        # add a file picker for the charge density
        self.charge_filepicker = FilePicker()
        self.charge_filepicker.line_edit.textChanged.connect(self.check_paths)
        basic_layout.addRow("Charge File", self.charge_filepicker)
        basic_layout.setAlignment(self.charge_filepicker, qc.Qt.AlignVCenter)

        # add a file picker for the total charge density
        self.total_filepicker = FilePicker()
        self.total_filepicker.line_edit.textChanged.connect(self.check_paths)
        basic_layout.addRow("Total Charge File (Optional)", self.total_filepicker)
        basic_layout.setAlignment(self.total_filepicker, qc.Qt.AlignVCenter)

        # add a file picker for the reference
        self.reference_filepicker = FilePicker()
        self.reference_filepicker.line_edit.textChanged.connect(self.check_paths)
        basic_layout.addRow("Reference File (Optional)", self.reference_filepicker)
        basic_layout.setAlignment(self.reference_filepicker, qc.Qt.AlignVCenter)

        # Add method dropdown
        self.bader_method_select = qw.QComboBox()
        for method in Bader.all_methods():
            self.bader_method_select.addItem(method)
        self.bader_method_select.setCurrentText("neargrid-weight")
        basic_layout.addRow("Bader Method", self.bader_method_select)

        # Add badelf method dropdown
        self.badelf_method_select = qw.QComboBox()
        for method in ["badelf", "voronelf", "zero-flux"]:
            self.badelf_method_select.addItem(method)
        self.badelf_method_select.setCurrentText("badelf")
        basic_layout.addRow("BadELF Method", self.badelf_method_select)
        # only show if BadELF is selected analysis
        self.analysis_type.currentTextChanged.connect(self.update_badelf_state)
        self.update_badelf_state(self.analysis_type.currentText())

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
        total_charge_path = Path(self.total_filepicker.file_path())
        
        valid_charge = charge_path.exists()
        if self.analysis_type == "BadELF" or self.analysis_type == "ElfLabeler":
            valid_reference = reference_path.exists()
        else:
            valid_reference = reference_path.exists() or not reference_path
        valid_total = total_charge_path.exists() or not total_charge_path    
        
        if valid_charge and valid_reference and valid_total:
            self.run_button.setEnabled(True)
        else:
            self.run_button.setEnabled(False)

    def run_bader(self):
        # disable button
        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")

        self.thread = qc.QThread()
        try:
            self.worker = BaderWorker(
                charge_path=Path(self.charge_filepicker.file_path()),
                total_path=Path(self.total_filepicker.file_path()),
                reference_path=Path(self.reference_filepicker.file_path()),
                analysis_type=self.analysis_type.currentText(),
                bader_method=self.bader_method_select.currentText(),
                badelf_method=self.badelf_method_select.currentText(),
                vacuum_tol=self.vacuum_tol.value(),
                basin_tol=self.basin_tol.value(),
            )
        except Exception as e:
            error = qc.Signal(str)
            error.connect(self.on_bader_error)
            error.emit(f"Bader class failed to load with the following error:\n {e}")
            return

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
    
    def update_badelf_state(self, text):
        # disable badelf options if not selected
        is_visible = text=="BadELF"
        self.badelf_method_select.setDisabled(not is_visible)
        row = self.basic_layout.labelForField(self.badelf_method_select)
        reference_row = self.basic_layout.labelForField(self.reference_filepicker)
        if is_visible:
            # show badelf method
            self.badelf_method_select.show()
            row.show()
        else:
            # hide badelf method
            row.hide()
            self.badelf_method_select.hide()     
            
        if is_visible or text=="ElfLabeler":
            # show reference grid as not optional
            reference_row.setText("Reference File")
        else:
            # show reference grid as optional
            reference_row.setText("Reference File (Optional)")
            
        # set reference grid as optional or not optional
        

class BaderWorker(qc.QObject):
    finished = qc.Signal(object)  # bader
    error = qc.Signal(str)

    def __init__(
        self,
        charge_path,
        total_path,
        reference_path,
        analysis_type,
        bader_method,
        badelf_method,
        vacuum_tol,
        basin_tol,
    ):
        super().__init__()
        self.charge_path = charge_path
        self.total_path = total_path
        self.reference_path = reference_path
        self.analysis_type = analysis_type
        self.bader_method = bader_method
        self.badelf_method = badelf_method
        self.vacuum_tol = vacuum_tol
        self.basin_tol = basin_tol

    @qc.Slot()
    def run(self):
        try:
            # get grids
            charge_path = self.charge_path
            charge_grid = Grid.from_dynamic(charge_path)

            total_path = self.total_path
            if total_path.name:
                total_grid = Grid.from_dynamic(total_path)
            else:
                total_grid = None
            reference_path = self.reference_path
            if reference_path.name:
                reference_grid = Grid.from_dynamic(reference_path)
            else:
                reference_grid = None
        except Exception as e:
            self.error.emit(f"Grid failed to load with the following error:\n {e}")
            return
        
        if self.analysis_type == "Bader":
            # create bader object
            from baderkit.bader import Bader as BaderObject
        elif self.analysis_type == "BadELF":
            from baderkit.elf_analysis import Badelf as BaderObject
        
        elif self.analysis_type == "ElfLabeler":
            from baderkit.elf_analysis import ElfLabeler as BaderObject
            
        bader = BaderObject(
            charge_grid=charge_grid,
            total_grid=total_grid,
            reference_grid=reference_grid,
            method=self.bader_method,
            badelf_method=self.badelf_method,
            vacuum_tol=self.vacuum_tol,
            basin_tol=self.basin_tol,
        )
            
        try:
            _ = bader.to_dict()  # force evaluation
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
