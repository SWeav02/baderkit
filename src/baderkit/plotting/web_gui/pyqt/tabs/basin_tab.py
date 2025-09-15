# -*- coding: utf-8 -*-

from qtpy import QtWidgets as qw
from qtpy.QtCore import Qt


class BasinTab(qw.QWidget):

    def __init__(self, main, parent=None):
        super().__init__(parent)

        self.main = main
        self.name = "Basins"
        
        self.basin_mode = False

        # Create a stacked layout at the base
        self.stackedlayout = qw.QStackedLayout()
        self.setLayout(self.stackedlayout)   # attach it to this QWidget

        # add a label for when there is no Bader result
        empty_label = qw.QLabel("Bader has not yet run")
        empty_label.setAlignment(Qt.AlignCenter)
        self.stackedlayout.addWidget(empty_label)

        # create a QWidget to hold the selectlayout
        select_widget = qw.QWidget()
        self.selectlayout = qw.QVBoxLayout(select_widget)
        self.stackedlayout.addWidget(select_widget)

        # create an atom/basin toggle
        self.basin_toggle = qw.QHBoxLayout()
        self.to_atom_button = qw.QRadioButton("Atoms")
        self.to_basin_button = qw.QRadioButton("Basins")
        # connect
        self.to_atom_button.toggled.connect(self.set_to_atom)
        self.to_basin_button.toggled.connect(self.set_to_basin)
        # add to layout
        self.basin_toggle.addWidget(self.to_atom_button)
        self.basin_toggle.addWidget(self.to_basin_button)
        self.basin_toggle.addStretch() # push to left
        self.selectlayout.addLayout(self.basin_toggle)

        # create an empty stacked layout for the arrays
        self.arraylayout = qw.QStackedLayout()
        self.selectlayout.addLayout(self.arraylayout)

    
    def set_bader(self):
        bader = self.main.bader
        bader_plotter = self.main.bader_plotter
        num_atoms = len(bader.structure)
        num_basins = len(bader.basin_maxima_frac)
        active_atoms = bader_plotter.visible_atom_basins
        active_basins = bader_plotter.visible_bader_basins
    
        num_columns = 5  # fixed number of columns per row
    
        # ----------------
        # Atom buttons
        # ----------------
        atom_grid = qw.QGridLayout()
        self.atom_buttons = []
        
        for i in range(num_atoms):
            row_idx = i // num_columns
            col_idx = i % num_columns
            button = qw.QPushButton(str(i))
            button.setCheckable(True)
            if i in active_atoms:
                button.setChecked(True)
            button.clicked.connect(self.set_plotter_atoms)
            button.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Preferred)
            atom_grid.addWidget(button, row_idx, col_idx)
            self.atom_buttons.append(button)
        
        # Fill empty slots in last row
        last_row_items = num_atoms % num_columns
        if last_row_items != 0:
            row_idx = num_atoms // num_columns
            for col_idx in range(last_row_items, num_columns):
                spacer = qw.QWidget()
                spacer.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Preferred)
                atom_grid.addWidget(spacer, row_idx, col_idx)
        
        # Create a widget to hold the grid
        atom_container = qw.QWidget()
        atom_layout_wrapper = qw.QVBoxLayout(atom_container)
        atom_layout_wrapper.addLayout(atom_grid)
        atom_layout_wrapper.addStretch()  # push buttons to top
        
        # Wrap in a scroll area
        atom_scroll_area = qw.QScrollArea()
        atom_scroll_area.setWidgetResizable(True)
        atom_scroll_area.setWidget(atom_container)
        atom_scroll_area.setMinimumHeight(200)  # optional: set visible height

    
        # ----------------
        # Basin buttons
        # ----------------
        basin_grid = qw.QGridLayout()
        self.basin_buttons = []
    
        for i in range(num_basins):
            row_idx = i // num_columns
            col_idx = i % num_columns
            button = qw.QPushButton(str(i))
            button.setCheckable(True)
            if i in active_basins:
                button.setChecked(True)
            button.clicked.connect(self.set_plotter_basins)
            button.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Preferred)
            basin_grid.addWidget(button, row_idx, col_idx)
            self.basin_buttons.append(button)
    
        # Fill empty slots in last row
        last_row_items = num_basins % num_columns
        if last_row_items != 0:
            row_idx = num_basins // num_columns
            for col_idx in range(last_row_items, num_columns):
                spacer = qw.QWidget()
                spacer.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Preferred)
                basin_grid.addWidget(spacer, row_idx, col_idx)
    
        basin_container = qw.QWidget()
        basin_layout_wrapper = qw.QVBoxLayout(basin_container)
        basin_layout_wrapper.addLayout(basin_grid)
        basin_layout_wrapper.addStretch()  # push buttons to top
        
        # Wrap in a scroll area
        basin_scroll_area = qw.QScrollArea()
        basin_scroll_area.setWidgetResizable(True)
        basin_scroll_area.setWidget(basin_container)
        basin_scroll_area.setMinimumHeight(200)  # optional: set visible height
    
        # ----------------
        # Replace old widgets in stacked layout
        # ----------------
        while self.arraylayout.count():
            item = self.arraylayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
    
        # Add widgets instead of layouts
        self.arraylayout.addWidget(atom_scroll_area)
        self.arraylayout.addWidget(basin_scroll_area)
    
        # Select correct mode
        if self.basin_mode:
            self.arraylayout.setCurrentIndex(1)
            self.to_basin_button.setChecked(True)
        else:
            self.arraylayout.setCurrentIndex(0)
            self.to_atom_button.setChecked(True)
    
        # Make arrays visible
        self.stackedlayout.setCurrentIndex(1)
            
    def set_to_basin(self):
        self.basin_mode = True
        self.arraylayout.setCurrentIndex(1)
        self.set_plotter_basins()
        # set atoms to none
        self.main.set_property([], "visible_atom_basins")
    
    def set_to_atom(self):
        self.basin_mode = False
        self.arraylayout.setCurrentIndex(0)
        self.set_plotter_atoms()
        # set basins to none
        self.main.set_property([], "visible_bader_basins")
        
    def set_plotter_basins(self):
        self.main.set_property(
            [i for i, button in enumerate(self.basin_buttons) if button.isChecked()],
            "visible_bader_basins"
            )
        
    def set_plotter_atoms(self):
        self.main.set_property(
            [i for i, button in enumerate(self.atom_buttons) if button.isChecked()],
            "visible_atom_basins"
            )