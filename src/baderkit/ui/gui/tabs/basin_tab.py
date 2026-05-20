# -*- coding: utf-8 -*-

from qtpy import QtWidgets as qw
from qtpy.QtCore import Qt

from baderkit.ui.gui.widgets import centered_widget


class BasinTab(qw.QWidget):

    def __init__(self, main, parent=None):
        super().__init__(parent)

        self.main = main
        self.name = "Basins"

        self.basin_mode = False

        # Create a stacked layout at the base
        self.stackedlayout = qw.QStackedLayout()
        self.setLayout(self.stackedlayout)  # attach it to this QWidget

        # add a label for when there is no Bader result
        empty_label = qw.QLabel("Bader has not yet run")
        empty_label.setAlignment(Qt.AlignCenter)
        self.stackedlayout.addWidget(empty_label)

        # Create a table that will hold the basin selection
        table = self.get_table_widget()
        self.stackedlayout.addWidget(table)
        self.table = table

    def set_bader(self):

        # clear table
        self.table.clearContents()
        self.table.setRowCount(0)
        # temporarily disable updates so that all updates appear at once
        self.table.setUpdatesEnabled(False)

        # create table
        self.create_table_rows()

        # resize columns
        self.table.resizeColumnsToContents()

        # Don't allow editing
        self.table.setEditTriggers(qw.QAbstractItemView.NoEditTriggers)

        # enable updates
        self.table.setUpdatesEnabled(True)
        # Make options visible
        self.stackedlayout.setCurrentIndex(1)

    def set_atom_basins(self):
        self.main.set_property(
            [i for i, button in self.atom_visibility.items() if button.isChecked()],
            "visible_atom_basins",
        )

    def set_feature_basins(self):
        self.main.set_property(
            [i for i, button in self.feature_visibility.items() if button.isChecked()],
            "visible_chemical_features",
        )

    def get_table_widget(self):
        analysis_type = self.main.analysis_type.currentText()
        if analysis_type == "Bader" or analysis_type == "BadELF":
            table = qw.QTableWidget()
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(
                ["Label", "Visible", "Charge", "Volume", "Coord"]
            )
            table.setAlternatingRowColors(True)
            return table
        elif analysis_type == "ElfLabeler":
            table = qw.QTableWidget()
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(["Label", "Visible"])
            table.setAlternatingRowColors(True)
            return table

    def create_table_rows(self):
        bader = self.main.bader
        analysis_type = self.main.analysis_type.currentText()
        if analysis_type == "Bader" or analysis_type == "BadELF":
            if analysis_type == "Bader":
                structure = bader.structure
            else:
                structure = bader.nna_structure
            self.atom_visibility = {}

            # create a table row for each atom
            for i, site in enumerate(structure):
                # create row
                self.table.insertRow(i)

                # get data
                charge = str(round(bader.atom_charges[i], 4))
                volume = str(round(bader.atom_volumes[i], 4))
                coords = "(" + ", ".join(f"{x:.3f}" for x in site.frac_coords) + ")"

                # create a widget for visibility
                visible_widget = qw.QCheckBox()
                self.atom_visibility[i] = visible_widget

                # connect checkbox
                visible_widget.toggled.connect(self.set_atom_basins)

                # create table items
                self.table.setItem(i, 0, qw.QTableWidgetItem(site.label))
                self.table.setCellWidget(i, 1, centered_widget(visible_widget))
                self.table.setItem(i, 2, qw.QTableWidgetItem(charge))
                self.table.setItem(i, 3, qw.QTableWidgetItem(volume))
                self.table.setItem(i, 4, qw.QTableWidgetItem(coords))

        elif analysis_type == "ElfLabeler":
            features = bader.types_in_system
            self.feature_visibility = {}

            # create a table row for each atom
            for i, feature in enumerate(features):
                # create row
                self.table.insertRow(i)

                # create a widget for visibility
                visible_widget = qw.QCheckBox()
                self.feature_visibility[i] = visible_widget

                # connect checkbox
                visible_widget.toggled.connect(self.set_feature_basins)

                # create table items
                self.table.setItem(i, 0, qw.QTableWidgetItem(feature))
                self.table.setCellWidget(i, 1, centered_widget(visible_widget))

    # # Iterate all items and delete their widgets
    # def clear_table_widgets(self):
    #     top_items = [
    #         self.table.topLevelItem(i) for i in range(self.table.topLevelItemCount())
    #     ]
    #     for item in top_items:
    #         self._clear_item_widgets_recursive(item)

    #     # finally clear the items themselves
    #     self.table.clear()

    # def _clear_item_widgets_recursive(self, item):
    #     for col in range(self.table.columnCount()):
    #         widget = self.table.itemWidget(item, col)
    #         if widget:
    #             self.table.removeItemWidget(item, col)
    #             widget.deleteLater()

    #     # recurse into children
    #     for i in range(item.childCount()):
    #         self._clear_item_widgets_recursive(item.child(i))
