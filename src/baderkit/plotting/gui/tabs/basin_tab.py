# -*- coding: utf-8 -*-

from qtpy import QtWidgets as qw
from qtpy.QtCore import Qt

from baderkit.plotting.gui.widgets import centered_widget


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

        # Create a tree that will hold the atom/basin selection
        tree = qw.QTreeWidget()
        tree.setColumnCount(4)
        tree.setHeaderLabels(["Label", "Visible", "Charge", "Volume", "Coord"])
        header = tree.header()
        header.setSectionResizeMode(0, qw.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qw.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, qw.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, qw.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, qw.QHeaderView.ResizeToContents)
        tree.setAlternatingRowColors(True)
        self.stackedlayout.addWidget(tree)
        self.tree = tree

    def set_bader(self):

        # clear tree
        self.clear_tree_widgets()
        # temporarily disable updates so that all updates appear at once
        self.tree.setUpdatesEnabled(False)

        bader = self.main.bader
        structure = bader.structure

        atom_trees = {}
        atom_visibility = {}

        # create a tree row for each atom
        for i, site in enumerate(structure):
            # get data
            charge = str(round(bader.atom_charges[i], 4))
            volume = str(round(bader.atom_volumes[i], 4))
            coords = "(" + ", ".join(f"{x:.3f}" for x in site.frac_coords) + ")"

            # create a widget for visibility
            visible_widget = qw.QCheckBox()
            if i == 0:
                visible_widget.setChecked(True)
            atom_visibility[i] = visible_widget

            # make parent row
            item = qw.QTreeWidgetItem(
                self.tree, [site.label, "", charge, volume, coords]
            )  # only string in first col
            self.tree.addTopLevelItem(item)

            # attach visibility widget
            self.tree.setItemWidget(item, 1, centered_widget(visible_widget))

            # store row
            atom_trees[i] = item

        # now add basins belonging to each atom
        self.basin_visibility = []

        for i, atom_idx in enumerate(bader.basin_atoms):
            # get data
            charge = str(round(bader.basin_charges[i], 4))
            volume = str(round(bader.basin_volumes[i], 4))
            coords = (
                "(" + ", ".join(f"{x:.3f}" for x in bader.basin_maxima_frac[i]) + ")"
            )

            # create a widget for visibility
            visible_widget = qw.QCheckBox()
            if atom_idx == 0:
                visible_widget.setChecked(True)
            self.basin_visibility.append(visible_widget)
            # connect to setter
            visible_widget.toggled.connect(self.set_plotter_basins)
            # connect to parent atom
            atom_visibility[atom_idx].toggled.connect(visible_widget.setChecked)

            # make child row under the species row
            basin_item = qw.QTreeWidgetItem(
                atom_trees[atom_idx], [f"basin {i}", "", charge, volume, coords]
            )
            self.tree.setItemWidget(basin_item, 1, centered_widget(visible_widget))

        # enable updates
        self.tree.setUpdatesEnabled(True)
        # Make options visible
        self.stackedlayout.setCurrentIndex(1)

    def set_plotter_basins(self):
        self.main.set_property(
            [i for i, button in enumerate(self.basin_visibility) if button.isChecked()],
            "visible_bader_basins",
        )

    # Iterate all items and delete their widgets
    def clear_tree_widgets(self):
        top_items = [
            self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())
        ]
        for item in top_items:
            self._clear_item_widgets_recursive(item)

        # finally clear the items themselves
        self.tree.clear()

    def _clear_item_widgets_recursive(self, item):
        for col in range(self.tree.columnCount()):
            widget = self.tree.itemWidget(item, col)
            if widget:
                self.tree.removeItemWidget(item, col)
                widget.deleteLater()

        # recurse into children
        for i in range(item.childCount()):
            self._clear_item_widgets_recursive(item.child(i))
