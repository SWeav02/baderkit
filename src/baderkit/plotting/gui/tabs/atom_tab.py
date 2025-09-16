# -*- coding: utf-8 -*-

from pymatgen.core import Species
from qtpy import QtWidgets as qw
from qtpy.QtCore import Qt

from baderkit.plotting.core.defaults import ATOM_COLORS
from baderkit.plotting.gui.widgets import ColorPicker, DoubleSpinBox, centered_widget


class AtomsTab(qw.QWidget):

    def __init__(self, main, parent=None):
        super().__init__(parent)

        self.main = main
        self.name = "Atoms"

        # Create a stacked layout at the base
        self.layout = qw.QStackedLayout()
        self.setLayout(self.layout)  # attach it to this QWidget

        # add a label for when there is no Bader result
        empty_label = qw.QLabel("Bader has not yet run")
        empty_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(empty_label)

        # add VBox for settings
        settings = qw.QWidget()
        self.settings_layout = qw.QVBoxLayout()
        settings.setLayout(self.settings_layout)
        self.layout.addWidget(settings)

        # add metallicity
        form = qw.QWidget()
        form_layout = qw.QFormLayout()
        form.setLayout(form_layout)
        self.settings_layout.addWidget(form)
        self.metallicity = DoubleSpinBox(
            min_value=0.0,
            max_value=1.0,
            step_size=0.01,
            main=main,
            plot_prop="atom_metallicness",
        )
        form_layout.addRow("Metallicity", self.metallicity)

        # Create a tree that will hold the settings
        tree = qw.QTreeWidget()
        tree.setColumnCount(4)
        tree.setHeaderLabels(["Label", "Visible", "Radius", "Color"])
        header = tree.header()
        header.setSectionResizeMode(0, qw.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qw.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, qw.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, qw.QHeaderView.Stretch)

        tree.setAlternatingRowColors(True)
        self.settings_layout.addWidget(tree)
        self.tree = tree

    def set_bader(self):

        # clear tree
        self.clear_tree_widgets()
        # temporarily disable updates to repainting after every insert
        self.tree.setUpdatesEnabled(False)

        bader = self.main.bader
        bader_plotter = self.main.bader_plotter
        structure = bader.structure

        self.metallicity.setValue(bader_plotter.atom_metallicness)

        # get types of atoms
        species = structure.symbol_set

        species_trees = {}
        species_radii = {}
        species_colors = {}
        species_visibility = {}

        # create a tree row for each atom type
        for symbol in species:
            sp = Species(symbol)

            # radius widget
            radius = sp.atomic_radius
            radius_widget = DoubleSpinBox(
                min_value=0.01,
                max_value=10.0,
                current_value=radius,
                step_size=0.01,
            )
            radius_widget.valueCommitted.connect(self.set_radii)
            species_radii[symbol] = radius_widget

            # color widget
            color = ATOM_COLORS.get(symbol, "#FFFFFF")
            color_widget = ColorPicker(initial=color)
            # color_widget.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Preferred)
            species_colors[symbol] = color_widget

            # visibility widget
            visible_widget = qw.QCheckBox()
            visible_widget.setChecked(True)
            visible_widget.setSizePolicy(
                qw.QSizePolicy.Minimum, qw.QSizePolicy.Preferred
            )
            species_visibility[symbol] = visible_widget

            # make parent row
            item = qw.QTreeWidgetItem(self.tree, [symbol])  # only string in first col
            self.tree.addTopLevelItem(item)

            # attach widgets to other columns
            self.tree.setItemWidget(item, 1, centered_widget(visible_widget))
            # self.tree.setItemWidget(item, 1, visible_widget)
            self.tree.setItemWidget(item, 2, centered_widget(radius_widget))
            self.tree.setItemWidget(item, 3, color_widget)

            species_trees[symbol] = item

        # now add per-atom rows
        self.atom_radii = []
        self.atom_colors = []
        self.atom_visibility = []

        for site in structure:
            sp = site.specie
            symbol = sp.symbol

            # radius
            radius = sp.atomic_radius
            radius_widget = DoubleSpinBox(
                min_value=0.01,
                max_value=10.0,
                current_value=radius,
                step_size=0.01,
            )
            radius_widget.valueCommitted.connect(self.set_radii)
            self.atom_radii.append(radius_widget)

            # link parent radius to child
            species_radii[symbol].valueChanged.connect(radius_widget.setValue)

            # color
            color = ATOM_COLORS.get(symbol, "#FFFFFF")
            color_widget = ColorPicker(initial=color)
            # color_widget.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Preferred)
            color_widget.colorChanged.connect(self.set_colors)
            self.atom_colors.append(color_widget)
            species_colors[symbol].colorChanged.connect(color_widget.set_color)

            # visibility
            visible_widget = qw.QCheckBox()
            visible_widget.setChecked(True)
            # visible_widget.setSizePolicy(qw.QSizePolicy.Minimum, qw.QSizePolicy.Preferred)
            visible_widget.toggled.connect(self.set_visibility)
            self.atom_visibility.append(visible_widget)
            species_visibility[symbol].toggled.connect(visible_widget.setChecked)

            # make child row under the species row
            atom_item = qw.QTreeWidgetItem(species_trees[symbol], [site.label])
            self.tree.setItemWidget(atom_item, 1, centered_widget(visible_widget))
            # self.tree.setItemWidget(atom_item, 1, visible_widget)
            self.tree.setItemWidget(atom_item, 2, centered_widget(radius_widget))
            self.tree.setItemWidget(atom_item, 3, centered_widget(color_widget))

        # enable updates
        self.tree.setUpdatesEnabled(True)
        # Make options visible
        self.layout.setCurrentIndex(1)

    def set_radii(self):
        # get radius from each atom
        radii = []
        for radius_widget in self.atom_radii:
            radii.append(radius_widget.value())
        self.main.set_property(radii, "radii")

    def set_colors(self):
        # get colors from each atom
        colors = []
        for color_widget in self.atom_colors:
            colors.append(color_widget.color())
        self.main.set_property(colors, "colors")

    def set_visibility(self):
        visible = []
        for i, visible_widget in enumerate(self.atom_visibility):
            if visible_widget.isChecked():
                visible.append(i)
        self.main.set_property(visible, "visible_atoms")

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
