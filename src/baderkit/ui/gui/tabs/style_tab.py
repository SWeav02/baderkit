# -*- coding: utf-8 -*-

from qtpy import QtWidgets as qw

from .atom_tab import AtomsTab
from .surface_tab import SurfaceTab
from .view_tab import ViewTab


class StyleTab(qw.QWidget):

    def __init__(self, main, parent=None):
        super().__init__(parent)

        self.main = main
        self.name = "Style"

        # add tabs
        self.layout = qw.QVBoxLayout()
        tab_box = qw.QTabWidget()
        self.layout.addWidget(tab_box)
        self.tabs = []
        for tab in [SurfaceTab, AtomsTab, ViewTab]:
            new_tab = tab(self.main)
            self.tabs.append(new_tab)
            tab_box.addTab(new_tab, new_tab.name)
        self.setLayout(self.layout)  # attach it to this QWidget

    def set_bader(self):
        # update each dropdown
        for tab in self.tabs:
            tab.set_bader()
