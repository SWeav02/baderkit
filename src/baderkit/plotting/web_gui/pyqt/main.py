# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 08:09:10 2025

@author: sammw
"""

import sys
import importlib.resources
# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets as qw
from qtpy.QtCore import Qt

import numpy as np

import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow as pvMainWindow

from baderkit.plotting.web_gui.pyqt.tabs import BaderTab, BasinTab, SurfaceTab, AtomsTab
from baderkit.plotting.core import BaderPlotter

class MainWindow(pvMainWindow):

    def __init__(self, parent=None, show=True):
        qw.QMainWindow.__init__(self, parent)

        # create blank plotter for now
        self.bader_plotter = None
        self.bader = None
        # Set up viewport for plotter (right)
        self.viewport = qw.QFrame()
        # give the frame a layout
        self.viewport_layout = qw.QStackedLayout()
        # add a helpful label
        self.viewport_label = qw.QLabel("Bader has not yet run")
        self.viewport_label.setAlignment(Qt.AlignCenter)
        self.viewport_layout.addWidget(self.viewport_label)
        # set layout
        self.viewport.setLayout(self.viewport_layout)

        
        # Sidebar (left)
        self.sidebar = qw.QTabWidget()
        for tab in [BaderTab, BasinTab, SurfaceTab, AtomsTab]:
            new_tab = tab(self)
            self.sidebar.addTab(new_tab, new_tab.name)
        
        self.splitter = qw.QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.viewport)
        
        # # Optional: set initial sizes
        self.splitter.setSizes([150, 450])  # sidebar width, main area width

        self.setWindowTitle("BaderKit")
        # self.resize(600, 400)

        self.setCentralWidget(self.splitter)

        if show:
            self.show()
            
        # TODO:
            # Summary tab. List results for basins/atoms somewhere.
            
            # View tab. Lattice border. Camera angle.
            
            # Export tab. Update export methods?

        
    def set_bader(self, bader):
        self.bader = bader
    
        # --- remove all children of the viewport layout safely ---
        while self.viewport_layout.count():
            item = self.viewport_layout.takeAt(0)
            if item is None:
                break
            w = item.widget()
            if w is not None:
                # detach then delete
                w.setParent(None)
                w.deleteLater()
    
        # --- create container that the interactor will live in ---
        container = qw.QWidget(self.viewport)   # parent to viewport to ensure proper stacking
        container.setContentsMargins(0, 0, 0, 0)
        cont_layout = qw.QVBoxLayout(container)
        cont_layout.setContentsMargins(0, 0, 0, 0)
        cont_layout.setSpacing(0)
    
        container.setSizePolicy(qw.QSizePolicy.Expanding,
                                qw.QSizePolicy.Expanding)
    
        # --- create plotter, ask it to attach to the container ---
        # otherwise reparent the interactor widget below.
        self.bader_plotter = BaderPlotter(bader, qt_plotter=True, qt_frame=container)
    
        # get the actual QWidget that must be inserted
        plot_widget = self.bader_plotter.plotter.interactor
    
        # sometimes pyvistaqt creates it with a different parent; force parent to container
        plot_widget.setParent(container)
        plot_widget.setContentsMargins(0, 0, 0, 0)
        plot_widget.setSizePolicy(qw.QSizePolicy.Expanding,
                                  qw.QSizePolicy.Expanding)
    
        # add the interactor to the container layout
        cont_layout.addWidget(plot_widget)
    
        # add the container into your viewport layout
        self.viewport_layout.addWidget(container)
    
        # tell layouts and widgets to recalculate sizes
        container.show()
        plot_widget.show()
        container.updateGeometry()
        self.viewport_layout.invalidate()
        self.viewport.update()
    
        # ensure proper stretching (make the container take available space)
        idx = self.viewport_layout.indexOf(container)
        if idx != -1:
            try:
                self.viewport_layout.setStretch(idx, 1)
            except Exception:
                # some layout types use setStretch / setStretchFactor; ignore if not available
                pass
    
        # final render / camera reset if you want
        try:
            self.bader_plotter.plotter.reset_camera()
            self.bader_plotter.plotter.render()
        except Exception:
            pass
        
        # connect close signal
        self.signal_close.connect(self.bader_plotter.plotter.close)
    
        # update sidebar tabs
        for i in range(self.sidebar.count()):
            page = self.sidebar.widget(i)
            page.set_bader()
    
    def set_property(self, new_value, prop: str = None, **kwargs):
        if prop is None:
            sender = self.sender()
            prop = sender.plot_prop
        # set property
        self.bader_plotter.plotter.suppress_rendering = True
        setattr(self.bader_plotter, prop, new_value)
        self.bader_plotter.plotter.suppress_rendering = False

def run_app():
    app = qw.QApplication(sys.argv)
    with importlib.resources.open_text("baderkit.plotting.web_gui.pyqt","stylesheet.qss") as f:
        app.setStyleSheet(f.read())
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
