# -*- coding: utf-8 -*-

from qtpy import QtCore as qc
from qtpy import QtWidgets as qw


class FilePicker(qw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.line_edit = qw.QLineEdit()
        self.browse_btn = qw.QPushButton("Browse...")

        self.browse_btn.clicked.connect(self.browse)

        layout = qw.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_btn)

        # Make the widget expand horizontally but keep a single-line fixed height
        self.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Fixed)
        self.line_edit.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Preferred)
        self.browse_btn.setSizePolicy(qw.QSizePolicy.Minimum, qw.QSizePolicy.Preferred)

        # Compute a single-line height based on children size hints (keeps baseline alignment)
        h = max(self.line_edit.sizeHint().height(), self.browse_btn.sizeHint().height())
        # add some small padding to be safe
        self._single_line_height = h
        self.setFixedHeight(self._single_line_height)

    def browse(self):
        file_path, _ = qw.QFileDialog.getOpenFileName(self, "Select a file")
        if file_path:
            self.line_edit.setText(file_path)

    def file_path(self):
        return self.line_edit.text()

    # Optional: return a sensible sizeHint (helps some layouts)
    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(qc.QSize(sh.width(), self._single_line_height))
