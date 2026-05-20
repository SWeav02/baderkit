# -*- coding: utf-8 -*-

from qtpy.QtWidgets import QDialog, QLabel, QVBoxLayout


class ErrorWindow(QDialog):
    def __init__(
        self,
        parent=None,
        message: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Warning!")
        layout = QVBoxLayout()
        message = QLabel(message)
        layout.addWidget(message)
        self.setLayout(layout)
        self.exec()
