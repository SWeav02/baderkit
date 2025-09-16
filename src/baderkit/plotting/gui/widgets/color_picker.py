# -*- coding: utf-8 -*-

from qtpy import QtCore, QtGui
from qtpy import QtWidgets as qw


class ColorPicker(qw.QWidget):
    colorChanged = QtCore.Signal(str)  # custom signal

    def __init__(
        self,
        initial="#BA8E23",
        parent=None,
        plot_prop: str = None,
        main: object = None,
    ):
        super().__init__(parent)

        # button with background showing current color
        self.button = qw.QPushButton()
        self.button.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Preferred)
        self.button.setMinimumHeight(16)
        self.button.clicked.connect(self.choose_color)

        self.set_color(initial)

        self.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Preferred)

        layout = qw.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)

        if main is not None:
            self.colorChanged.connect(main.set_property)
            self.plot_prop = plot_prop

    def set_color(self, color: str):
        """Set the current color and update the button style."""
        self._color = QtGui.QColor(color)
        if self.isEnabled():
            self.button.setStyleSheet(
                f"background-color: {self._color.name()}; border: 1px solid gray;"
            )
        else:
            self.button.setStyleSheet("")  # let Qt grey it out
        self.colorChanged.emit(self._color.name())

    def color(self):
        return self._color.name()

    def choose_color(self):
        """Open the QColorDialog and update if user picks a color."""
        color = qw.QColorDialog.getColor(self._color, self, "Select Color")
        if color.isValid():
            self.set_color(color.name())

    def setEnabled(self, enabled: bool):
        """Override setEnabled so button updates correctly."""
        super().setEnabled(enabled)
        self.button.setEnabled(enabled)
        # Update style depending on enabled state
        if enabled:
            self.set_color(self._color.name())
        else:
            self.button.setStyleSheet("")  # reset -> Qt greys out
