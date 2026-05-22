# -*- coding: utf-8 -*-

from qtpy import QtCore
from qtpy.QtWidgets import QSpinBox


class SpinBox(QSpinBox):
    valueCommitted = QtCore.Signal(float)  # custom signal with value

    def __init__(
        self,
        parent=None,
        min_value: int = 0,
        max_value: int = 100,
        current_value: int = 0,
        step_size: int = 1,
        plot_prop: str = None,
        main: str = None,
    ):
        super().__init__(parent)
        self.setRange(min_value, max_value)
        self.setSingleStep(step_size)  # increment step
        self.setValue(current_value)  # default value
        self.plot_prop = plot_prop  # Used when connected to main
        self.editingFinished.connect(self._emit_value)
        if main is not None:
            self.valueCommitted.connect(main.set_property)

    def _emit_value(self):
        self.valueCommitted.emit(self.value())
