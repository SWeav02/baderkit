# -*- coding: utf-8 -*-
import qtpy.QtWidgets as qw
from qtpy.QtCore import Qt


def centered_widget(widget: qw.QWidget) -> qw.QWidget:
    container = qw.QWidget()
    layout = qw.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    # Check the widget's horizontal size policy
    if widget.sizePolicy().horizontalPolicy() in (
        qw.QSizePolicy.Expanding,
        qw.QSizePolicy.MinimumExpanding,
    ):
        # Let it expand and center vertically
        layout.addWidget(widget, stretch=1, alignment=Qt.AlignVCenter)
    else:
        # Keep it centered both ways
        layout.addWidget(widget, alignment=Qt.AlignCenter)

    if isinstance(widget, qw.QCheckBox):
        widget.setStyleSheet("margin: 0px; padding: 0px;")

    return container
