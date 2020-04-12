from PyQt5.QtCore import pyqtSignal
from pyqtgraph import PlotWidget, InfiniteLine, TextItem, mkPen
import numpy as np

# A plot widget with some extra functionality for model fitting
class ModelFitWidget(PlotWidget):
    sigFitRangeChanged = pyqtSignal()

    def __init__(self, *args, **kwargs):
        PlotWidget.__init__(self, *args, **kwargs)
        self.fit_range_marker = InfiniteLine(movable=True, pen=mkPen('w', width=2))
        self.text_no_data = TextItem('Press "Analyze" to display results', anchor=(0.5, 0.5))
        self._log_mode = False
        self.showGrid(x=True, y=True)
        self.addItem(self.fit_range_marker)
        self.addItem(self.text_no_data)
        self.reset()
        self.fit_range_marker.sigPositionChangeFinished.connect(self.sigFitRangeChanged.emit)

    def reset(self):
        self.getPlotItem().setXRange(-1.0, 1.0)
        self.getPlotItem().setYRange(-1.0, 1.0)
        self.text_no_data.setVisible(True)
        self.fit_range_marker.setVisible(False)
        self.fit_range_marker.setPos(0.0)

    def setup(self):
        self.text_no_data.setVisible(False)
        self.fit_range_marker.setVisible(True)

    def get_range(self):
        if self._log_mode:
            return np.power(10.0, self.fit_range_marker.pos().x())
        else:
            return self.fit_range_marker.pos().x()

    def set_range(self, new_range : float):
        self._update_range_marker(new_range)

    def set_log(self, value : bool):
        fit_range = self.get_range()
        self._log_mode = value
        self.setLogMode(self._log_mode, False)
        self._update_range_marker(fit_range)

    def _update_range_marker(self, marker_pos):
        if self._log_mode:
            if marker_pos <= 0.0:
                marker_pos = 1.0
            self.fit_range_marker.setPos(np.log10(marker_pos))
        else:
            self.fit_range_marker.setPos(marker_pos)
        self.sigFitRangeChanged.emit()