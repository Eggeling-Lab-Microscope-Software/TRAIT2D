from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5 import QtGui

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Prettier math font
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'

class MathTextLabel(QWidget):

    def __init__(self, mathText, parent=None, **kwargs):
        super(QWidget, self).__init__(parent, **kwargs)

        l = QVBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)

        r, g, b, a = self.palette().base().color().getRgbF()

        self._figure = Figure(edgecolor=(r, g, b), facecolor=(r, g, b))
        self._figure.clear()
        text = self._figure.suptitle(
            mathText,
            x=0.0,
            y=1.0,
            horizontalalignment='left',
            verticalalignment='top',
            size=QtGui.QFont().pointSize()*1.5
        )
        self._canvas = FigureCanvas(self._figure)
        self._canvas.draw()
        l.addWidget(self._canvas)

        (x0, y0), (x1, y1) = text.get_window_extent().get_points()
        w = int(x1-x0)
        h = int(y1-y0)

        self._figure.set_size_inches(w/80, h/80)
        self.setFixedSize(w, h)