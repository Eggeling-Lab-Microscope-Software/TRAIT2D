from iscat_lib.analysis import Track
from iscat_lib.exceptions import *
import sys
import pyqtgraph as pg
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout, QWidget, QDialog
from PyQt5.QtCore import Qt, QRectF

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Prettier math font
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'

# A plot widget with some extra functionality for model fitting
class ModelFitWidget(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        pg.PlotWidget.__init__(self, *args, **kwargs)
        self.fit_range_marker = pg.InfiniteLine(movable=True, pen=pg.mkPen('w', width=2))
        self._log_mode = False
        self.addItem(self.fit_range_marker)

    def reset(self):
        self.clear()
        self.addItem(self.fit_range_marker)

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

class MathTextLabel(QtWidgets.QWidget):

    def __init__(self, mathText, parent=None, **kwargs):
        super(QtWidgets.QWidget, self).__init__(parent, **kwargs)

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

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi('gui_analysis.ui', self)

        self.track = None

        self.pushButtonLoadTrack.clicked.connect(self.load_trajectory_dialog)
        self.pushButtonAnalyzeMSD.clicked.connect(self.analyze_msd)
        self.checkBoxLogPlotMSD.stateChanged.connect(self.plotMSD.set_log)
        # TODO
        self.pushButtonClipboardMSD.clicked.connect(self.not_implemented)
        self.pushButtonFormula_1.clicked.connect(self.show_formula_model_1)
        self.pushButtonFormula_2.clicked.connect(self.show_formula_model_2)

    def show_trackid_dialog(self):
        Dialog = QtWidgets.QDialog()
        uic.loadUi("dialog_trackid.ui", Dialog)
        Dialog.show()
        resp = Dialog.exec_()

        if resp == QtWidgets.QDialog.Accepted:
            return Dialog.spinBoxId.value()

    def load_trajectory(self, filename : str, id=None):
        try:
            self.track = Track.from_file(filename, id=id)
        except Exception as inst:
            if type(inst) == LoadTrackMissingIdError:
                id = self.show_trackid_dialog()
                self.load_trajectory(filename=filename, id=id)
            if type(inst) == LoadTrackIdNotFoundError:
                mb = QMessageBox()
                mb.setText("There is no track with that ID!")
                mb.setWindowTitle("Error")
                mb.setIcon(QMessageBox.Critical)
                mb.exec()

        if self.track==None:
            return

        self.plotMSD.reset()

        self.statusbar.showMessage(
            "Loaded track of length {}".format(self.track.get_size()))      

    def load_trajectory_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load track file", "", "Track files (*.csv)")
        if filename == '':
            return
        self.load_trajectory(filename, id=None)

    def analyze_msd(self):
        if self.track == None:
            self.statusbar.showMessage("Load a track first!")
            mb = QMessageBox()
            mb.setText("Load a track first!")
            mb.setWindowTitle("Error")
            mb.setIcon(QMessageBox.Warning)
            mb.exec()
            return

        T = self.track.get_t()[0:-3]
        fitPoints = np.argwhere(T >= self.plotMSD.get_range())[0]
        if fitPoints <= 0:
            fitPoints = None
        results = self.track.msd_analysis(nFitPoints=fitPoints)["results"]

        # Show results for model 1 in GUI
        self.lineEditParam1_1.setText(
            "{:5e}".format(results["model1"]["params"][0]))
        self.lineEditParam2_1.setText(
            "{:5e}".format(results["model1"]["params"][1]))
        self.lineEditRelLikelihood_1.setText(
            "{:5f}".format(results["model1"]["rel_likelihood"]))
        self.lineEditBIC_1.setText("{:5f}".format(results["model1"]["BIC"]))

        # Show results for model 2 in GUI
        self.lineEditParam1_2.setText(
            "{:5e}".format(results["model2"]["params"][0]))
        self.lineEditParam2_2.setText(
            "{:5e}".format(results["model2"]["params"][1]))
        self.lineEditParam3_2.setText(
            "{:5e}".format(results["model2"]["params"][2]))
        self.lineEditRelLikelihood_2.setText(
            "{:5f}".format(results["model2"]["rel_likelihood"]))
        self.lineEditBIC_2.setText("{:5f}".format(results["model2"]["BIC"]))

        # Plot analysis results
        def model1(t, D, delta2): return 4 * D * t + 2 * delta2
        def model2(t, D, delta2, alpha): return 4 * D * t**alpha + 2 * delta2

        n_points = results["n_points"]
        MSD = self.track.get_msd()
        reg1 = results["model1"]["params"]
        reg2 = results["model2"]["params"]
        m1 = model1(T, *reg1)
        m2 = model2(T, *reg2)

        self.plotMSD.reset()
        self.plotMSD.addLegend()
        self.plotMSD.plot(T, MSD, name='MSD')
        self.plotMSD.plot(T[0:n_points], m1[0:n_points],
                          pen=(1, 3), name='Model 1')
        self.plotMSD.plot(T[0:n_points], m2[0:n_points],
                          pen=(2, 3), name='Model 2')

        self.plotMSD.set_range(T[n_points])

    def show_formula_model_1(self):
        mathText = r'$\mathrm{MSD}(t_i) = 4 \cdot D \cdot t_i + 2 \delta^2$'
        mb = MathTextLabel(mathText)
        mb.setFocus(True)
        mb.setWindowModality(Qt.ApplicationModal)
        mb.show()

    def show_formula_model_2(self):
        mathText = r'$\mathrm{MSD}(t_i) = 4 \cdot D \cdot \left( \dfrac{t_i}{\tau} \right)^\alpha + 2 \delta^2$'
        mb = MathTextLabel(mathText)
        mb.setFocus(True)
        mb.setWindowModality(Qt.ApplicationModal)
        mb.show()

    def not_implemented(self):
        mb = QMessageBox()
        mb.setText("Not implemented")
        mb.exec()


def main():
    app = QtWidgets.QApplication(sys.argv)

    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
