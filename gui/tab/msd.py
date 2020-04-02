from iscat_lib.analysis import Track
from iscat_lib.exceptions import *

import numpy as np

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QWidget

from gui.plot import ModelFitWidget
from gui.render_math import MathTextLabel

class widgetMSD(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        uic.loadUi('gui/tab/msd.ui', self)

        self.parent = parent

        self.pushButtonAnalyze.clicked.connect(self.analyze)
        self.checkBoxLogPlot.stateChanged.connect(self.plot.set_log)
        self.pushButtonFormula_1.clicked.connect(self.show_formula_model_1)
        self.pushButtonFormula_2.clicked.connect(self.show_formula_model_2)
        # TODO: Clipboard

        self.parent.sigTrackLoaded.connect(self.reset)

        # Set plot labels
        self.plot.setLabel('left', "MSD", units="m")
        self.plot.setLabel('bottom', "T", units="s")


    def reset(self):
        self.lineEditParam1_1.setText("")
        self.lineEditParam2_1.setText("")
        self.lineEditRelLikelihood_1.setText("")
        self.lineEditBIC_1.setText("")

        self.lineEditParam1_2.setText("")
        self.lineEditParam2_2.setText("")
        self.lineEditParam3_2.setText("")
        self.lineEditRelLikelihood_2.setText("")
        self.lineEditBIC_2.setText("")

        self.plot.reset()

    def analyze(self):
        if self.parent.track == None:
            self.parent.statusbar.showMessage("Load a track first!")
            mb = QMessageBox()
            mb.setText("Load a track first!")
            mb.setWindowTitle("Error")
            mb.setIcon(QMessageBox.Warning)
            mb.exec()
            return

        T = self.parent.track.get_t()[0:-3]
        fitPoints = np.argwhere(T >= self.plot.get_range())[0]
        if fitPoints <= 0:
            fitPoints = None
        results = self.parent.track.msd_analysis(nFitPoints=fitPoints)["results"]

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
        MSD = self.parent.track.get_msd()
        reg1 = results["model1"]["params"]
        reg2 = results["model2"]["params"]
        m1 = model1(T, *reg1)
        m2 = model2(T, *reg2)

        self.plot.reset()
        self.plot.setup()
        self.plot.addLegend()
        self.plot.plot(T, MSD, name='MSD')
        self.plot.plot(T[0:n_points], m1[0:n_points],
                          pen=(1, 2), name='Model 1')
        self.plot.plot(T[0:n_points], m2[0:n_points],
                          pen=(2, 2), name='Model 2')

        self.plot.set_range(T[n_points])
        self.plot.autoRange()


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