from iscat_lib.analysis import Track
from iscat_lib.exceptions import *

import numpy as np

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QWidget

from gui.plot import ModelFitWidget
from gui.render_math import MathTextLabel

class widgetADC(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        uic.loadUi('gui/tab/adc.ui', self)

        self.parent = parent

        self.pushButtonAnalyze.clicked.connect(self.analyze)
        self.checkBoxLogPlot.stateChanged.connect(self.plot.set_log)
        self.pushButtonFormula_1.clicked.connect(self.show_formula_model_1)
        self.pushButtonFormula_2.clicked.connect(self.show_formula_model_2)
        self.pushButtonFormula_3.clicked.connect(self.show_formula_model_3)

        self.parent.sigTrackLoaded.connect(self.plot.reset)

        self.plot.setLabel('left', "Dapp")
        self.plot.setLabel('bottom', "T", units="s")

    def analyze(self):
        if self.parent.track == None:
            self.parent.statusbar.showMessage("Load a track first!")
            mb = QMessageBox()
            mb.setText("Load a track first!")
            mb.setWindowTitle("Error")
            mb.setIcon(QMessageBox.Warning)
            mb.exec()
            return

        results = self.parent.track.adc_analysis()

        T = self.parent.track.get_t()[0:-3]
        fracFitPoints = self.plot.get_range() / T[-1]
        if fracFitPoints <= 0:
            fracFitPoints = 0.25
        results = self.parent.track.adc_analysis(fractionFitPoints=fracFitPoints)["results"]

        self.lineEditParam1_1.setText(
            "{:5e}".format(results["brownian"]["params"][0]))
        self.lineEditParam2_1.setText(
            "{:5e}".format(results["brownian"]["params"][1]))
        self.lineEditRelLikelihood_1.setText(
            "{:5f}".format(results["brownian"]["rel_likelihood"]))
        self.lineEditBIC_1.setText("{:5f}".format(results["brownian"]["bic"]))

        self.lineEditParam1_2.setText(
            "{:5e}".format(results["confined"]["params"][0]))
        self.lineEditParam2_2.setText(
            "{:5e}".format(results["confined"]["params"][1]))
        self.lineEditParam3_2.setText(
            "{:5e}".format(results["confined"]["params"][2]))
        self.lineEditRelLikelihood_2.setText(
            "{:5f}".format(results["confined"]["rel_likelihood"]))
        self.lineEditBIC_2.setText("{:5f}".format(results["confined"]["bic"]))

        self.lineEditParam1_3.setText(
            "{:5e}".format(results["hop"]["params"][0]))
        self.lineEditParam2_3.setText(
            "{:5e}".format(results["hop"]["params"][1]))
        self.lineEditParam3_3.setText(
            "{:5e}".format(results["hop"]["params"][2]))
        self.lineEditParam4_3.setText(
            "{:5e}".format(results["hop"]["params"][3]))
        self.lineEditRelLikelihood_3.setText(
            "{:5f}".format(results["hop"]["rel_likelihood"]))
        self.lineEditBIC_3.setText("{:5f}".format(results["hop"]["bic"]))

        # Modelle definieren
        R = 1/6
        dt = T[1] - T[0]
        def model_brownian(t, D, delta): return D + \
            delta**2 / (2 * t * (1 - 2*R*dt/t))
        def model_confined(t, D_micro, delta, tau): return D_micro * (tau/t) * \
            (1 - np.exp(-tau/t)) + delta ** 2 / (2 * t * (1 - 2 * R * dt / t))
        def model_hop(t, D_macro, D_micro, delta, tau): return D_macro + D_micro * \
            (tau/t) * (1 - np.exp(-tau/t)) + \
            delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        n_points = results["n_points"]
        Dapp = self.parent.track.get_adc_analysis_results()["Dapp"]
        reg1 = results["brownian"]["params"]
        reg2 = results["confined"]["params"]
        reg3 = results["hop"]["params"]
        m1 = model_brownian(T, *reg1)
        m2 = model_confined(T, *reg2)
        m3 = model_hop(T, *reg3)       

        self.plot.reset()
        self.plot.setup()
        self.plot.addLegend()
        self.plot.plot(T, Dapp, name='Dapp')
        self.plot.plot(T[0:n_points], m1[0:n_points], pen=(1, 3), name = 'Brownian')
        self.plot.plot(T[0:n_points], m2[0:n_points], pen=(2, 3), name = 'Confined')
        self.plot.plot(T[0:n_points], m3[0:n_points], pen=(3, 3), name = 'Hopping')

        self.plot.set_range(T[n_points])
        self.plot.autoRange()     

    def show_formula_model_1(self):
        mathText = r'$todo$'
        mb = MathTextLabel(mathText)
        mb.setFocus(True)
        mb.setWindowModality(Qt.ApplicationModal)
        mb.show()

    def show_formula_model_2(self):
        mathText = r'$todo$'
        mb = MathTextLabel(mathText)
        mb.setFocus(True)
        mb.setWindowModality(Qt.ApplicationModal)
        mb.show()

    def show_formula_model_3(self):
        mathText = r'$todo$'
        mb = MathTextLabel(mathText)
        mb.setFocus(True)
        mb.setWindowModality(Qt.ApplicationModal)
        mb.show()