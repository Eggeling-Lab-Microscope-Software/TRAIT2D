from iscat_lib.analysis import Track
from iscat_lib.exceptions import *

import numpy as np

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QWidget, QApplication

from gui.plot import ModelFitWidget
from gui.render_math import MathTextLabel

class widgetSD(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        uic.loadUi('gui/tab/sd.ui', self)

        self.parent = parent

        self.pushButtonAnalyze.clicked.connect(self.analyze)
        self.checkBoxLogPlot.stateChanged.connect(self.plot.set_log)
        self.pushButtonFormula_1.clicked.connect(self.show_formula_model_1)
        self.pushButtonFormula_2.clicked.connect(self.show_formula_model_2)
        self.pushButtonFormula_3.clicked.connect(self.show_formula_model_3)
        self.pushButtonClipboard.clicked.connect(self.results_to_clipboard)
        self.pushButtonSetRange.clicked.connect(self.set_range_from_spinbox)

        self.parent.sigTrackLoaded.connect(self.reset)
        self.plot.sigFitRangeChanged.connect(self.on_range_changed)

        self.plot.setLabel('left', "Dapp")
        self.plot.setLabel('bottom', "T", units="s")

        self.legend = self.plot.addLegend()
        self.plot_dapp = self.plot.plot([], name='Dapp')
        self.plot_brownian = self.plot.plot([], pen=(1, 3), name = 'Brownian')
        self.plot_confined = self.plot.plot([], pen=(2, 3), name = 'Confined')
        self.plot_hopping  = self.plot.plot([], pen=(3, 3), name = 'Hopping')

    def results_to_clipboard(self):
        if self.parent.track == None or not self.parent.track.get_sd_analysis_results()["analyzed"]:
            self.parent.statusbar.showMessage("Load a track first!")
            mb = QMessageBox()
            mb.setText("Analyze a track first!")
            mb.setWindowTitle("Error")
            mb.setIcon(QMessageBox.Warning)
            mb.exec()
            return

        QApplication.clipboard().setText(str(self.parent.track.get_sd_analysis_results()["results"]))

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

        self.lineEditParam1_3.setText("")
        self.lineEditParam2_3.setText("")
        self.lineEditParam3_3.setText("")
        self.lineEditParam4_3.setText("")
        self.lineEditRelLikelihood_3.setText("")
        self.lineEditBIC_3.setText("")

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

        dt = T[1] - T[0]
        J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

        fit_max_time = self.plot.get_range()
        if fit_max_time <= 0.0:
            fit_max_time = None

        maxfev = int(self.lineEditMaxIt.text())

        initial_guesses = {"brownian" : [0.0, 0.0], "confined" : [0.0, 0.0, 0.0], "hop" : [0.0, 0.0, 0.0, 0.0]}

        if self.checkBoxUseInitial.checkState():
            if (self.lineEditParam1_1.text() != ""):
                initial_guesses["brownian"][0] = float(self.lineEditParam1_1.text())
            if (self.lineEditParam2_1.text() != ""):
                initial_guesses["brownian"][1] = float(self.lineEditParam2_1.text())


            if (self.lineEditParam1_2.text() != ""):
                initial_guesses["confined"][0] = float(self.lineEditParam1_2.text())
            if (self.lineEditParam2_2.text() != ""):
                initial_guesses["confined"][1] = float(self.lineEditParam2_2.text())
            if (self.lineEditParam3_2.text() != ""):
                initial_guesses["confined"][2] = float(self.lineEditParam3_2.text())

            if (self.lineEditParam1_3.text() != ""):
                initial_guesses["hop"][0] = float(self.lineEditParam1_3.text())
            if (self.lineEditParam2_3.text() != ""):
                initial_guesses["hop"][1] = float(self.lineEditParam2_3.text())
            if (self.lineEditParam3_3.text() != ""):
                initial_guesses["hop"][2] = float(self.lineEditParam3_3.text())
            if (self.lineEditParam4_3.text() != ""):
                initial_guesses["hop"][3] = float(self.lineEditParam4_3.text())

        try:
            results = self.parent.track.sd_analysis(fit_max_time=fit_max_time, initial_guesses = initial_guesses, maxfev=maxfev)["results"]
        except RuntimeError:
            mb = QMessageBox()
            mb.setText("A model fit failed! Try raising the maximum iterations or different initial values.")
            mb.setWindowTitle("Fit Error")
            mb.setIcon(QMessageBox.Warning)
            mb.exec()
            return

        # Show results for brownian model in GUI
        self.lineEditParam1_1.setText("{:5e}".format(results["brownian"]["params"][0]))
        self.lineEditParam1Error_1.setText("{:5e}".format(results["brownian"]["errors"][0]))
        self.lineEditParam2_1.setText("{:5e}".format(results["brownian"]["params"][1]))
        self.lineEditParam2Error_1.setText("{:5e}".format(results["brownian"]["errors"][1]))
        self.lineEditRelLikelihood_1.setText("{:5f}".format(results["brownian"]["rel_likelihood"]))
        self.lineEditBIC_1.setText("{:5f}".format(results["brownian"]["bic"]))

        # Show results for confined model in GUI
        self.lineEditParam1_2.setText("{:5e}".format(results["confined"]["params"][0]))
        self.lineEditParam1Error_2.setText("{:5e}".format(results["confined"]["errors"][0]))
        self.lineEditParam2_2.setText("{:5e}".format(results["confined"]["params"][1]))
        self.lineEditParam2Error_2.setText("{:5e}".format(results["confined"]["errors"][1]))
        self.lineEditParam3_2.setText("{:5e}".format(results["confined"]["params"][2]))
        self.lineEditParam3Error_2.setText("{:5e}".format(results["confined"]["errors"][2]))
        self.lineEditRelLikelihood_2.setText("{:5f}".format(results["confined"]["rel_likelihood"]))
        self.lineEditBIC_2.setText("{:5f}".format(results["confined"]["bic"]))

        # Show results for hopping in GUI
        self.lineEditParam1_3.setText("{:5e}".format(results["hop"]["params"][0]))
        self.lineEditParam1Error_3.setText("{:5e}".format(results["hop"]["errors"][0]))
        self.lineEditParam2_3.setText("{:5e}".format(results["hop"]["params"][1]))
        self.lineEditParam2Error_3.setText("{:5e}".format(results["hop"]["errors"][1]))
        self.lineEditParam3_3.setText("{:5e}".format(results["hop"]["params"][2]))
        self.lineEditParam3Error_3.setText("{:5e}".format(results["hop"]["errors"][2]))
        self.lineEditParam4_3.setText("{:5e}".format(results["hop"]["params"][3]))
        self.lineEditParam4Error_3.setText("{:5e}".format(results["hop"]["errors"][3]))
        self.lineEditRelLikelihood_3.setText("{:5f}".format(results["hop"]["rel_likelihood"]))
        self.lineEditBIC_3.setText("{:5f}".format(results["hop"]["bic"]))

        # Define Models
        R = float(self.doubleSpinBoxInputParam1.value())
        def model_brownian(t, D, delta): return D + \
            delta**2 / (2 * t * (1 - 2*R*dt/t))
        def model_confined(t, D_micro, delta, tau): return D_micro * (tau/t) * \
            (1 - np.exp(-tau/t)) + delta ** 2 / (2 * t * (1 - 2 * R * dt / t))
        def model_hop(t, D_macro, D_micro, delta, tau): return D_macro + D_micro * \
            (tau/t) * (1 - np.exp(-tau/t)) + \
            delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        n_points = results["n_points"]
        Dapp = self.parent.track.get_sd_analysis_results()["Dapp"]
        reg1 = results["brownian"]["params"]
        reg2 = results["confined"]["params"]
        reg3 = results["hop"]["params"]
        m1 = model_brownian(T, *reg1)
        m2 = model_confined(T, *reg2)
        m3 = model_hop(T, *reg3)       

        T_dapp = self.parent.track.get_sd_analysis_results()["J"] * dt

        n_point_t = None
        if fit_max_time is not None:
            n_points_t = int(np.argwhere(T < fit_max_time)[-1])
        else:
            n_points_t = int(0.25 * T[0:J[-1]].size)

        self.plot.reset()
        self.plot.setup()

        self.plot_dapp.setData(T_dapp, Dapp)
        self.plot_brownian.setData(T[0:n_points_t], m1[0:n_points_t])
        self.plot_confined.setData(T[0:n_points_t], m2[0:n_points_t])
        self.plot_hopping.setData(T[0:n_points_t], m3[0:n_points_t])

        self.plot.set_range(T[n_points_t])
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

    def on_range_changed(self):
        self.doubleSpinBoxRange.setValue(self.plot.get_range() * 1000.0)

    def set_range_from_spinbox(self):
        self.plot.set_range(self.doubleSpinBoxRange.value() / 1000.0)
