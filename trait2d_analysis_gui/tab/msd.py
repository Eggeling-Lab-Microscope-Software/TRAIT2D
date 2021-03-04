from trait2d.analysis import Track
from trait2d.exceptions import *

import numpy as np

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QWidget, QApplication
from PyQt5.QtGui import QIntValidator

from trait2d_analysis_gui.plot import ModelFitWidget
from trait2d_analysis_gui.render_math import MathTextLabel

import os

class widgetMSD(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        
        dirname = os.path.dirname(__file__)
        uic.loadUi(os.path.join(dirname, 'msd.ui'), self)

        self.parent = parent

        self.pushButtonAnalyze.clicked.connect(self.analyze)
        self.checkBoxLogPlot.stateChanged.connect(self.plot.set_log)
        self.pushButtonClipboard.clicked.connect(self.results_to_clipboard)
        self.pushButtonSetRange.clicked.connect(self.set_range_from_spinbox)

        self.lineEditMaxIt.setValidator(QIntValidator())

        self.parent.sigTrackLoaded.connect(self.reset)
        self.plot.sigFitRangeChanged.connect(self.on_range_changed)

        # Set plot labels
        self.plot.setLabel('left', "MSD", units="m")
        self.plot.setLabel('bottom', "T", units="s")

        self.plot_msd = self.plot.plot([], name='MSD')
        self.plot_model1 = self.plot.plot([],
                          pen=(1, 2), name='Model 1')
        self.plot_model2 = self.plot.plot([],
                          pen=(2, 2), name='Model 2')

    def results_to_clipboard(self):
        if self.parent.track is None or self.parent.track.get_msd_analysis_results() is None:
            self.parent.statusbar.showMessage("Analyze a track first!")
            mb = QMessageBox()
            mb.setText("Analyze a track first!")
            mb.setWindowTitle("Error")
            mb.setIcon(QMessageBox.Warning)
            mb.exec()
            return

        QApplication.clipboard().setText(str(self.parent.track.get_msd_analysis_results()["fit_results"]))

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

        fit_max_time = self.plot.get_range()
        if fit_max_time <= 0.0:
            fit_max_time = None

        maxfev = int(self.lineEditMaxIt.text())

        initial_guesses = {"model1" : [0.0, 0.0], "model2" : [0.0, 0.0, 0.0]}

        if self.checkBoxUseInitial.checkState():
            if (self.lineEditParam1_1.text() != ""):
                initial_guesses["model1"][0] = float(self.lineEditParam1_1.text())
            if (self.lineEditParam2_1.text() != ""):
                initial_guesses["model1"][1] = float(self.lineEditParam2_1.text())


            if (self.lineEditParam1_2.text() != ""):
                initial_guesses["model2"][0] = float(self.lineEditParam1_2.text())
            if (self.lineEditParam2_2.text() != ""):
                initial_guesses["model2"][1] = float(self.lineEditParam2_2.text())
            if (self.lineEditParam3_2.text() != ""):
                initial_guesses["model2"][2] = float(self.lineEditParam3_2.text())

        R = float(self.doubleSpinBoxInputParam1.value())

        try:
            results = self.parent.track.msd_analysis(fit_max_time=fit_max_time, initial_guesses=initial_guesses, maxfev=maxfev, R=R)
        except RuntimeError:
            mb = QMessageBox()
            mb.setText("A model fit failed! Try raising the maximum iterations or different initial values.")
            mb.setWindowTitle("Fit Error")
            mb.setIcon(QMessageBox.Warning)
            mb.exec()
            return

        fit_results = results["fit_results"]

        # Show fit results for model 1 in GUI
        self.lineEditParam1_1.setText("{:5e}".format(fit_results["model1"]["params"][0]))
        self.lineEditParam1Error_1.setText("{:5e}".format(fit_results["model1"]["errors"][0]))
        self.lineEditParam2_1.setText("{:5e}".format(fit_results["model1"]["params"][1]))
        self.lineEditParam2Error_1.setText("{:5e}".format(fit_results["model1"]["errors"][1]))
        self.lineEditRelLikelihood_1.setText("{:5f}".format(fit_results["model1"]["rel_likelihood"]))
        self.lineEditBIC_1.setText("{:5f}".format(fit_results["model1"]["bic"]))

        # Show fit results for model 2 in GUI
        self.lineEditParam1_2.setText("{:5e}".format(fit_results["model2"]["params"][0]))
        self.lineEditParam1Error_2.setText("{:5e}".format(fit_results["model2"]["errors"][0]))
        self.lineEditParam2_2.setText("{:5e}".format(fit_results["model2"]["params"][1]))
        self.lineEditParam2Error_2.setText("{:5e}".format(fit_results["model2"]["errors"][1]))
        self.lineEditParam3_2.setText("{:5e}".format(fit_results["model2"]["params"][2]))
        self.lineEditParam3Error_2.setText("{:5e}".format(fit_results["model2"]["errors"][2]))
        self.lineEditRelLikelihood_2.setText("{:5f}".format(fit_results["model2"]["rel_likelihood"]))
        self.lineEditBIC_2.setText("{:5f}".format(fit_results["model2"]["bic"]))

        # Plot analysis results
        def model1(t, D, delta2): return 4 * D * t + 2 * delta2
        def model2(t, D, delta2, alpha): return 4 * D * t**alpha + 2 * delta2

        n_points = results["n_points"]
        MSD = self.parent.track.get_msd()
        reg1 = fit_results["model1"]["params"]
        reg2 = fit_results["model2"]["params"]
        m1 = model1(T, *reg1)
        m2 = model2(T, *reg2)

        self.plot.reset()
        self.plot.setup()
        self.plot_msd.setData(T, MSD)
        self.plot_model1.setData(T[0:n_points], m1[0:n_points])
        self.plot_model2.setData(T[0:n_points], m2[0:n_points])

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

    def on_range_changed(self):
        self.doubleSpinBoxRange.setValue(self.plot.get_range() * 1000.0)

    def set_range_from_spinbox(self):
        self.plot.set_range(self.doubleSpinBoxRange.value() / 1000.0)