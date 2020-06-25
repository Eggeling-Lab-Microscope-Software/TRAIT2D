from iscat_lib.analysis import Track, ModelDB
from iscat_lib.analysis.models import ModelBrownian, ModelConfined, ModelHop
from iscat_lib.exceptions import *

import numpy as np

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QWidget, QApplication

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
        if self.parent.track == None or not self.parent.track.get_adc_analysis_results()["analyzed"]:
            self.parent.statusbar.showMessage("Load a track first!")
            mb = QMessageBox()
            mb.setText("Analyze a track first!")
            mb.setWindowTitle("Error")
            mb.setIcon(QMessageBox.Warning)
            mb.exec()
            return

        QApplication.clipboard().setText(str(self.parent.track.get_adc_analysis_results()["results"]))

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
        fit_max_time = self.plot.get_range()
        if fit_max_time <= 0.0:
            fit_max_time = None

        maxfev = int(self.lineEditMaxIt.text())

        try:
            ModelDB().remove_model(ModelBrownian)
            ModelDB().remove_model(ModelConfined)
            ModelDB().remove_model(ModelHop)
        except:
            pass

        model_brownian = ModelBrownian()
        model_confined = ModelConfined()
        model_hop = ModelHop()

        if self.checkBoxUseInitial.checkState():
            if (self.lineEditParam1_1.text() != ""):
                model_brownian.initial[0] = float(self.lineEditParam1_1.text())
            if (self.lineEditParam2_1.text() != ""):
                model_brownian.initial[1] = float(self.lineEditParam2_1.text())


            if (self.lineEditParam1_2.text() != ""):
               model_confined.initial[0] = float(self.lineEditParam1_2.text())
            if (self.lineEditParam2_2.text() != ""):
                model_confined.initial[1] = float(self.lineEditParam2_2.text())
            if (self.lineEditParam3_2.text() != ""):
                model_confined.initial[2] = float(self.lineEditParam3_2.text())

            if (self.lineEditParam1_3.text() != ""):
                model_hop.initial[0] = float(self.lineEditParam1_3.text())
            if (self.lineEditParam2_3.text() != ""):
                model_hop.initial[1] = float(self.lineEditParam2_3.text())
            if (self.lineEditParam3_3.text() != ""):
                model_hop.initial[2] = float(self.lineEditParam3_3.text())
            if (self.lineEditParam4_3.text() != ""):
                model_hop.initial[3] = float(self.lineEditParam4_3.text())

        ModelDB().add_model(model_brownian)
        ModelDB().add_model(model_confined)
        ModelDB().add_model(model_hop)

        R = float(self.doubleSpinBoxInputParam1.value())

        try:
            results = self.parent.track.adc_analysis(R=R, fit_max_time=fit_max_time, maxfev=maxfev)["results"]
        except RuntimeError:
            mb = QMessageBox()
            mb.setText("A model fit failed! Try raising the maximum iterations or different initial values.")
            mb.setWindowTitle("Fit Error")
            mb.setIcon(QMessageBox.Warning)
            mb.exec()
            return

        # Show results for brownian model in GUI
        self.lineEditParam1_1.setText("{:5e}".format(results["models"]["ModelBrownian"]["params"][0]))
        self.lineEditParam1Error_1.setText("{:5e}".format(results["models"]["ModelBrownian"]["errors"][0]))
        self.lineEditParam2_1.setText("{:5e}".format(results["models"]["ModelBrownian"]["params"][1]))
        self.lineEditParam2Error_1.setText("{:5e}".format(results["models"]["ModelBrownian"]["errors"][1]))
        self.lineEditRelLikelihood_1.setText("{:5f}".format(results["models"]["ModelBrownian"]["rel_likelihood"]))
        self.lineEditBIC_1.setText("{:5f}".format(results["models"]["ModelBrownian"]["bic"]))

        # Show results for confined model in GUI
        self.lineEditParam1_2.setText("{:5e}".format(results["models"]["ModelConfined"]["params"][0]))
        self.lineEditParam1Error_2.setText("{:5e}".format(results["models"]["ModelConfined"]["errors"][0]))
        self.lineEditParam2_2.setText("{:5e}".format(results["models"]["ModelConfined"]["params"][1]))
        self.lineEditParam2Error_2.setText("{:5e}".format(results["models"]["ModelConfined"]["errors"][1]))
        self.lineEditParam3_2.setText("{:5e}".format(results["models"]["ModelConfined"]["params"][2]))
        self.lineEditParam3Error_2.setText("{:5e}".format(results["models"]["ModelConfined"]["errors"][2]))
        self.lineEditRelLikelihood_2.setText("{:5f}".format(results["models"]["ModelConfined"]["rel_likelihood"]))
        self.lineEditBIC_2.setText("{:5f}".format(results["models"]["ModelConfined"]["bic"]))

        # Show results for hopping in GUI
        self.lineEditParam1_3.setText("{:5e}".format(results["models"]["ModelHop"]["params"][0]))
        self.lineEditParam1Error_3.setText("{:5e}".format(results["models"]["ModelHop"]["errors"][0]))
        self.lineEditParam2_3.setText("{:5e}".format(results["models"]["ModelHop"]["params"][1]))
        self.lineEditParam2Error_3.setText("{:5e}".format(results["models"]["ModelHop"]["errors"][1]))
        self.lineEditParam3_3.setText("{:5e}".format(results["models"]["ModelHop"]["params"][2]))
        self.lineEditParam3Error_3.setText("{:5e}".format(results["models"]["ModelHop"]["errors"][2]))
        self.lineEditParam4_3.setText("{:5e}".format(results["models"]["ModelHop"]["params"][3]))
        self.lineEditParam4Error_3.setText("{:5e}".format(results["models"]["ModelHop"]["errors"][3]))
        self.lineEditRelLikelihood_3.setText("{:5f}".format(results["models"]["ModelHop"]["rel_likelihood"]))
        self.lineEditBIC_3.setText("{:5f}".format(results["models"]["ModelHop"]["bic"]))

        dt = T[1] - T[0]

        Dapp = self.parent.track.get_adc_analysis_results()["Dapp"]
        reg1 = results["models"]["ModelBrownian"]["params"]
        reg2 = results["models"]["ModelConfined"]["params"]
        reg3 = results["models"]["ModelHop"]["params"]
        m1 = model_brownian(T, *reg1)
        m2 = model_confined(T, *reg2)
        m3 = model_hop(T, *reg3)       

        self.plot.reset()
        self.plot.setup()

        indexes = results["indexes"]

        self.plot_dapp.setData(T, Dapp)
        self.plot_brownian.setData(T[indexes], m1[indexes])
        self.plot_confined.setData(T[indexes], m2[indexes])
        self.plot_hopping.setData(T[indexes], m3[indexes])

        self.plot.set_range(T[indexes[-1]])
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