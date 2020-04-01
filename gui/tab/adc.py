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

        self.pushButtonFormula_1.clicked.connect(self.show_formula_model_1)
        self.pushButtonFormula_2.clicked.connect(self.show_formula_model_2)
        self.pushButtonFormula_3.clicked.connect(self.show_formula_model_3)

        self.parent.sigTrackLoaded.connect(self.plot.reset)

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