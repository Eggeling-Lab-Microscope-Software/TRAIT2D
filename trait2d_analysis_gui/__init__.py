import sys

import pandas as pd

from PyQt5 import QtCore, uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QFileDialog, QMessageBox

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

from trait2d.analysis import Track, ModelDB
from trait2d.analysis.models import ModelBrownian, ModelConfined, ModelHop
from trait2d.exceptions import *

import trait2d_analysis_gui.tab.msd
import trait2d_analysis_gui.tab.adc

import os

class MainWindow(QMainWindow):

    sigTrackLoaded = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load .ui file
        dirname = os.path.dirname(__file__)
        uic.loadUi(os.path.join(dirname, 'gui_analysis.ui'), self)

        # Add widgets to tabs
        wid_msd = trait2d_analysis_gui.tab.msd.widgetMSD(self)
        wid_adc = trait2d_analysis_gui.tab.adc.widgetADC(self)

        self.layoutMSD.addWidget(wid_msd, 0, 0)
        self.layoutADC.addWidget(wid_adc, 0, 0)

        self.statusBar().setSizeGripEnabled(False)

        # Initialize with no track
        self.track = None

        self.pushButtonLoadTrack.clicked.connect(self.load_track_dialog)

        ModelDB().add_model(ModelBrownian)
        ModelDB().add_model(ModelConfined)
        ModelDB().add_model(ModelHop)

    def show_trackid_dialog(self):
        Dialog = QDialog()
        dirname = os.path.dirname(__file__)
        uic.loadUi(os.path.join(dirname, "dialog_trackid.ui"), Dialog)
        Dialog.show()
        resp = Dialog.exec_()

        if resp == QDialog.Accepted:
            return Dialog.lineEditId.text()

    def show_import_dialog(self, column_names):
        Dialog = QDialog()
        dirname = os.path.dirname(__file__)
        uic.loadUi(os.path.join(dirname, "dialog_import.ui"), Dialog)
        Dialog.show()

        Dialog.comboBoxColId.addItem('')
        for c in column_names:
            Dialog.comboBoxColX.addItem(c)
            Dialog.comboBoxColY.addItem(c)
            Dialog.comboBoxColT.addItem(c)
            Dialog.comboBoxColId.addItem(c)

        Dialog.comboBoxColX.setCurrentText('')
        Dialog.comboBoxColY.setCurrentText('')
        Dialog.comboBoxColT.setCurrentText('')
        Dialog.comboBoxColId.setCurrentText('')

        if 'x' in column_names:
            Dialog.comboBoxColX.setCurrentText('x')
        if 'y' in column_names:
            Dialog.comboBoxColY.setCurrentText('y')
        if 't' in column_names:
            Dialog.comboBoxColT.setCurrentText('t')
        if 'id' in column_names:
            Dialog.comboBoxColId.setCurrentText('id')

        resp = Dialog.exec()

        if resp == QDialog.Accepted:
            return  Dialog.comboBoxLength.currentText(), \
                    Dialog.comboBoxTime.currentText(), \
                   [Dialog.comboBoxColX.currentText(),
                    Dialog.comboBoxColY.currentText(),
                    Dialog.comboBoxColT.currentText(),
                    Dialog.comboBoxColId.currentText()]

        return None, None, None

    def load_track(self, filename : str, id=None, unit_length = None, unit_time = None, col_names = None):
        if unit_length is None or unit_time is None or col_names is None:
            unit_length, unit_time, col_names = self.show_import_dialog(pd.read_csv(filename).columns)
            if unit_length is None or unit_time is None or col_names is None:
                return
        try:
            self.track = Track.from_file(filename, id=id, unit_length=unit_length, unit_time=unit_time, col_name_x = col_names[0], col_name_y = col_names[1], col_name_t = col_names[2], col_name_id = col_names[3]).normalized(normalize_t = True, normalize_xy = False)
        except LoadTrackMissingIdError:
            id = self.show_trackid_dialog()
            if id is None:
                return
            self.load_track(filename=filename, id=id, unit_length=unit_length, unit_time=unit_time, col_names=col_names)
        except LoadTrackIdNotFoundError:
            mb = QMessageBox()
            mb.setText("There is no track with that ID!")
            mb.setWindowTitle("Error")
            mb.setIcon(QMessageBox.Critical)
            mb.exec()
        except KeyError as e:
            mb = QMessageBox()
            mb.setText("Key not found: {}!".format(str(e)))
            mb.setWindowTitle("Error")
            mb.setIcon(QMessageBox.Critical)
            mb.exec()
        if self.track==None:
            return

        self.sigTrackLoaded.emit()

        self.statusbar.showMessage(
            "Loaded track of length {}".format(self.track.get_size()))      

    def load_track_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load track file", "", "Track files (*.csv)")
        if filename == '':
            return
        self.load_track(filename, id=None)


def main():
    app = QApplication(sys.argv)

    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
