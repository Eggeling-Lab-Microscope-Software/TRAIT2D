import sys

from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QFileDialog, QMessageBox

from iscat_lib.analysis import Track
from iscat_lib.exceptions import *

import gui.tab.msd

class MainWindow(QMainWindow):

    sigTrackLoaded = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load .ui file
        uic.loadUi('gui_analysis.ui', self)

        # Add widgets to tabs
        wid_msd = gui.tab.msd.widgetMSD(self)
        self.layoutMSD.addWidget(wid_msd, 0, 0)

        # Initialize with no track
        self.track = None

        self.pushButtonLoadTrack.clicked.connect(self.load_track_dialog)

    def show_trackid_dialog(self):
        Dialog = QDialog()
        uic.loadUi("gui/dialog_trackid.ui", Dialog)
        Dialog.show()
        resp = Dialog.exec_()

        if resp == QDialog.Accepted:
            return Dialog.spinBoxId.value()

    def load_track(self, filename : str, id=None):
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
