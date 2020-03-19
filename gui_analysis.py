import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog

from iscat_lib.analysis import Track


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi('gui_analysis.ui', self)

        self.pushButtonLoadTrack.clicked.connect(self.load_trajectory)

    def load_trajectory(self, filename: str):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load track file", "", "Track files (*.csv)")
        if filename == '':
            return
        self.track = Track.from_file(filename)
        self.statusbar.showMessage(
            "Loaded track of length {}".format(self.track.get_size()))


def main():
    app = QtWidgets.QApplication(sys.argv)

    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
