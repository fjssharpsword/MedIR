from pyqtgraph import QtGui, QtCore
import os

# This class is run for showing the location of electrodes for 10-20 system EEG
# when the user clicks on Help Button.
#
class DemoMontageImageViewer(QtGui.QWidget):

    def __init__(self):
        super(DemoMontageImageViewer, self).__init__()

        # get the current path of main.py
        #
        cwd = os.getcwd()

        layout = QtGui.QHBoxLayout(self)
        self.setLayout(layout)

        # this is a png image, but the git is set to ignore png
        # images, so just calling it img for now
        #
        path_to_image = cwd + "/resources/montage_image.img"
        pixmap = QtGui.QPixmap(path_to_image)
        lbl = QtGui.QLabel(self)
        lbl.setPixmap(pixmap)
        layout.addWidget(lbl)

        self.setWindowTitle('Scalp electrodes')
        

