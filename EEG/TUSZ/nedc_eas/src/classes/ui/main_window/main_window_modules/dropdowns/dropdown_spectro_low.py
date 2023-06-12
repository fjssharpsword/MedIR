from pyqtgraph.Qt import QtGui

class DemoSpectroLowComboBox(QtGui.QComboBox):
    def __init__(self):
        QtGui.QComboBox.__init__(self)

        # declare the list of strings that will be shown in sensitivity
        # dropdown menu. these are invisible until after first edf load.
        # all values are in uV/mm
        #
        self_values = ["0",
                       "4",
                       "8",
                       "12",
                       "16",
                       "20",
                       "30"]

        # add the list of strings to the sensitivity dropdown menu
        #
        for i in range(len(self_values)):
            self.addItem(
                self_values[i])
        #
        # end of for

        # initially disable dropdown (until an EDF file is loaded by the user)
        #
        self.setEnabled(False)

        # set the width of the spectrogram_low dropdown menu in pixels
        #
        self.setMinimumWidth(100)

        # TODO: is this necessary? Doesn't seem to do anything.
        #
        self.setEditable(True)

        # default value is 0 Hz
        #
        self.setCurrentIndex(0)
