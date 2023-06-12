from pyqtgraph.Qt import QtGui

class DemoSensitivityComboBox(QtGui.QComboBox):
    def __init__(self):
        QtGui.QComboBox.__init__(self)

        # declare the list of strings that will be shown in sensitivity
        # dropdown menu. these are invisible until after first edf load.
        # all values are in uV/mm
        #
        self.values = ["1000",
                       "500",
                       "200",
                       "100",
                       "50",
                       "20",
                       "10",
                       "5",
                       "2",
                       "1",
                       "0.5",
                       "0.2",
                       "0.1",
                       "0.05",
                       "0.02",
                       "0.01"]

        # add the list of strings to the sensitivity dropdown menu
        #
        for i in range(len(self.values)):
            self.addItem(self.values[i])
        #
        # end of for

        # initially disable dropdown (until an EDF file is loaded by the user)
        #
        self.setEnabled(False)

        # set the width of the sensitivity dropdown menu in pixels
        #
        self.setMinimumWidth(100)

        # TODO: is this necessary? Doesn't seem to do anything.
        #
        self.setEditable(True)

        # set time_scale_combobox equals to 10 uV/mm.
        #
        self.setCurrentIndex(6)
