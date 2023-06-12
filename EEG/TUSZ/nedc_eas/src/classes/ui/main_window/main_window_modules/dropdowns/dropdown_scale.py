from pyqtgraph.Qt import QtGui

class DemoScaleComboBox(QtGui.QComboBox):
    def __init__(self):
        QtGui.QComboBox.__init__(self)

        # declare the list of strings that will be shown in time_scale
        # dropdown menu. these are invisible until after first edf load.
        # all parameters are in seconds.
        #
        self.values = ["100",
                       "60",
                       "30",
                       "20",
                       "15",
                       "10",
                       "5",
                       "2",
                       "1",
                       "0.5",
                       "0.2",
                       "0.1"]

        # add the list of strings to the time_scale dropdown menu
        #
        for i in range(len(self.values)):
            self.addItem(self.values[i])
        #
        # end of for

        # initially disable dropdown (until an EDF file is loaded by the user)
        #
        self.setEnabled(False)

        # set the width of the channel dropdown menu in pixels
        #
        self.setMinimumWidth(100)

        # TODO: is this necessary? Doesn't seem to do anything.
        #
        self.setEditable(True)

        # set time_scale_combobox equals to 10 seconds.
        #
        self.setCurrentIndex(5)
