from pyqtgraph.Qt import QtGui

# Whereas the other dropdown menus have their options initialized in
# their respective initialization methods, this dropdown's options
# should be initialized upon reading of an edf header.
#
class DemoChannelsComboBox(QtGui.QComboBox):
    def __init__(self):
        QtGui.QComboBox.__init__(self)

        # set the default text that is displayed in the dropdown.
        # this text persists after loading of edf.
        #
        self.setItemText(0, "All Channels")

        # set the width of the channel dropdown menu in pixels
        #
        self.setMinimumWidth(130)

        # initially disable dropdown (until an EDF file is loaded by the user)
        #
        self.setEnabled(False)
    #
    # end of function

    def set_montage_names(self, montage_names_a):

        self.clear()

        # first adds the option of all channels to this combobox.
        #
        self.addItem("All Channels")
        for i in range(len(montage_names_a)):
            self.addItem(montage_names_a[i])
        #
        # end of for
