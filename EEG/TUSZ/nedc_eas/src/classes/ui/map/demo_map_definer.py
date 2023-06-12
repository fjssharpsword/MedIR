from pyqtgraph.Qt import QtGui, QtCore

# class: DemoMapDefiner
#
# this QWidget class is used for defining, and loading new maps.
# this class mirros demo_montage_definer
#
class DemoMapDefiner(QtGui.QWidget):

    # signal to notify DemoEventLoop when this widget closes
    #
    signal_definer_closed=QtCore.Signal(bool)

    # method: __init__
    #
    # arguments: none
    #
    # returns: none
    #
    # this method constructs our widget, each below init_*() method constructs
    # this widget's child widgets
    #
    def __init__(self):

        super(DemoMapDefiner, self).__init__()

        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)

    # connect local map_file_name variable up to DemoEventLoop
    #
    def setup_load_function(self,
                            event_loop_a):
        self.load_function = event_loop_a.load_new_map
    
    # this method gets a map file to load, and passes it to event loop
    #
    def load_button_pressed(self):

        self.map_file_name,_ = QtGui.QFileDialog.getOpenFileName(
            self, "Load Map File", "src/defaults/map/", "*.txt")

        # if we got a file to load
        #
        if self.map_file_name:
            self.load_function(self.map_file_name)
            self.close()
