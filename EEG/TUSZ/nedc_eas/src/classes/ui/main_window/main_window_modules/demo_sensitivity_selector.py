from pyqtgraph.Qt import QtGui, QtCore

# class: DemoSensitivitySelector
#
# this class is responsible for the widget that appears when clicking the
# channel sensitivity button. It creates a dropdown for each channel, allowing
# the user to select specific sensitivity per channel
#
class DemoSensitivitySelector(QtGui.QWidget):

    # signal to notify event loop when sensitivity has changed
    #
    signal_sens_changed = QtCore.Signal()

    # method: __init__
    #
    # arguments:
    #  -dict_main_window_a: dictionary to decide what the default sensitivity is
    #
    # returns: none
    #
    # constructs the widget and all it's labels and dropdowns
    #
    def __init__(self,
                 dict_main_window_a,
                 montage_names_a):
        super(DemoSensitivitySelector, self).__init__()

        self.dict_main_window = dict_main_window_a
        self.montage_names = montage_names_a

        # these are the values that will be added to the dropdown
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
        
        self.layout_grid = QtGui.QGridLayout()
        self.setLayout(self.layout_grid)

        self.set_montage_labels()
        self.set_sensitivity_dropdowns()
        self.connect_dropdowns_to_signal()

    # method: set_montage_labels
    #
    # arguments: none
    #
    # returns: none
    #
    # this method creates all channel name labels for the widget.
    #
    def set_montage_labels(self):

        # dict to store labels
        #
        self.dict_channel_names = {}

        # keep track what row we are on
        #
        row = 0

        # iterate over indices of montage_names
        #
        for name in range(len(self.montage_names)):

            # create label, with string from montage_names
            #
            label = QtGui.QLabel(self.montage_names[name])

            # add label to layout
            #
            self.layout_grid.addWidget(label, row, 0)

            # store label, this is necessary, as python will garbage collect
            # these labels if not stored
            #
            self.dict_channel_names[name] = label
            
            row += 1

    # method: set_sensitivity_dropdowns
    #
    # arguments: none
    #
    # returns: none
    #
    # this method creates the dropdowns for the widget
    #
    def set_sensitivity_dropdowns(self):

        # dict to store dropdowns
        self.dict_sens_dropdowns = {}
        row = 0

        for name in range(len(self.montage_names)):

            # create, allow to type, and make the dropdown smaller
            #
            dropdown = QtGui.QComboBox()
            dropdown.setEditable(True)
            dropdown.setFixedHeight(25)
            dropdown.setFixedWidth(90)

            # add all values from list to dropdown
            #
            for value in self.values:
                dropdown.addItem(value)

            # set default to 10, this will normally get changed from the main
            # sensitivity dropdown
            #
            dropdown.setCurrentIndex(6)

            # add to layout
            #
            self.layout_grid.addWidget(dropdown, row, 1)

            # store dropdown
            self.dict_sens_dropdowns[name] = dropdown
            row += 1

    # method: get_sensitivity_list
    #
    # arguments: none
    #
    # returns: list, containing each channel specific sensitivity value
    #
    # this method is called whenever any sensitivity value is changed
    #
    def get_sensitivity_list(self):

        sensitivity_list = []

        # iterate over all dropdowns
        #
        for dropdown in self.dict_sens_dropdowns:

            # get text from dropdown
            #
            sens_value = int(self.dict_sens_dropdowns[dropdown].currentText())
            sensitivity_list.append(sens_value)
            
        return sensitivity_list

    # method: set_all_dropdowns
    #
    # arguments:
    #  -index: index of all channels dropdown
    #  -value: value at index
    #
    # returns: none
    #
    # this method is called when the user changes the all channels sensitivity.
    # it sets all per channel dropdowns to that value
    #
    def set_all_dropdowns(self,
                          index,
                          value):

        # disconnect the dropdowns from the signal, we will manually emit the signal
        #
        self.disconnect_dropdowns_from_signal()

        # check if the index was created through typing
        #
        if index not in range(len(self.values)):

            # add this new value to our list here
            #
            self.values.append(value)

            # update each dropdown
            #
            for dropdown in self.dict_sens_dropdowns:
                self.dict_sens_dropdowns[dropdown].addItem(value)

        # change all dropdowns to this value
        #
        for dropdown in self.dict_sens_dropdowns:
            self.dict_sens_dropdowns[dropdown].setCurrentIndex(index)

        # reconnect signal, after all dropdowns have changed
        #
        self.connect_dropdowns_to_signal()

        # manually emit signal
        #
        self.signal_sens_changed.emit()

    # method: connect_dropdowns_to_signal
    #
    # arguments: none
    #
    # returns: none
    #
    # this method allows us to notify EventLoop whenever any dropdown has changed
    #
    def connect_dropdowns_to_signal(self):

        for dropdown in self.dict_sens_dropdowns:
            self.dict_sens_dropdowns[dropdown]. \
                currentIndexChanged.connect(self.signal_sens_changed)

    def disconnect_dropdowns_from_signal(self):

        for dropdown in self.dict_sens_dropdowns:
            self.dict_sens_dropdowns[dropdown]. \
                currentIndexChanged.disconnect()
