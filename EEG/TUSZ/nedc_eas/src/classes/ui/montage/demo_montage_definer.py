from pyqtgraph.Qt import QtGui, QtCore

# class: DemoMontageDefiner
#
# this QWidget class is used for defining, and loading new montages.
# More generally, this menu shows when an edf header and montage do
# not match.
#
class DemoMontageDefiner(QtGui.QWidget):

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

        super(DemoMontageDefiner, self).__init__()

        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)

        self.init_matched_channels_list_view()
        self.init_no_match_channels_list_view()
        self.init_select_channel_definition()
        self.init_buttons()

    def init_matched_channels_list_view(self):

        self.label_matched_channels = QtGui.QLabel("Channel Definitions")
        self.label_matched_channels.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label_matched_channels, 0, 1, 1, 1)

        self.list_view_matched_channels = QtGui.QListWidget(self)

        # allow items in this list view to be dragged inside this widget.
        #
        self.list_view_matched_channels. \
            setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.list_view_matched_channels.setMinimumWidth(250)
        self.layout.addWidget(self.list_view_matched_channels, 1, 1, 10, 1)

    def init_no_match_channels_list_view(self):

        self.label_no_match_channels = QtGui.QLabel("Unmatched Definitions")
        self.label_no_match_channels.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label_no_match_channels, 0, 0, 1, 1)

        self.list_view_no_match_channels = QtGui.QListWidget(self)

        # don't allow the items in this list view to be dragged
        #
        self.list_view_no_match_channels. \
            setDragDropMode(QtGui.QAbstractItemView.NoDragDrop)
        self.list_view_no_match_channels.setMinimumWidth(250)
        self.layout.addWidget(self.list_view_no_match_channels, 1, 0, 10, 1)

    def init_select_channel_definition(self):
        self.label_select_definition = QtGui.QLabel("Select Channel Definition")
        self.label_select_definition.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label_select_definition, 0, 2, 1, 2)

        self.label_minuend = QtGui.QLabel("Minuend")
        self.label_minuend.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label_minuend, 2, 2, 1, 1)

        self.dropdown_minuend = QtGui.QComboBox()
        self.dropdown_minuend.setEditable(False)
        self.dropdown_minuend.currentIndexChanged.connect(self.update_channel_name)
        self.layout.addWidget(self.dropdown_minuend, 3, 2, 1, 1)

        self.label_subtrahend = QtGui.QLabel("Subtrahend")
        self.label_subtrahend.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label_subtrahend, 2, 3, 1, 1)

        self.dropdown_subtrahend = QtGui.QComboBox()
        self.dropdown_subtrahend.setEditable(False)
        self.dropdown_subtrahend.currentIndexChanged.connect(self.update_channel_name)
        self.layout.addWidget(self.dropdown_subtrahend, 3, 3, 1, 1)

        self.label_channel_name = QtGui.QLabel("Channel Name")
        self.label_channel_name.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label_channel_name, 5, 2, 1, 1)
        
        self.line_edit_channel_name = QtGui.QLineEdit(self)
        self.layout.addWidget(self.line_edit_channel_name, 6, 2, 1, 1)

    def init_buttons(self):

        self.add_button = QtGui.QPushButton("Add Channel Definition")
        self.add_button.clicked.connect(self.add_button_pressed)
        self.layout.addWidget(self.add_button, 6, 3, 1, 2)

        self.remove_button = QtGui.QPushButton("Remove Definition")
        self.remove_button.clicked.connect(self.remove_button_pressed)
        self.layout.addWidget(self.remove_button, 8, 2, 1, 1)

        self.clear_button = QtGui.QPushButton("Clear All Definitions")
        self.clear_button.clicked.connect(self.clear_button_pressed)
        self.layout.addWidget(self.clear_button, 8, 3, 1, 1)

        self.save_button = QtGui.QPushButton("Save Montage")
        self.save_button.clicked.connect(self.save_button_pressed)
        self.layout.addWidget(self.save_button, 10, 3, 1, 1)

        self.load_button = QtGui.QPushButton("Load Montage")
        self.load_button.clicked.connect(self.load_button_pressed)
        self.layout.addWidget(self.load_button, 10, 2, 1, 1)

    # connect local montage_file_name variable up to DemoEventLoop
    #
    def setup_load_function(self,
                            event_loop_a):
        self.load_function = event_loop_a.load_new_montage

    # method: set_montage
    #
    # arguments:
    #  -montage_minuend_a: dict of minuends read from montage file
    #  -montage_subtrahend_a: dict of subtrahends read from montage file
    #  -edf_channels_a: dict of channels read from edf header
    #  -montage_names_a: dict of montage definition names
    #
    def set_montage(self,
                    montage_minuend_a,
                    montage_subtrahend_a,
                    edf_channels_a,
                    montage_names_a):

        self.minuends = montage_minuend_a
        self.subtrahends = montage_subtrahend_a
        self.edf_channels = edf_channels_a
        self.montage_names = montage_names_a

        self.add_items()
        self.add_dropdown_items()

    # method: add_items
    #
    # arguments: none
    #
    # returns: none
    #
    # this method add our minuends, subtrahends, and names to either
    # matched_list_view or unmatched_list_view
    #
    def add_items(self):

        self.list_items = {}
        self.size = QtCore.QSize(20,20)
        self.font = QtGui.QFont()
        self.font.setPointSize(12)
        print(self.minuends)
        # make sure list_views are totally clear
        #
        self.clear_button_pressed()
        # iterate over minuends, no need to iterate over subtrahends, as we will
        # never have a subtrahend without a minuend
        #
        for i in range(len(self.minuends)):

            # special case where we only have a minuend
            #
            if self.subtrahends[i] is None:

                list_item = QtGui.QListWidgetItem(
                    self.montage_names[i] + ": " + self.minuends[i])
                list_item.setFont(self.font)
                list_item.setSizeHint(self.size)

                # check if minuend is in edf_channels
                #
                if self.minuends[i] in self.edf_channels:
                    self.list_view_matched_channels.addItem(list_item)
                else:
                    self.list_view_no_match_channels.addItem(list_item)
                
            # if we are able to match a minuend and subtrahend to edf
            #
            elif self.minuends[i] in self.edf_channels and \
               self.subtrahends[i] in self.edf_channels:
                
                # create list_item corresponding to montage
                #
                list_item = QtGui.QListWidgetItem(self.montage_names[i] + ": " + 
                                                  self.minuends[i] +
                                                  " -- " +
                                                  self.subtrahends[i])
                list_item.setFont(self.font)
                list_item.setSizeHint(self.size)

                # add to matched list_view
                #
                self.list_view_matched_channels.addItem(list_item)

            # when we cannot match pair to edf channels
            #
            else:

                # create list_item corresponding to montage
                #
                list_item = QtGui.QListWidgetItem(self.montage_names[i] + ": " + 
                                                  self.minuends[i] +
                                                  " -- " +
                                                  self.subtrahends[i])
                list_item.setFont(self.font)
                list_item.setSizeHint(self.size)

                # add to unmatched list_view
                #
                self.list_view_no_match_channels.addItem(list_item)

    # method: add_dropdown_items
    #
    # arguments: none
    #
    # returns: none
    #
    # this method adds all edf_channels to minuend and subtrahend dropdown
    #
    def add_dropdown_items(self):

        # remove all previously added items
        #
        for i in range(self.dropdown_minuend.count()):
            self.dropdown_minuend.removeItem(0)
            self.dropdown_subtrahend.removeItem(0)

        # we want the first item to be a None item
        #
        self.dropdown_minuend.addItem("None")
        self.dropdown_subtrahend.addItem("None")

        # add items to dropdown
        #
        for i in range(len(self.edf_channels)):
            self.dropdown_minuend.addItem(self.edf_channels[i])
            self.dropdown_subtrahend.addItem(self.edf_channels[i])

    # this method gets called whenever one of the dropdowns change text
    #
    def update_channel_name(self):
        self.line_edit_channel_name.setText(self.get_channel_name())

    # method: get_channel_name
    #
    # arguments: none
    #
    # returns: none
    #
    # this method combines the minuend and subtrahend, and creates a mockup
    # of a good channel_name, although user can edit this line_edit
    #
    def get_channel_name(self):

        # get str value of text in dropdowns
        #
        minuend = str(self.dropdown_minuend.currentText())
        subtrahend = str(self.dropdown_subtrahend.currentText())

        # we never want a montage definition without a minuend
        #
        if minuend == "None":
            return ""

        # if no subtrahend, name is just minuend
        #
        elif subtrahend == "None":
            return minuend[0:minuend.index('-')].replace("EEG ", "")

        # combine minuend and subtrahend
        #
        else:
            return minuend[0:minuend.index('-')].replace("EEG ", "") + "-" \
                + subtrahend[0:subtrahend.index('-')].replace("EEG ", "")

    # this method adds minuend, subtrahend, and channel name to the
    # matched channel definition list_view
    #
    def add_button_pressed(self):
        list_item = QtGui.QListWidgetItem(str(self.line_edit_channel_name.text()) +
                                          ": " +
                                          str(self.dropdown_minuend.currentText()) +
                                          " -- " +
                                          str(self.dropdown_subtrahend.currentText()))
        list_item.setFont(self.font)
        list_item.setSizeHint(self.size)
        self.list_view_matched_channels.addItem(list_item)
        self.list_view_matched_channels.setCurrentItem(list_item)


    # this method removes the selected item in matched channel definitions
    #
    def remove_button_pressed(self):
        current_row = self.list_view_matched_channels.currentRow()
        self.list_view_matched_channels.takeItem(current_row)

    # this method clear both list_views of all their items
    #
    def clear_button_pressed(self):
        for i in range(self.list_view_matched_channels.count()):
            self.list_view_matched_channels.takeItem(0)
        for i in range(self.list_view_no_match_channels.count()):
            self.list_view_no_match_channels.takeItem(0)

    # this method get a file name, and writes each line from matched channel
    # definitions to a file
    #
    def save_button_pressed(self):

        self.save_montage_file_name,_ = QtGui.QFileDialog.getSaveFileName(
            self, "Save Montage File",
            "",
            ".txt")

        # if we got a file to save to
        #
        if self.save_montage_file_name:

            save_montage_file = open(self.save_montage_file_name, "w")

            for i in range(self.list_view_matched_channels.count()):
                save_montage_file.write(
                    "montage = " + str(i) + "," +
                    str(self.list_view_matched_channels.item(i).text()) +
                    "\n")

            save_montage_file.close()

    # this method gets a montage file to load, and passes it to event loop
    #
    def load_button_pressed(self):

        self.montage_file_name,_ = QtGui.QFileDialog.getOpenFileName(
            self, "Load Montage File", "src/defaults/", 
            "Montage Files (*.txt)")

        # if we got a file to load
        #
        if self.montage_file_name:
            self.load_function(self.montage_file_name)
            self.close()

    def show_montage_definer(self):
        self.show()
        self.activateWindow()
        self.raise_()

    # reimplementation, allows us to emit a signal when the widget closes
    #
    def closeEvent(self,
                   event):
        self.signal_definer_closed.emit(True)
        event.accept()
