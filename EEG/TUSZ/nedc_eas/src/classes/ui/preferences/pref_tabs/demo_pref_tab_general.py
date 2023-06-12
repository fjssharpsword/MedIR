from pyqtgraph.Qt import QtGui, QtCore

MESSAGE_STRING = '''Order of Views is only implemented through saving.\n\n
After saving, changes will be made after restarting.'''

#------------------------------------------------------------------------------
#
# file: DemoPrefTabGeneral
#
# this file holds the general preferences QLineEdits and QLabels to be placed
# into DemoPrefencesWidget's general tab.
#
class DemoPrefTabGeneral(QtGui.QWidget):

    # method: __init__
    #
    # arguments:
    #  - dict_main_window_a: dictionary of of options from section [MainWindow]
    #
    # returns: none
    #
    # this method initializes DemoPrefencesTabGeneral and calls their
    # other init functions
    #
    def __init__(self,
                 dict_main_window_a,
                 dict_order_a,
                 dict_waveform_a):

        super(DemoPrefTabGeneral, self).__init__()
        self.layout_grid = QtGui.QGridLayout(self)

        self.dict_main_window = dict_main_window_a
        self.dict_order = dict_order_a
        self.dict_waveform = dict_waveform_a

        self.init_initial_window_height_line_edit()

        self.init_initial_window_width_line_edit()

        self.init_default_timescale_line_edit()

        self.init_default_sensitivity_line_edit()

        self.init_signal_color_line_edit()

        dividing_line = QtGui.QFrame(self)
        dividing_line.setFrameShape(QtGui.QFrame.VLine)
        dividing_line.setFrameShadow(QtGui.QFrame.Sunken)
        self.layout_grid.addWidget(dividing_line, 0, 3, 5, 1)

        self.init_list_view()

        self.is_changed = False
    #
    # end of method

    # method: init_default_timescale_line_edit
    #
    # arguments: none
    #
    # returns: none
    #
    # this method creates the QLabel and QLineEdit for default timescale preference
    #
    def init_default_timescale_line_edit(self):
        self.label_default_timescale = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_default_timescale, 0, 0, 1, 1)
        self.line_edit_default_timescale = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_default_timescale, 0, 1, 1, 1)
        self.label_default_timescale.setText( "Default Timescale (sec)")
        self.line_edit_default_timescale.setText(self.dict_main_window
                                                 ['initial_time_scale'])
        self.line_edit_default_timescale.textChanged.connect(self.set_changed)
        self.line_edit_default_timescale.setMaximumWidth(150)
        self.line_edit_default_timescale.setAlignment(QtCore.Qt.AlignLeft)
    #
    # end of method

    def init_default_sensitivity_line_edit(self):
        self.label_default_sensitivity = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_default_sensitivity, 1, 0, 1, 1)
        self.line_edit_default_sensitivity = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_default_sensitivity, 1, 1, 1, 1)
        self.label_default_sensitivity.setText("Default Sensitivity")
        self.line_edit_default_sensitivity.setText(self.dict_main_window
                                                   ['initial_sensitivity'])
        self.line_edit_default_sensitivity.textChanged.connect(self.set_changed)
        self.line_edit_default_sensitivity.setMaximumWidth(150)
        self.line_edit_default_sensitivity.setAlignment(QtCore.Qt.AlignLeft)

    def init_initial_window_height_line_edit(self):
        self.label_initial_window_height = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_initial_window_height, 2, 0, 1, 1)
        self.line_edit_initial_window_height = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_initial_window_height, 2, 1, 1, 1)
        self.label_initial_window_height.setText(
            "Initial Window Height (Pixels)")
        self.line_edit_initial_window_height.setText(self.dict_main_window
                                                     ['y_pixels_initial_number'])
        self.line_edit_initial_window_height.textChanged.connect(self.set_changed)
        self.line_edit_initial_window_height.setMaximumWidth(150)
        self.line_edit_initial_window_height.setAlignment(QtCore.Qt.AlignLeft)
        
    def init_initial_window_width_line_edit(self):
        self.label_initial_window_width = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_initial_window_width, 3, 0, 1, 1)
        self.line_edit_initial_window_width = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_initial_window_width, 3, 1, 1, 1)
        self.label_initial_window_width.setText(
            "Initial Window Width (Pixels)")
        self.line_edit_initial_window_width.setText(self.dict_main_window
                                                    ['x_pixels_initial_number'])
        self.line_edit_initial_window_width.textChanged.connect(self.set_changed)
        self.line_edit_initial_window_width.setMaximumWidth(150)
        self.line_edit_initial_window_width.setAlignment(QtCore.Qt.AlignLeft)

    def init_signal_color_line_edit(self):
        self.label_signal_color = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_signal_color, 4, 0, 1, 1)
        self.line_edit_signal_color = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_signal_color, 4, 1, 1, 1)
        self.label_signal_color.setText("Waveform Signal Color")
        self.line_edit_signal_color.setText(str(self.dict_waveform['signal_color_pen']))
        self.line_edit_signal_color.textChanged.connect(self.set_changed)
        self.line_edit_signal_color.setMaximumWidth(150)
        self.line_edit_signal_color.setAlignment(QtCore.Qt.AlignLeft)

    def init_list_view(self):

        self.label_order_of_views = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_order_of_views, 0, 4, 1, 1)
        self.label_order_of_views.setText("Order of Views: ")
        self.list_view = QtGui.QListWidget()
        self.list_view.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        
        self.list_view.currentRowChanged.connect(self.set_changed)
        self.layout_grid.addWidget(self.list_view, 1, 4, 4, 1)

        self.energy_view = QtGui.QListWidgetItem("Energy")
        self.spectrogram_view = QtGui.QListWidgetItem("Spectrogram")
        self.waveform_view = QtGui.QListWidgetItem("Waveform")

        size = QtCore.QSize(35,35)

        self.energy_view.setSizeHint(size)
        self.spectrogram_view.setSizeHint(size)
        self.waveform_view.setSizeHint(size)

        font = QtGui.QFont()
        font.setPointSize(15)

        self.energy_view.setFont(font)
        self.spectrogram_view.setFont(font)
        self.waveform_view.setFont(font)

        # this dict allows for us to decode from the string inside the view_item
        # to the view_item itself
        #
        self.decode_views = {'Energy': self.energy_view,
                             'Spectrogram': self.spectrogram_view,
                             'Waveform': self.waveform_view}

        self.add_items()

    # method: add_items
    #
    # arguments: none
    #
    # returns: none
    #
    # this method add the view_items to the ListWidget in the order specified
    # by the config file
    #
    def add_items(self):
        view_items_string = self.dict_order['view_items']
        view_items = view_items_string.replace(" ", "").split(',')

        for view in view_items:
            self.list_view.addItem(self.decode_views[view])

    def show_dialog(self):
        message = QtGui.QMessageBox(self)
        message.setWindowTitle("Help")
        message.setText(MESSAGE_STRING)
        message.show()

    # method: get_settings
    #
    # arguments: none
    #
    # returns:
    #  -time_scale: QString text from the timescale QLineEdit
    #  -sensitivity: QString text from the sensitivity QLineEdit
    #  -window_width: text from window width QLineEdit as an integer
    #  -window_height: text from window height QLineEdit as an integer
    #
    # this method returns the values to be connected to DemoEventLoop and further
    #
    def get_settings(self):
        
        time_scale = self.line_edit_default_timescale.text()

        sensitivity = self.line_edit_default_sensitivity.text()

        window_width = int(self.line_edit_initial_window_width.text())

        window_height = int(self.line_edit_initial_window_height.text())

        signal_color_string = str(self.line_edit_signal_color.text())
        signal_color_tuple = tuple(map(int, signal_color_string[1:-1]
                                       .replace(" ", "").split(",")))

        view_items = []
        for view_index in range(self.list_view.count()):
            view_items.append(str(self.list_view.item(view_index).text()))

        view_items_string = str(view_items)[1:-1].replace("'", "")

        return(time_scale,
               sensitivity,
               window_width,
               window_height,
               signal_color_tuple,
               view_items_string)

    # method: set_changed
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called whenever any of the widgets have changed text/values
    #
    def set_changed(self):
        self.is_changed = True

    # method: set_unchanged
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called when DemoEventLoop is finished updating preference values
    def set_unchanged(self):
        self.is_changed = False
