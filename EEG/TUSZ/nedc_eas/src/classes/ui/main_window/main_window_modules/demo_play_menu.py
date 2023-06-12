from pyqtgraph.Qt import QtGui, QtCore

# class: DemoPlayMenu
#
# this class is a QWidget, that handles the controlling of the play function
#
class DemoPlayMenu(QtGui.QWidget):

    # signal to be emitted when the window is closed without the start button
    #
    signal_no_start=QtCore.Signal()

    # method: __init__
    #
    # arguments: none
    #
    # this method sets up all the widgets to be placed inside the menu
    #
    def __init__(self):

        super(DemoPlayMenu, self).__init__()

        # start with a top level layout
        #
        self.top_level_layout = QtGui.QGridLayout()
        self.setLayout(self.top_level_layout)

        # add a secondary layout for only the slider
        #
        self.slider_layout = QtGui.QGridLayout()
        self.top_level_layout.addLayout(self.slider_layout, 0, 0, 1, 5)

        # create the widgets
        #
        self.init_play_slider()
        self.init_interval_selector()
        self.init_start_and_close_button()

        # setup flag, this becomes true when when start button is pressed
        #
        self.start_pressed = False

    # this allows us to call play_edf_file in DemoEventLoop from this class
    #
    def setup_play_function(self,
                            event_loop_a):
        self.play_function = event_loop_a.play_edf_file

    def init_play_slider(self):
        
        self.play_slider = QtGui.QSlider(self)
        self.play_slider.setOrientation(QtCore.Qt.Horizontal)
        self.play_slider.setTickInterval(10)
        self.play_slider.setMinimum(1)
        self.play_slider.setMaximum(10)
        self.play_slider.setValue(10)
        self.slider_layout.addWidget(self.play_slider, 0, 1, 1, 3)

        self.label_fast = QtGui.QLabel("Fast")
        self.slider_layout.addWidget(self.label_fast, 0, 4, 1, 1)
        self.label_slow = QtGui.QLabel("Slow")
        self.slider_layout.addWidget(self.label_slow, 0, 0, 1, 1)

    def init_interval_selector(self):

        self.label_interval = QtGui.QLabel(self)
        self.top_level_layout.addWidget(self.label_interval, 1, 0, 1, 1)
        self.line_edit_interval = QtGui.QLineEdit(self)
        self.top_level_layout.addWidget(self.line_edit_interval, 1, 1, 1, 1)
        self.label_interval.setText("Interval (sec)")
        self.line_edit_interval.setText("1")
        self.line_edit_interval.setMaximumWidth(70)

    def init_start_and_close_button(self):
        self.start_button = QtGui.QPushButton("Start")
        self.top_level_layout.addWidget(self.start_button, 2, 4, 1, 1)
        self.start_button.clicked.connect(self.start_button_clicked)
        self.start_button.setMinimumWidth(100)

        self.close_button = QtGui.QPushButton("Close")
        self.top_level_layout.addWidget(self.close_button, 2, 0, 1, 1)
        self.close_button.clicked.connect(self.close_button_clicked)
        self.close_button.setMinimumWidth(100)

    # method: start_button_clicked
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is responsible for calling play_edf_file, and passing the
    # speed and interval values along to DemoEventLoop
    #
    def start_button_clicked(self):
        
        self.start_pressed = True

        # get values from widgets
        #
        play_speed = int(self.play_slider.value())
        interval = int(self.line_edit_interval.text())
        
        self.close()
        self.play_function(play_speed,
                           interval)

    def close_button_clicked(self):
        self.close()

    # reimplementation of closeEvent of this widget, allows us to discern
    # closing when we click the start button, or otherwise
    #
    def closeEvent(self,
                   event):
        if self.start_pressed is False:
            self.signal_no_start.emit()
        self.start_pressed = False
        event.accept()
