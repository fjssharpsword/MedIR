
import imp
from pyqtgraph.Qt import QtCore, QtGui

# submodules
#
from .main_window_modules.demo_menu_bar import DemoMenuBar
from .main_window_modules.demo_tool_bar import DemoToolBar
from .main_window_modules.demo_map_info_bar import DemoMapInfoBar
from .main_window_modules.demo_info_bar import DemoInfoBar
from .main_window_modules.demo_slider import DemoSlider
from .main_window_modules.demo_unused_plot_widget import DemoUnusedPlotWidget
from .main_window_modules.demo_anno_type_selector import DemoAnnoTypeSelector
from .main_window_modules.demo_sensitivity_selector import DemoSensitivitySelector
from .main_window_modules.demo_play_menu import DemoPlayMenu

# TODO: move this functin to a better file
#
from classes.anno.demo_annotator import _make_map_dictionary

# window_manager is a global object (essentially a glorified list)
# that keeps references to various kinds of windows
# (DemoSearchUserInterface, DemoMainWindow, DemoTextReportWidget)
# until these windows are closed. See the documentation in this class
# for the reasoning, if you care.
#
from classes.ui.demo_window_manager import window_manager

# ------------------------------------------------------------------------------
#
# classes are listed here
#
# ------------------------------------------------------------------------------


class DemoMainWindow(QtGui.QMainWindow):

    # it seems that signals must be declared before the constructor (?)
    #
    sig_resized = QtCore.Signal(object)
    sig_closed = QtCore.Signal(object)
    sig_spec_frequency_changed = QtCore.Signal(float,
                                               float)

    def __init__(self,
                 config_dict_a,
                 dict_anno_a,
                 montage_names_a):
        super(DemoMainWindow, self).__init__()

        QtGui.QMainWindow.__init__(self)

        self.config_dict = config_dict_a

        # give the main window a title
        #
        self.setWindowTitle("NEDC EAS")

        #  resize main window frsom config file parameters
        #
        self.resize(int(self.config_dict['x_pixels_initial_number']),
                    int(self.config_dict['y_pixels_initial_number']))

        # go ahead and select a font to use using config file parameters
        #
        self.font = QtGui.QFont()
        self.font.setPointSize(
            int(self.config_dict['font_size_dropdown_lbls']))

        # create, name, and set central widget in main window.
        # all subsequent widgets will be set in central widget
        # or the layout_grid which governs it
        #
        self.central_widget = QtGui.QWidget(self)
        self.setCentralWidget(self.central_widget)

        # create and name the layout in which all of the widgets are set
        # the layout is a part of self.central_widget
        #
        self.layout_grid = QtGui.QGridLayout(self.central_widget)

        # create tool_bar horizontal section, add to it the following tools:
        #  1) channel dropdown dropdown menu

        #  2) sensitivity dropdown dropdown menu
        #  3) time_scale dropdown dropdown menu
        #  4) 4 annotation navigation buttons (first, last, next, previous)
        #
        self.tool_bar = DemoToolBar(self.font)
        self.layout_grid.addLayout(self.tool_bar, 0, 0, 1, 3)

        # its nice to refer to these via this class rather than the tool_bar class
        #
        self.dropdown_sensitivity = self.tool_bar.dropdown_sensitivity
        self.dropdown_time_scale = self.tool_bar.dropdown_time_scale

        # create a map that shows the current map file being used
        #
        self.map_info_bar = DemoMapInfoBar(self.font)
        self.layout_grid.addLayout(self.map_info_bar, 1, 0, 1, 3)

        # create info_bar horizontal section, add to it the following displays:
        #  1) patient name
        #  2) session date
        #  3) session start time
        #
        # also, add the spectrogram frequency dropdown menus
        #
        self.info_bar = DemoInfoBar(self.font)
        self.layout_grid.addLayout(self.info_bar, 2, 0, 1, 3)

        # its nice to refer to these via this class rather than the info_bar class
        #
        self.label_spectro_freq = self.info_bar.label_spectro_freq_dropdown
        self.dropdown_spectro_low = self.info_bar.dropdown_spectro_low
        self.dropdown_spectro_high = self.info_bar.dropdown_spectro_high

        # if the spectrogram frequency changes, it needs to inform
        # other parts of the program. It does this by emitting this signal
        #
        self.dropdown_spectro_low.currentIndexChanged.connect(
            self.emit_spectrogram_frequency_range)
        self.dropdown_spectro_high.currentIndexChanged.connect(
            self.emit_spectrogram_frequency_range)

        # initialize unused_plot_widget which is used for adjusting position of
        # slider. We do not plot anything in this widget, but without it, the
        # expanding policy for the adjustment of the slider cannot be used.
        #
        self.unused_plot_widget = DemoUnusedPlotWidget()
        self.layout_grid.addWidget(self.unused_plot_widget, 4, 2, 1, 1)

        # create, name, set orientation of slider, and then add it to
        # layout_grid
        #
        self.horizontal_slider = DemoSlider(config_dict_a)
        self.layout_grid.addWidget(self.horizontal_slider, 4, 2, 1, 1)

        self.play_button = QtGui.QPushButton()
        self.play_button.setText("Play")
        self.play_button.clicked.connect(self.open_play_menu)
        self.layout_grid.addWidget(self.play_button, 4, 0, 1, 1)

        # self.frame_layout holds all plot widgets:
        #
        #           self.frame
        #               |
        #               V
        #           self.frame_layout
        #               |
        #               V
        # demo_scroll_area_channels_sigplots as channels_widget
        #   (this is a QtGui.QStackedWidget object)
        #       |                             |
        #       V                             |
        #  page_only_waveform                 V
        #       |                          page_mixed_view
        #       |                             |
        #       V                             V
        #   special case                   standard case
        #       |                          |     |     |
        #       V                          V     |     |
        #   special waveform plot      waveform  V     |
        #                                     energy   V
        #                                          spectrogram
        #
        # all of the widgets below self.frame_layout are added via the
        # add_channels_widget method
        #
        self.frame = QtGui.QFrame()
        self.frame.setFrameShape(QtGui.QFrame.Box)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_layout = QtGui.QGridLayout(self.frame)
        self.layout_grid.addWidget(self.frame, 3, 0, 1, 3)

        # self.menu_bar holds all of the menus such as File, Filter, etc.
        #
        self.menu_bar = DemoMenuBar(self)
        self.setMenuBar(self.menu_bar)

        self.init_keyboard_shortcuts()

        self.anno_type_selector = DemoAnnoTypeSelector(_make_map_dictionary(dict_anno_a))

        # connect annotation button to selector widget
        #
        self.tool_bar.annotations_button.clicked.connect(
            self.show_anno_type_selector)

        self.sensitivity_selector = DemoSensitivitySelector(config_dict_a,
                                                            montage_names_a)

        self.tool_bar.channels_button.clicked.connect(
            self.show_sensitivity_selector)

        self.play_menu = DemoPlayMenu()
        self.play_menu.signal_no_start.connect(self.update_play_button)

        # bool to see if ctrl is being held
        #
        self.ctrl_held = False

        # inform the window manager about this new window.
        # see comments in that class for reasoning for this, if you care.
        #
        window_manager.manage(self)
    #
    # end of constructor function

    # enable menu/Filter, menu/View, menu/Format as an EDF
    # file is loaded by the user.
    #

    def enable_tools_on_edf_read(self):

        self.menu_bar.menu_view.setEnabled(True)
        self.menu_bar.menu_filter.setEnabled(True)
        self.menu_bar.menu_fft.setEnabled(True)

        self.menu_bar.menu_file.action_file_save.setEnabled(True)
        self.menu_bar.menu_file.action_file_save_as.setEnabled(True)

        # enable interaction with dropdown tools
        #
        self.dropdown_sensitivity.setEnabled(True)
        self.dropdown_time_scale.setEnabled(True)
        self.dropdown_spectro_low.setEnabled(True)
        self.dropdown_spectro_high.setEnabled(True)

        # after creating all child widgets, make sure we are focused on parent
        #
        self.setFocus(True)

    # method: resizeEvent
    #
    # args:
    #  -event: emitted when the main window is resized (usually by th user)
    #
    # returns: none
    #
    # reimplementation of resizeEvent so that the spectrogram plot can
    # know to update its geometryx
    #
    def resizeEvent(self, resizeEvent):
        QtGui.QMainWindow.resizeEvent(self, resizeEvent)
        self.sig_resized.emit(self)

    # method: closeEvent
    #
    # args:
    #  -event: Close events are sent to widgets that the user wants to close,
    #          usually by choosing "Close" from the window menu, or by
    #          clicking the X title bar button.  will go here
    #
    # returns: none
    #
    # reimplementation of closeEvent so that the window_manager can
    # now that this is closed, and remove its reference to this
    # widget.
    #
    def closeEvent(self,
                   event):
        self.sig_closed.emit(self)
        QtGui.QMainWindow.closeEvent(self, event)

    def emit_spectrogram_frequency_range(self):
        spec_low_freq = float(self.dropdown_spectro_low.currentText())
        spec_high_freq = float(self.dropdown_spectro_high.currentText())
        self.sig_spec_frequency_changed.emit(spec_low_freq,
                                             spec_high_freq)

    ################################################################
    #### TODO: everything from here down probably be moved #########
    ################################################################

    # method: about_eas
    #
    # arguments: None
    #
    # return: None
    #
    # This method opens a message box to show information about EAS.
    #
    def about_eas(self):
        QtGui.QMessageBox.information(
            self,
            "About NEDC EAS",
            "NEDC EAS 5.1.1",
            QtGui.QMessageBox.Ok)
    #
    # end of function

    # method: quit_eas
    #
    # arguments: None
    #
    # return: None
    #
    # This method exits EAS
    #
    def quit_eas(self):
        QtGui.QApplication.quit()

    #
    # end of function

    def input_dialogue_for_filter_frequency_selection(self,
                                                      title_a,
                                                      prompt_a,
                                                      default_value_a,
                                                      upper_limit_a):
        lower_limit = 0
        num_decimals_accepted = 2
        cutoff_frequency, ok = QtGui.QInputDialog. \
            getDouble(self,
                      title_a,
                      prompt_a,
                      default_value_a,
                      lower_limit,
                      upper_limit_a,
                      num_decimals_accepted)
        return cutoff_frequency, ok

    def init_keyboard_shortcuts(self):
        self.menu_bar.menu_file.action_file_save.setShortcut(
            QtGui.QKeySequence('Ctrl+S'))
        self.menu_bar.menu_file.action_print.setShortcut(
            QtGui.QKeySequence('Ctrl+P'))
        self.menu_bar.menu_file.action_open.setShortcut(
            QtGui.QKeySequence.Open)

        self.menu_bar.menu_eas.action_preferences.setShortcut(
            QtGui.QKeySequence.Preferences)

    def show_anno_type_selector(self):
        self.anno_type_selector.show()

        # this ensures that when the button is clicked, this menu will
        # be in front of the Main Window
        #
        self.anno_type_selector.activateWindow()
        self.anno_type_selector.raise_()

    def show_sensitivity_selector(self):
        self.sensitivity_selector.show()
        self.sensitivity_selector.activateWindow()
        self.sensitivity_selector.raise_()

    # method: keyPressEvent
    #
    # these methods are reimplementations of events, these allow us
    # to check when the shift button is being held,
    # used by DemoEventLoop.zoom_to_timescale
    #
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.ctrl_held = True

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.ctrl_held = False

    # method: update_play_button
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called whenever we want the text of play_button to change.
    #
    def update_play_button(self):

        if self.play_button.text() == "Play":
            self.play_button.setText("Stop")
        else:
            self.play_button.setText("Play")

    # method: open_play_menu
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called when we click the play button, if it is displaying "play",
    # then we want to open DemoPlayMenu and continue, if not we stop the play function
    #
    def open_play_menu(self):

        if self.play_button.text() == "Play":
            self.play_menu.show()
            self.play_menu.activateWindow()
            self.play_menu.raise_()
            self.update_play_button()
        else:
            self.update_play_button()

            # we call play_function() here in order to stop the playing
            #
            self.play_menu.play_function()

    def update_montage_used(self,
                            montage_used):
        self.info_bar.label_montage_used.setText(
            "Montage Being Used:   " + montage_used)

    def update_map_used(self,
                        map_used):
        self.map_info_bar.label_map_used.setText(
            "Map Being Used:   " + map_used)
