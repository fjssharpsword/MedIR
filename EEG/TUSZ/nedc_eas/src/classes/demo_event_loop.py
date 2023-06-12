#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/demo_event_loop.py
#
# this file includes the controlling event loop logic for a demo of EAS.
# this file also serves to connect various user controls to their respective
# functions.
#
# ------------------------------------------------------------------------------

# import these components from pyqtgraph rather than PyQt4 or PyQt5
#
from pyqtgraph import QtGui, QtCore

# system modules
#
import os
import sys
import numpy as np

# locally defined modules
#
from .conversion.demo_converter import DemoConverter
from .config.demo_preferences import DemoPreferences

from .sigplots.demo_sigplots_widget import DemoSigplotsWidget
from .sigplots.demo_colorized_fft_widget import DemoColorizedFFTWidget

from .anno.demo_annotator import DemoAnnotator
from .dsp.filter.demo_dsp_filter import DemoDSPFilter
from .ui.main_window.demo_main_window import DemoMainWindow
from .ui.montage.demo_montage_definer import DemoMontageDefiner
from .ui.map.demo_map_definer import DemoMapDefiner

from .search import demo_search

# nedc modules
#
from nedc_mont_tools import Montage
from nedc_edf_tools import Edf
import nedc_file_tools as nft

# EK - 04.15.2017: This is the documentation for the original
# calculation of what is, in effect, a magic number:
#
# scale = 100f -->  100 microVolt / 10 mm . Since each mm is  K pixel it
# means 100 MVol/k10 the output is :  K/scale.
# Note: corr added later  to match with nikolet.
# each mm is around 3.7 pixels 10 mm is 37 pixels.
#
GAIN_FACTOR = 80
MAX_PLAY_SPEED = 2000

# ------------------------------------------------------------------------------
#
# classes are listed here
#
# ------------------------------------------------------------------------------

# class: DemoEventLoop
#


class DemoEventLoop(QtCore.QEventLoop):
    def __init__(self,
                 command_line_mode=False,
                 montage_file_to_use=None,
                 ann_map=None,
                 xml_schema=None):
        QtCore.QEventLoop.__init__(self)

        # orient demo with current path. Any directory orientation should
        # probably go here
        #
        self.init_path()

        # parse the config file and generate some dictionaries with config data
        # also, define some parameters which are used for plot widget and its
        # axes which will be reassigned after reading an EDF file.
        #
        self.init_preferences()
        self.init_montage(montage_file_to_use)
        self.init_map(ann_map)
        self.init_xml(xml_schema)
        self.preferences.set_montage(self.montage_file)

        self.montage_definer = DemoMontageDefiner()
        self.montage_definer.setup_load_function(self)
        self.edf = Edf()

        self.map_definer = DemoMapDefiner()
        self.map_definer.setup_load_function(self)

        # user_interface - includes menu_bar, frame, tool_bar, dropdown menus,
        # info_bar with session info, horizontal slider, etc.
        #
        self.ui = DemoMainWindow(self.preferences.dict_main_window,
                                 self.map_file,
                                 self.montage_names)

        if command_line_mode is False:
            self.ui.show()
            self.ui.activateWindow()
            self.ui.raise_()

        self.montage_definer.signal_definer_closed.connect(
            self.ui.setEnabled)

        self.ui.map_info_bar.update_map_used_label(self.map_file)
        self.ui.info_bar.update_montage_used_label(self.montage_file)

        # initialize widget that will be responsible for all signal display
        #
        self.sigplots_widget = DemoSigplotsWidget(
            self.ui.frame,
            self.y_max,
            self.time_scale,
            self.number_signals,
            self.montage_names,
            self.preferences.dict_sigplot_misc,
            self.preferences.dict_waveform,
            self.preferences.dict_spectrogram,
            self.preferences.dict_energy,
            self.preferences.dict_order)

        # TODO: move somewhere better
        # connect each spectrogram plot to the color lookup table
        #
        for spectro in self.sigplots_widget. \
                page_mixed_view.dict_spectrograms.values():
            self.preferences.widget.tab_spectrogram.lut_widget.addImageItem(
                spectro.img)
            self.ui.sig_resized.connect(spectro.update_image_windowing)

        # connect the mixed view page to the label cursor display
        # method by iterating over each plot in the mixed view
        # page
        #
        for plot in self.sigplots_widget. \
                page_mixed_view.dict_sigplots_all.values():
            plot.sig_mouse_moved.connect(self.label_cursor_display)

        # add the signal plot widget to the main window's frame
        #
        self.ui.frame_layout.addWidget(self.sigplots_widget)

        self.sensitivity_value_list = []
        initial_sens = int(
            self.preferences.dict_main_window['initial_sensitivity'])
        for i in range(self.number_signals):
            self.sensitivity_value_list.append(initial_sens)

        self.sigplots_widget.gnrl_set_sensitivity(
            [GAIN_FACTOR / val for val in self.sensitivity_value_list])

        # this needs to be initiated before connecting the menu
        # functions.
        #
        self.freq_filter = DemoDSPFilter(
            self.ui.input_dialogue_for_filter_frequency_selection,
            self.preferences.dict_rhythms,
            self.preferences.dict_detrender,
            self.preferences.dict_filters_default)

        self.fft_widget = DemoColorizedFFTWidget(self.preferences.dict_rhythms,
                                                 self.montage_names)
        self.annotator = DemoAnnotator(self.sigplots_widget,
                                       self.map_file,
                                       self.preferences.dict_roi,
                                       self.time_scale,
                                       self.montage_names,
                                       self.ui,
                                       self.montage_file,
                                       self.xml_schema)

        self.init_connect_slider_cursor_dropdown_functions()

        # create ints to store zoom_to_timescale selections
        #
        self.first_click_bound = -1
        self.second_click_bound = -1

        # define a main method for searching annotations including searching
        # for first, last, next and previous annotation.
        #
        self.init_connect_navigation_actions()

        # initialize the converter by providing it a function to page
        # through the edf
        #
        self.converter = DemoConverter(self.ui.horizontal_slider.setValue,
                                       self.demo_directory)

        # give the menu bar and preferences widget one-time access to
        # this loop, so that they have all of the functions they need
        #
        self.ui.menu_bar.connect_actions_to_event_loop(self)
        self.preferences.setup_updating_functions(self)

        self.ui.play_menu.setup_play_function(self)

        self.first_run_setup_done = False

        self.view_page_one_flag = True
    #
    # end of function

    def make_demo_search_window(self):
        if hasattr(self, 'edf_file'):
            demo_search.DemoSearch(self.edf_file)
        else:
            demo_search.DemoSearch()

    # method: init_path
    #
    # arguments: None
    #
    # returns: None
    #
    # this method orients the program to the local directory structure.
    # this is necessary for edf loading, processing
    # TODO: make more flexible!
    #
    def init_path(self):
        self.calling_directory = str(os.path.dirname(
            os.path.realpath(sys.argv[0])))
        self.demo_directory = (self.calling_directory).replace("src", "")
        self.d_sep_char = os.sep

    # method: init_preferences
    #
    # arguments: None
    #
    # returns: None
    #
    # this method declares a configuration file, initializes a
    # configuration parser, and creates a number of dictionaries of
    # configuration options. These dictionaries are organized by
    # configuration option type (ie annotation colors) and all begin
    # with the prefix cfg_dict.
    #
    def init_preferences(self):

        self.config_file = self.calling_directory + self.d_sep_char + \
            "defaults" + self.d_sep_char + \
            "eas.cfg"

        self.preferences = DemoPreferences(self.config_file)

        # the length of raw_channels-axis.
        #
        self.y_max = float(self.preferences.dict_main_window['y_max'])

        self.pref_slider_initial_position = int(
            self.preferences.dict_main_window['slider_initial_position'])

        self.pref_initial_time_scale = self.time_scale = int(
            self.preferences.dict_main_window['initial_time_scale'])

        # the length of time axis.
        #
        self.total_time_recording = float(
            self.preferences.dict_main_window['total_time_recording'])

        # TODO: read from cfg file
        #
        self.buffer_extra_size_in_seconds = 10.0

    #
    # end of function

    def init_map(self, ann_map_a):

        if ann_map_a is not None:
            self.map_file = ann_map_a
        else:
            self.map_file = nft.get_fullpath(
                self.preferences.dict_ann_map['ann_eeg_map'])

    def init_xml(self, xml_schema):

        if xml_schema is not None:
            self.xml_schema = xml_schema
        else:
            self.xml_schema = nft.get_fullpath(
                self.preferences.xml_schema['xml_schema'])

    def init_montage(self,
                     montage_file_a):

        if montage_file_a is not None:
            self.montage_file = montage_file_a
        else:
            self.montage_file = nft.get_fullpath(self.preferences.dict_montage['prev_montage'])

        self.montage_module = Montage(self.montage_file)

        self.montage_names = {}
        i = 0
        for key in self.montage_module.montage_d:
            self.montage_names.update({i: key})
            i = i + 1
        self.number_signals = len(self.montage_names)
    #
    # end of function

    # method: init_connect_navigation_actions
    #
    # arguments: none
    #
    # return: none
    #
    # This method sets up the four methods used for finding annotations based
    # on which push button is pressed (first, last, next, previous).
    # It also connects page forward and page backward to keyboard shortcuts
    # 'f' and 'b'
    #
    def init_connect_navigation_actions(self):

        self.ui.tool_bar.push_button_first_annotation.clicked.connect(
            self.find_first_annotation)
        self.ui.tool_bar.push_button_last_annotation.clicked.connect(
            self.find_last_annotation)
        self.ui.tool_bar.push_button_next_annotation.clicked.connect(
            self.find_next_annotation)
        self.ui.tool_bar.push_button_previous_annotation.clicked.connect(
            self.find_previous_annotation)

        self.forward_shortcut = QtGui.QShortcut(self.ui)
        self.forward_shortcut.setKey(QtCore.Qt.Key_Right)
        self.forward_shortcut.activated.connect(self.navigate_forward)

        self.backward_shortcut = QtGui.QShortcut(self.ui)
        self.backward_shortcut.setKey(QtCore.Qt.Key_Left)
        self.backward_shortcut.activated.connect(self.navigate_backward)

        # navigate by page using "," and "."
        self.page_forward_shortcut = QtGui.QShortcut(self.ui)
        self.page_forward_shortcut.setKey(QtCore.Qt.Key_Period)
        self.page_forward_shortcut.activated.connect(self.page_forward)

        self.page_backward_shortcut = QtGui.QShortcut(self.ui)
        self.page_backward_shortcut.setKey(QtCore.Qt.Key_Comma)
        self.page_backward_shortcut.activated.connect(self.page_backward)

    #
    # end of function

    # method: init_connect_slider_cursor_dropdown_functions
    #
    # arguments: none
    #
    # return: none
    #
    # This method defines the actions of the horizontal navigation slider
    #
    # the dropdown menus are connected to functions defined in this
    # module. If they they are connected earlier, than they try to call
    # functions which require signal data, which has not been acquired yet.
    #
    def init_connect_slider_cursor_dropdown_functions(self):

        # get the cursor time display to work for the only waveform
        # view by connecting its mouse movement signal with the
        # self.label_cursor_display function
        #
        self.sigplots_widget.page_only_waveform.signal_plot.scene().\
            sigMouseMoved.connect(self.label_cursor_display)

        self.sigplots_widget.page_only_waveform.signal_plot.scene().\
            sigMouseClicked.connect(self.zoom_to_timescale)

        for plot in self.sigplots_widget. \
                page_mixed_view.dict_waveform_plots.values():
            plot.sigMouseClicked.connect(self.zoom_to_timescale)

        # When slider is moved by the user the method of
        # self.time_range_display_changed() will be called.
        #
        self.ui.horizontal_slider.valueChanged. \
            connect(self.time_range_display_changed)

        # define the actions of dropdowns.
        # this is where the dropdown menus are connected to functions in
        # TODO: update this comment
        #
        self.ui.dropdown_time_scale.currentIndexChanged.connect(
            self.time_scale_changed)
        self.ui.dropdown_sensitivity.currentIndexChanged.connect(
            self.prepare_for_update_sensitivity)
        self.ui.sensitivity_selector.signal_sens_changed.connect(
            self.update_sensitivity)
        self.ui.sig_spec_frequency_changed.connect(
            self.sigplots_widget.page_mixed_view.spec_frequency_changed)
    #
    # end of function

    # method: first_run_setup
    #
    # arguments: none
    #
    # return: none
    #
    def first_run_setup(self):

        self.first_run_setup_done = True

        self.ui.enable_tools_on_edf_read()
        self.annotator.enable_all_user_interaction()

    def read_header(self,
                    edf_file_a):

        # check the header. If there is an error during the process of
        # reading edf file an error message will be shown.
        #
        try:
            # check the header. If there is an error during the process of
            # reading edf file an error message will be shown.
            #
            status = self.edf.is_edf(fname=edf_file_a)

            if status is True:
                header, _ = self.edf.read_edf(edf_file_a, scale=False)
                self.total_time_recording = \
                    header['ghdi_num_recs'] * header['ghdi_dur_rec']

            return True
        except AssertionError:
            QtGui.QMessageBox.information(self.ui,
                                          "Error",
                                          "The EDF file is not valid.",
                                          QtGui.QMessageBox.Ok)
            return False

    # This method opens a dialog box to let the user selects an EDF file.
    def prompt_user_for_open_edf_file(self):
        if not hasattr(self, 'previous_dir'):
            self.previous_dir = self.demo_directory

        # open the dialog box
        #
        new_edf_file, _ = QtGui.QFileDialog.getOpenFileName(
            self.ui,
            "Open EDF File",
            self.previous_dir,
            "EDF Files (*.edf *.edfx);;All files (*.*)")

        new_edf_file = str(QtCore.QDir.toNativeSeparators(new_edf_file))
        self.previous_dir = os.path.dirname(os.path.abspath(new_edf_file))

        if new_edf_file:
            self.open_edf_file(new_edf_file)

    def _open_edf_file_set_timing_info(self):

        self.ui.horizontal_slider.valueChanged.disconnect()

        # initialize some variables for reading signals in
        #
        self.time_axis_start = self.slider_current_pos = \
            self.pref_slider_initial_position
        self.time_axis_end = self.time_axis_start + \
            self.pref_initial_time_scale

        # move slider back to initial position
        # TODO: simplify this
        #
        self.ui.horizontal_slider.setValue(self.time_axis_start)
        self.ui.horizontal_slider.set(self.time_scale,
                                      self.total_time_recording)

        # sets the number of seconds that the edf will be read into
        # the future and past (to speed up local movement)
        #
        self.set_buffer_boundaries()

        self.annotator.set_total_time_recording(self.total_time_recording)

        self.ui.horizontal_slider.valueChanged.connect(
            self.time_range_display_changed)

    # method: open_edf_file
    #
    # arguments: optional file_a argument that should only be used
    # when launching initial or new instances of demo_event_loop class
    #
    # return: none
    #
    # wraps self.read_edf_file to make it behave properly with the gui.
    #
    def open_edf_file(self,
                      file_a,
                      ignore_annotations=False):

        # to increase efficiency after opening different new edf files.
        #
        try:
            sys.exc_clear()
        except:
            pass
        sys.exc_traceback = sys.last_traceback = None

        self.edf_file = str(file_a)

        self.ui.setWindowTitle("NEDC EAS:   " + file_a)

        if ignore_annotations is False:
            self.annotator.setup_on_opening_new_edf(file_a)

        header_found_and_valid = self.read_header(file_a)
        if header_found_and_valid is False:
            return

        self._open_edf_file_set_timing_info()

        raw_channels = self.read_edf_file()

        header_and_montage_match = self.montage_module.check(
            raw_channels, self.montage_module.montage_d)
        self.montage_minuend = self.montage_module.get_minuends()

        try:
            self.montage_subtrahend = self.montage_module.get_subtrahends()
        except:
            self.montage_subtrahend = []
            for i in range(len(self.montage_minuend)):
                self.montage_subtrahend.append(None)

        # if the header does not match, or we have no signals in montage
        #
        if (header_and_montage_match is False) or (self.number_signals == 0):
            self.setup_montage_definer()
            self.ui.setEnabled(False)
            return

        self.y_data = self.montage_module.apply(
            raw_channels, self.montage_module.montage_d)

        # convert y_data
        """
        i = 0
        self.y_data = {}
        for key in y_data:
            self.y_data.update({i:y_data[key]})
            i = i + 1
        """
        self.freq_filter.update_sample_rate(self.sampling_rate)
        self.sigplots_widget.page_mixed_view.set_sampling_rate(
            self.sampling_rate)
        self.preferences.set_sampling_rate(self.sampling_rate)
        self.fft_widget.fft_plot.set_sampling_rate(self.sampling_rate)

        self.display_patient_recording_info()

        if self.first_run_setup_done is False:
            self.first_run_setup()

        self.time, self.signal = self.potentially_filter_frequency()
        self.sigplots_widget.gnrl_set_signal_data(self.time, self.signal)
        self.fft_widget.set_signal_data(self.time, self.signal)

        # show the signals, labels, annotations
        #
        self.eeg_plotter()
        self.slider_current_position = 0
        self.ui.horizontal_slider.setValue(self.slider_current_position)

    #
    # end of function

    # method: setup_montage_definer
    #
    # arguments: none
    #
    # returns: none
    #
    # this method gives montage_definer the information needed to setup
    # it's dropdowns and list_views.
    #
    def setup_montage_definer(self):
        self.montage_definer.set_montage(self.montage_minuend,
                                         self.montage_subtrahend,
                                         self.channels_read_from_edf,
                                         self.montage_names)
        self.montage_definer.show_montage_definer()

    # method: load_new_montage
    #
    # arguments:
    #  -montage_file_name_a: file name of montage to be read
    #
    # returns: none
    #
    # this method allows us to open and load new montages at will,
    # it creates a new instance of DemoEventLoop, with a new config
    # file and new montage and closes itself.
    #
    def load_new_montage(self,
                         montage_file_name_a):

        # create new event loop, with new montage file
        #
        new_loop = DemoEventLoop(montage_file_to_use=montage_file_name_a)

        # if we had an edf file in this loop, open it in the next loop
        #
        if hasattr(self, 'edf_file'):
            new_loop.open_edf_file(self.edf_file)

        # delete this loop, as well as all it's other widgets,
        # also stop this loop entirely.
        #
        self.deleteLater()
        self.ui.deleteLater()
        self.sigplots_widget.deleteLater()
        self.montage_definer.deleteLater()
        self.exit()

    # method: load_new_map
    #
    # arguments
    #  -montage_file_name_a: file name of montage to be read
    #
    # returns: none
    #
    # this method allows us to open and load new montages at will,
    # it creates a new instance of DemoEventLoop, with a new config
    # file and new montage and closes itself.
    #
    def load_new_map(self, map_file_name_a):

        # create new event loop, with new montage file
        #
        if self.montage_file == nft.get_fullpath(self.preferences.dict_montage['prev_montage']):
            new_loop = DemoEventLoop(ann_map=map_file_name_a)
        else:
            new_loop = DemoEventLoop(ann_map=map_file_name_a, montage_file_to_use=self.montage_file)

        # if we had an edf file in this loop, open it in the next loop
        #
        if hasattr(self, 'edf_file'):
            new_loop.open_edf_file(self.edf_file)

        # delete this loop, as well as all it's other widgets,
        # also stop this loop entirely.
        #
        self.deleteLater()
        self.ui.deleteLater()
        self.sigplots_widget.deleteLater()
        self.map_definer.deleteLater()
        self.exit()


    # method: read_edf_file
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is a wrapper for demo_edf_reader.load_edf()
    # it also handles informing self.freq_filter of the nyquist limit
    #

    def read_edf_file(self):

        # read header and raw data from edf file.
        #
        self.header, raw_sig = self.edf.read_edf(self.edf_file, scale=True)

        # assign to attributes from header
        #
        self.sampling_rate = self.header['sample_frequency']
        self.channels_read_from_edf = self.header['chan_labels']
        self.total_time_recording = self.header['ghdi_num_recs'] * \
            self.header['ghdi_dur_rec']
        self.local_patient_identification = self.header['ltpi_patient_id'] + "   " \
            + self.header['ltpi_gender'] \
            + self.header['ltpi_dob'] + " "\
            + self.header['ltpi_full_name'] + " " \
            + self.header['ltpi_age']
        self.startdate_recording = self.header['ghdi_start_date']
        self.starttime_recording = self.header['ghdi_start_time']

        return raw_sig

    # for having a smoother slider. This is for reading raw data from
    # (time_axis_start - self.buffer_extra_size_in_seconds) to
    # (time_axis_end + self.buffer_extra_size_in_seconds).
    #
    def set_buffer_boundaries(self):

        buffer_start = self.time_axis_start - self.buffer_extra_size_in_seconds
        buffer_end = self.time_axis_end + self.buffer_extra_size_in_seconds

        # to limit the start and endpoints of time axis of buffer
        #
        buffer_start = max(buffer_start, 0)
        buffer_end = min(buffer_end, self.total_time_recording)

        # bug fix - now floating point time ranges "work"
        #
        self.time_buffer_start = int(buffer_start)
        self.time_buffer_end = int(buffer_end)

    # method: display_patient_recording
    #
    # arguments: none
    #
    # return: none
    #
    # This method displays the patient's informations on the main window
    # including the name of the patient, date of recording and start time of
    # recording.
    #
    def display_patient_recording_info(self):

        # patient name.
        #
        self.ui.info_bar.label_patient_name.setText(
            'Patient: ' + self.local_patient_identification)

        # date of recording.
        #
        self.ui.info_bar.label_date.setText(
            'Date: ' + self.startdate_recording + "  ")

        # start time of recording.
        #
        self.ui.info_bar.label_start_time.setText(
            'Start Time: ' + self.starttime_recording)
    #
    # end of function

    # method: eeg_plotter
    #
    # arguments: none
    #
    # return: none
    #
    # This method updates the waveform plot every time slider is moved by the
    # user.
    #
    def eeg_plotter(self):

        # if self doesn't have a value 'slider_current_pos', this is a good
        # indicator no edf has been loaded, and so we shouldn't proceed to plot
        #
        if not hasattr(self, 'slider_current_pos'):
            return

        # get the start and end time (potentially update from slider)
        #
        self.time_axis_start = self.slider_current_pos
        self.time_axis_end = self.slider_current_pos + self.time_scale

        # This line is checking the value of time axis when time range changes.
        # If the time is out of 200 buffer it reads new raw data records.
        #
        if ((self.time_axis_end > self.time_buffer_end) or
                (self.time_axis_start < self.time_buffer_start)):

            self.set_buffer_boundaries()

        time, signal = self.get_signal_buffer()
        self.time, self.signal = self.potentially_filter_frequency()
        self.sigplots_widget.gnrl_set_signal_data(self.time, self.signal)
        self.sigplots_widget.gnrl_set_windowing_info(self.slider_current_pos,
                                                     self.time_scale)

        self.sigplots_widget.update_all_plots()

        self.fft_widget.set_window_info(self.slider_current_pos,
                                        self.time_scale)
        self.fft_widget.fft_plot.do_plot()

        self.annotator.plot_annotations_for_current_time_window(
            self.slider_current_pos)
    #
    # end of function

    def get_signal_buffer(self):

        nsamp = self.header['sample_frequency'] * self.time_scale * 2

        time = self.time[int(self.time_axis_start):int(
            nsamp)+int(self.time_axis_start) - 38]

        signal = {}
        for i in range(len(self.signal)):
            signal.update({i: (self.signal[i][int(self.time_axis_start):int(
                nsamp)+int(self.time_axis_start) - 38])})

        return time, signal

    def potentially_filter_frequency(self):

        # get the array of time for the entire file
        #
        nstep = 1/self.header['sample_frequency']
        self.t_data = np.arange(0, self.total_time_recording, step=nstep)

        if self.freq_filter.non_trivial_filter_selected:

            filter_delay = int(self.freq_filter.phase_delay)
            filtered_y_data = {}

            i = 0
            for key in self.montage_module.montage_d:
                filtered_y_data[i] = \
                    self.freq_filter.do_filter(self.y_data[key])[filter_delay:]
                i += 1
            # TODO: there is a cleaner way to do this indexing
            #
            offset = len(self.t_data) - filter_delay

            return (self.t_data[:offset], filtered_y_data)
        else:
            return (self.t_data, self.y_data)

    # method: label_cursor_display
    #
    # arguments:
    #  -mouse_event: an event listener used to track the mouse
    #
    # return: none
    #
    # This method displays the current position of the slider in front of the
    # "Cursor:" textbox in the main window.
    #
    def label_cursor_display(self,
                             mouse_event):
        if self.view_page_one_flag:
            plot = self.sigplots_widget.page_only_waveform.signal_plot
            mouse_in_seconds = plot.get_mouse_secs_if_in_plot(mouse_event)

        else:
            for plot in self.sigplots_widget. \
                    page_mixed_view.dict_sigplots_all.values():

                mouse_in_seconds = plot.get_mouse_secs_if_in_plot(mouse_event)
                if mouse_in_seconds is not None:
                    break
        try:
            mouse_temp, ms = divmod(mouse_in_seconds, 1)
            ms = round(ms, 2) * 100
            mouse_temp, seconds = divmod(mouse_temp, 60)
            hours, minutes = divmod(mouse_temp, 60)
        except:
            return

        ms = str(int(ms)).zfill(2)
        seconds = str(int(seconds)).zfill(2)
        minutes = str(int(minutes)).zfill(2)
        hours = str(int(hours)).zfill(2)

        cursor_disp_string = (hours + ":" + minutes + ":" + seconds + "." + ms)

        self.ui.tool_bar.label_cursor.setText(cursor_disp_string)
    #
    # end of method

    # method: zoom_to_timescale
    #
    # arguments:
    #  -mouse_event: event holding a Point of where the mouse was clicked
    #
    # returns: none
    #
    # this method allows the user to ctrl+Left click in 2 locations, and
    # zoom to that timescale. The first click while create an indicator,
    # showing the user where the left bound will be. The second click
    # determines what the new timescale will be.
    #
    def zoom_to_timescale(self,
                          mouse_event):

        # when we are viewing just waveform
        #
        if self.view_page_one_flag is True:
            plot = self.sigplots_widget.page_only_waveform.signal_plot
        else:

            # get access to an arbitrary waveform plot
            #
            plot = self.sigplots_widget.page_mixed_view. \
                dict_waveform_plots.values().next()

        # we only want to do this if the user is holding ctrl
        #
        if self.ui.ctrl_held is True:

            # case of first click
            #
            if self.first_click_bound == self.second_click_bound == -1:

                # get x value of Point of mouse_event in seconds
                #
                point_in_seconds = plot.plotItem.vb.mapSceneToView(
                    mouse_event.pos())

                # x value of Point
                #
                self.first_click_bound = point_in_seconds.x()

                # draw indicator line
                #
                if self.view_page_one_flag is True:
                    self.sigplots_widget.page_only_waveform.\
                        draw_zoom_to_timescale_line(self.first_click_bound)
                else:
                    for plot in self.sigplots_widget.page_mixed_view. \
                            dict_waveform_plots.values():
                        plot.draw_zoom_to_timescale_line(
                            self.first_click_bound)

            # case of second click
            #
            elif self.second_click_bound == -1:

                # get x value of Point of mouse_event in seconds
                #
                point_in_seconds = plot.plotItem.vb.mapSceneToView(
                    mouse_event.pos())

                self.second_click_bound = point_in_seconds.x()

                # find difference of bounds, make sure positive
                #
                new_time_scale = abs(round(self.second_click_bound -
                                           self.first_click_bound))

                # this prevents problems with setting time scale after updating
                #
                self.ui.dropdown_time_scale.setCurrentIndex(-1)

                # change dropdown to this new_time_scale, this will update all
                # other modules
                #
                self.ui.dropdown_time_scale.lineEdit().setText(str(new_time_scale))
                self.time_scale_changed()

                # update slider_current_pos to the first click pos
                #
                self.slider_current_pos = round(self.first_click_bound)
                self.ui.horizontal_slider.setValue(self.slider_current_pos)

                # reset values for next method call
                #
                self.first_click_bound = self.second_click_bound = -1
    #
    # end of function

    # method: play_edf_file
    #
    # arguments:
    #  -play_speed: int between 1 and 10, divides MAX_PLAY_SPEED constant to find
    #                 how fast we should play the edf file
    #  -interval: interval at which to navigate by
    #
    # returns: none
    #
    # this method is called when start is clicked on the DemoPlayMenu, this method
    # starts a timer, that calls a navigation function at a certain rate.
    #
    def play_edf_file(self,
                      play_speed=1,
                      interval=1):

        # we scale the play_speed to our timer value here, max value of play_speed
        # is 200, because the max value from the slider is 10.
        #
        play_speed = MAX_PLAY_SPEED / play_speed
        self.play_interval = interval

        if not hasattr(self, "timer"):
            self.timer = QtCore.QTimer()

        # kind of odd logic, we want the edf to play while the button
        # says Stop, not Start
        #
        if self.ui.play_button.text() == "Stop":
            self.timer.start(play_speed)
            self.timer.timeout.connect(self.navigate_forward_by_interval)
        else:
            self.timer.stop()

    # method: switch_views
    #
    # arguments: none
    #
    # return: none
    #
    # This method is called through the menu options, and it is used to switch
    # between combinations of the three available views: spectrogram, waveform,
    # and energy.
    # it is essentially a wrapper to the channels_wdiget.switch_views method
    # TODO: THIS SHOULD BE IMPLEMENTED USING SIGNAL.emit()
    #
    def switch_views(self):

        view_status_dict = self.ui.menu_bar.menu_view.get_view_status_dict()

        self.sigplots_widget.switch_views(view_status_dict)

        self.view_page_one_flag = (view_status_dict['waveform']
                                   and not view_status_dict['spectrogram']
                                   and not view_status_dict['energy'])

        # get a boolean to pass to the following functions which
        # display or hide spectrogram dropdown menus and labels
        #
        disp_spectro = view_status_dict['spectrogram']
        self.ui.label_spectro_freq.setVisible(disp_spectro)
        self.ui.dropdown_spectro_low.setVisible(disp_spectro)
        self.ui.dropdown_spectro_high.setVisible(disp_spectro)

        # these lines fix the problem when switching views, sometimes the view
        # would not plot correctly (mostly not plotting at all). these lines
        # tell the event loop to process all events waiting to finish, then replot.
        #
        self.processEvents()
        self.eeg_plotter()
    #
    # end of function

    # method: time_scale_changed
    #
    # arguments:
    #  -current: current time index in plot
    #
    # return: none
    #
    #  This method is run when time scale is changed.
    #
    def time_scale_changed(self):

        try:
            self.time_scale = float(self.ui.dropdown_time_scale.currentText())

        # this exception is thrown when there is nothing in the combobox
        #
        except:
            pass

        self.annotator.set_time_scale(self.time_scale)

        self.eeg_plotter()

        self.ui.horizontal_slider.set(self.time_scale,
                                      self.total_time_recording)
        self.sigplots_widget.setFocus(True)
    #
    # end of function

    # method: prepare_for_update_sensitivity
    #
    # arguments:
    #  -index: index of sensitivity dropdown
    #
    # returns: none
    #
    def prepare_for_update_sensitivity(self,
                                       index):

        # get sensitivity value at corresponding index
        #
        value = self.ui.dropdown_sensitivity.currentText()

        # send these to the sensitivity selector
        #
        self.ui.sensitivity_selector.set_all_dropdowns(index, value)

        self.sigplots_widget.setFocus(True)

    # method: update_sensitivity
    #
    # arguments: none
    #
    # return: none
    #
    # This method is run when sensitivity or channel comboboxes are changed.
    #
    def update_sensitivity(self):

        self.sensitivity_value_list = self.ui.sensitivity_selector.get_sensitivity_list()

        self.sensitivity_scale = \
            [GAIN_FACTOR / val for val in self.sensitivity_value_list]

        self.sigplots_widget.gnrl_set_sensitivity(self.sensitivity_scale)

        self.eeg_plotter()
    #
    # end of function

    # method: time_range_display_changed
    #
    # arguments: none
    #
    # return: none
    #
    # This method updates the plot when the slider is moved.
    #
    def time_range_display_changed(self):

        # Gets the current position of slider.
        #
        self.slider_current_pos = self.ui.horizontal_slider.sliderPosition()

        if hasattr(self, 'edf_file'):
            # updates signal plots according to new position of the slider.
            #
            self.eeg_plotter()
    #
    # end of function

    # method: find_first_annotation
    #
    # arguments: none
    #
    # return: none
    #
    # This method navigates to the first non-filtered annotation
    #
    def find_first_annotation(self):
        try:
            first_annotation_location = \
                self.annotator.get_first_annotation()

            # set the slider in the location of first annotation.
            #
            self.slider_current_pos = first_annotation_location
            self.ui.horizontal_slider.setValue(self.slider_current_pos)

        except:
            pass
    #
    # end of function

    # method: find_last_annotation
    #
    # arguments: none
    #
    # return: none
    #
    # This method navigates to the last non-filtered annotation
    #
    def find_last_annotation(self):
        try:
            last_annotation_location = \
                self.annotator.get_last_annotation()

            # set the slider in the location of first annotation.
            #
            self.slider_current_pos = last_annotation_location
            self.ui.horizontal_slider.setValue(self.slider_current_pos)

        except:
            pass
    #
    # end of function

    # method: find_next_annotation
    #
    # arguments: none
    #
    # return: none
    #
    # This method navigates to the next non-filtered annotation
    #
    def find_next_annotation(self):
        try:
            next_annotation_location = \
                self.annotator.get_next_annotation(self.slider_current_pos)

            # set the slider in the location of first annotation.
            #
            self.slider_current_pos = next_annotation_location
            self.ui.horizontal_slider.setValue(self.slider_current_pos)
        except:
            pass
    #
    # end of function

    # method: find_previous_annotation
    #
    # arguments: none
    #
    # return: none
    #
    # This method navigates to the previous non-filtered annotation
    #
    def find_previous_annotation(self):
        try:
            previous_annotation_location = \
                self.annotator.get_previous_annotation(self.slider_current_pos)

            # set the slider in the location of first annotation.
            #
            self.slider_current_pos = previous_annotation_location
            self.ui.horizontal_slider.setValue(self.slider_current_pos)

        except:
            pass
    #
    # end function

    # TODO: everything from here down should be moved into DemoMain Window
    def page_forward(self):
        self.slider_current_position = self.slider_current_pos + self.time_scale
        self.ui.horizontal_slider.setValue(self.slider_current_position)

    def page_backward(self):
        self.slider_current_position = self.slider_current_pos - self.time_scale
        self.ui.horizontal_slider.setValue(self.slider_current_position)

    def navigate_forward(self):
        self.slider_current_position = self.slider_current_pos + 1
        self.ui.horizontal_slider.setValue(self.slider_current_position)

    def navigate_backward(self):
        self.slider_current_position = self.slider_current_pos - 1
        self.ui.horizontal_slider.setValue(self.slider_current_position)

    def navigate_forward_by_interval(self):
        self.slider_current_position = self.slider_current_pos + self.play_interval
        self.ui.horizontal_slider.setValue(self.slider_current_position)

    def print_edf_file(self,
                       make_json_a=False,
                       timescale_to_use_a=None,
                       start_time_to_use_a=None,
                       end_time_to_use_a=None,
                       edf_to_convert_a=None,
                       signal_type_to_print_a=None):

        # save horizontal position for later reset
        #
        old_horizontal_pos = self.ui.horizontal_slider.sliderPosition()

        # create some variables to feed to the printer function
        #
        if timescale_to_use_a is None:
            timescale_to_use_a = self.time_scale

        if start_time_to_use_a is None:
            start_time_to_use_a = 0

        if end_time_to_use_a is None:
            end_time_to_use_a = self.total_time_recording - 1

        if edf_to_convert_a is None:
            try:
                edf_to_convert_a = self.edf_file
            except:
                pass
        widget_to_print = self.sigplots_widget

        self.converter.convert_edf_file(make_json_a,
                                        int(start_time_to_use_a),
                                        int(end_time_to_use_a),
                                        int(timescale_to_use_a),
                                        widget_to_print,
                                        edf_to_convert_a)

        # reset horizontal slider to previous position
        #
        self.ui.horizontal_slider.setValue(old_horizontal_pos)

    def show_fft_plot(self):
        self.fft_widget.show()
        self.fft_widget.activateWindow()
        self.fft_widget.raise_()
    # updates attributes from preferences general tab
    # TODO: this should be moved into DemoMainWindow
    #

    def update_general_preferences(self,
                                   time_scale_a,
                                   sensitivity_a,
                                   window_width_a,
                                   window_height_a):

        # this prevents problems with setting time scale after updating
        #
        self.ui.dropdown_time_scale.setCurrentIndex(-1)
        self.ui.dropdown_time_scale.lineEdit().setText(time_scale_a)
        self.time_scale_changed()

        self.ui.dropdown_sensitivity.lineEdit().setText(sensitivity_a)
        self.update_sensitivity()

        self.ui.resize(window_width_a,
                       window_height_a)
