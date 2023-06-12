#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/sigplots/demo_sigplot_spectrogram.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from classes.dsp.demo_dsp_taper import DemoDSPTaper
from classes.dsp.demo_dsp_buffer import DemoDSPBuffer
from classes.dsp.demo_psd_calculator import DemoPSDCalculator

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: DemoSpectrogram
#
# This class contains methods to read multiple types of label files
# and covert them. It inherits from pyqtgraph GraphicsView widget,
# which in turn is a wrapper for pyqtgraph GraphicsItem item.
#
# FOOD FOR THOUGHT: perhaps replace first line of __init__ with this?
# Would improve performance, but is buggy
# (see line 39 in path/to/pyqtgraph/__init__.py)
# suepr(DemoSpectrogram, self).__init__(useOpenGL=True)

#
class DemoSpectrogram(pg.GraphicsView):

    # used to indicate to parent classes where the mouse is on the
    # screen (only in x coordinates, at least for now)
    #
    sig_mouse_moved=QtCore.Signal(object)

    # method: __init__
    #
    # arguments:
    #  - cfg_dict_a:   sets various parameters, ie nfft, zero_padding, etc.
    #  - time_scale_a: how many seconds on page
    #
    # returns: none
    #
    # this method initializes the DemoSpectrogram class, including its
    # component classes DemoDSPTaper, DemoDSPBuffer, and
    # DemoPSDCalculator
    #
    def __init__(self,
                 cfg_dict_a,
                 time_scale_a=None):
        super(DemoSpectrogram, self).__init__()

        # read some defaults from config dictionary
        #
        self.nfft = int(cfg_dict_a['nfft'])
        self.window_size = float(cfg_dict_a['window_size'])
        self.lower_freq_bound = int(cfg_dict_a['default_lower_freq_bound'])
        self.upper_freq_bound = int(cfg_dict_a['default_upper_freq_bound'])

        decimation_factor = float(cfg_dict_a['decimation_factor'])
        window_type = cfg_dict_a['window_type']

        # set up the class that will handle the various window
        # functions for us this is a convenience class wrapping the
        # functions found here:
        # https://docs.scipy.org/doc/numpy/reference/routines.window.html
        #
        self.taper = DemoDSPTaper(window_type)

        # this class will handle the windowing, framing, general buffering
        # needed to compute the array of spectra
        #
        self.buffer = DemoDSPBuffer(decimation_factor,
                                    use_zero_padding=True,
                                    taper=self.taper)

        # this will compute the array of spectra referred to as a spectrogram
        #
        self.psd_calc = DemoPSDCalculator(self.nfft)

        # set up the pg.ImageItem that will be used to display the spectrogram
        #
        self.init_image()

        # initialize a flag to check if we are at right end of eeg
        # used to ensure image windowing is correct
        #
        self.end_of_eeg_flag = False

    # method: init_image
    #
    # arguments: None
    #
    # returns: None
    #
    # this method sets up a pg.ImageItem and addreses other graphical issues
    #        
    def init_image(self):

        # at the end of the eeg we want the default color of this plot
        # to match the generally white background.
        #
        self.setBackground('w')

        # set up image which will be used to display spectrogram
        #
        self.img = pg.ImageItem()
        self.addItem(self.img)
        self.img.setParent(self)

        # this rectangle ensures that the spectrogram fills up the
        # space allocated to it
        # 
        range_rectangle = QtCore.QRectF(0, 0, 0, 0)
        range_rectangle.setTop(10000)
        range_rectangle.setBottom((-1) * 10000)
        self.setRange(range_rectangle)

    # method: update_image_windowing
    #
    # arguments: None
    #
    # return: None
    #
    # calling this method should ensure that the spectrogram image is
    # maximized given the available real estate in the
    # DemoSpectrogram widget
    #
    def update_image_windowing(self):

        # get the dimensions of the available screen real-estate
        #
        y_size = self.height()
        x_size = self.width()

        # the origin is the top left hand corner of the image
        #
        y_origin = -y_size / 2
        x_origin = -x_size / 2

        # if we are at the end of the edf, the size of the plot must be resized
        # to reflect this and correspond with the width of the other views
        #
        if self.end_of_eeg_flag is True:

            # ensure that the spectrogram is scaled correctly in the
            # horizontal direction (be aligned with the waveform)
            #
            rescale_quotient = self.samp_rate * self.time_scale / self.t_range

            # update the paramter that will determine the size of the plot
            #
            x_size = x_size / rescale_quotient

            self.end_of_eeg_flag = False

        # this check needs to take place because the
        # update_image_windowing method might be called before the
        # self.do_plot has ever been called. This might happen if the
        # user resizes the main window without having ever opening the
        # spectrogram view.
        #
        if hasattr(self, 'image_array_narrowed_in_freq'):

            # this does actual maximization
            #
            self.img.setRect(QtCore.QRectF(
                QtCore.QPointF(x_origin, y_origin),
                QtCore.QSizeF(x_size, y_size)))
    #
    # end of method

    # method: set_signal_data
    #
    # arguments:
    #  -t_data_a: an array of linearly spaced data corresponding to time
    #  -y_data_a: the data which is later chopped up and fft'd
    #
    # return: None
    #
    # a simple set method
    #
    def set_signal_data(self,
                        y_data_a):

        self.buffer.set_data(y_data_a)
    #
    # end of method

    def set_sampling_rate(self,
                          sampling_rate_a):
        self.samp_rate = sampling_rate_a

        # update self.zero_padding and tapering window
        #
        window_size_in_samples = int(self.window_size * self.samp_rate)
        self.zero_padding = self.nfft - window_size_in_samples
        self.taper.set_taper(window_size_in_samples)

        self.set_freq_range(self.lower_freq_bound,
                            self.upper_freq_bound)
    # method: set_x_range
    #
    # arguments:
    #  -left_index_a: index in self.t_data to start ffts on
    #  -time_scale_a: time scale to used to determine
    #    how far to the right the right index should be
    #
    # return: None
    #
    # this method is similiar to the pyqtgraph.PlotWidget.setXRange()
    # method it informs this widget what parts of self.buffer.data)
    # to use when creating the display
    #
    def set_x_range(self,
                    left_index_a,
                    time_scale_a):

        # update the time scale, and set the variable dist between ffts
        # to correspond to this time_scale
        # higher time scales have a greater distance between ffts because
        # if we computed the same number of ffts for larger time scales, there
        # would be a major slowdown.
        # float is to account for the most broad case (such as 0.5)
        #
        self.time_scale = float(time_scale_a)

        # left index marks the beginning of self.buffer.frame, and
        # happens to be the first index for which an fft will be computed
        # right index corresponds to the end of self.buffer.frame, and
        # will be around the final index for which an fft will be computed
        #
        right_index = self.get_right_index_and_end_of_eeg_test(left_index_a,
                                                               time_scale_a)

        self.t_range = right_index - left_index_a

        # what will be used to compute n ffts, where:
        # n = len(self.buffer.frame) / dist_between_fft_windows
        #
        self.buffer.set_frame(left_index_a,
                              right_index)

        self.buffer.setup_for_make_window_array(self.time_scale,
                                                self.nfft,
                                                self.zero_padding)
    #
    # end of method

    # method: get_right_index_and_end_of_eeg_test
    #
    # arguments:
    #  left_index_a: used to compute right index if not at end of eeg
    #  time_scale_a: used to compute right index if not at end of eeg
    #
    # return:
    #  right_index: index around which the rightmost fft will be computed
    #
    # test for end of eeg in self.update_image_windowing(). this is
    # used to make sure the spectrogram is scaled correctly so as to
    # align with the waveform
    #
    def get_right_index_and_end_of_eeg_test(self,
                                            left_index_a,
                                            time_scale_a):
        # this is the right index if we are not at the end of the eeg
        #
        right_index = int(left_index_a +
                          (time_scale_a * self.samp_rate)- 1)

        # lets figure out where the end of the eeg is
        #
        last_index_in_eeg = len(self.buffer.data) - 1

        # if we are at the end of the eeg, correct right_index and set
        # the flag for use in self.update_image_windowing().
        # Otherwise, ensure that the flag is not set.
        #
        if right_index > last_index_in_eeg:
            right_index = last_index_in_eeg
            self.end_of_eeg_flag = True
        else:
            self.end_of_eeg_flag = False

        return right_index
    #
    # end of function

    # method: set_freq_range
    #
    # arguments:
    #  low_a - lower bound in hertz
    #  high_a - upper bound in hertz
    #
    # return: None
    #
    # this method sets the lower and upper frequency bounds for the
    # display of the spectrogram.
    #
    def set_freq_range(self,
                       low_a,
                       high_a):

        # check bounds for outrageous conditions
        #
        if low_a >= high_a:
            return

        # set some frequency bounds. these aren't used for anything except for
        # the immediately proceeding calculation of self.lower_freq_ind and
        # self.upper_freq_ind
        # TODO: do we really need to make thes self.*?
        #
        self.lower_freq_bound = low_a
        self.upper_freq_bound = high_a

        self.lower_freq_ind = int(( low_a * self.nfft / self.samp_rate))
        self.upper_freq_ind = int((high_a * self.nfft / self.samp_rate) + 1)
    #
    # end of method

    # method: do_plot
    #
    # arguments: None
    #
    # return: None
    #
    # this is the big dog method - it does the actual plotting of the
    # spectrogram. It does:
    # 1) gets the necessary array of windows from self.buffer
    # 2) gets the psd calculator to compute the spectrogram
    # 3) deals with the user selected frequency window
    # 4) actually sets the image in place
    #
    def do_plot(self):

        #  create the array of windows, each of which will correspond to
        #   -one fft
        #   -one vertical spectrum
        #   -one pixel width
        #     not literal pixel, strechable via self.update_image_windowing()
        #
        window_array = self.buffer.make_window_array()

        # calc spectrogram for entire frequency range (0 to sample_rate / 2)
        # 
        self.full_freq_array = self.psd_calc.calc_spectra_array(window_array)

        # get indices for columns to replace with markers
        #
        n_columns = self.full_freq_array.shape[0]
        marker_indices = [int(round(n_columns * i / 10.0)) for i in range(1, 10)]

        # make the time markers
        #
        for marker_index in marker_indices:
            self.full_freq_array[marker_index, :] = 0

        # chop up the result to correspond to the appropriate frequency range
        # also, flip the array so the lower frequencies are on bottom.
        # finally, set the image in place.
        #
        self.chop_image_array_for_frequency_and_set_image()
    #
    # end of method

    # method: chop_image_array_for_frequency_and_set_image
    #
    # arguments: None
    #
    # returns: None
    #
    # chop up the self.full_freq_array to correspond to the
    # appropriate frequency range. also flips self.full_freq_array so
    # the the low frequencies are on the bottom
    #
    def chop_image_array_for_frequency_and_set_image(self):

        # resize to account for frequency selection
        #
        self.image_array_narrowed_in_freq = \
            self.full_freq_array[:, self.lower_freq_ind:self.upper_freq_ind]

        # flips the image vertically
        #  presumably we want:
        #   higher frequencies on top ~~~~~~~~~~~~~~~~~
        #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #   lower frequencies on bottom ~~~~~~~~~~~~~~~
        #
        self.image_array_narrowed_in_freq = \
            self.image_array_narrowed_in_freq[:, ::-1]

        # put the image in place
        #
        self.img.setImage(self.image_array_narrowed_in_freq,
                          transform=self.transform())
        self.update_image_windowing()

    # method: update_preferences
    #
    # arguments:
    #  - nfft_a:         int - size of nfft to compute
    #  - zero_padding_a: int - how many zeros to append (for temporal resolution)
    #  - ratio_a:        float -ratio time scale to distance between fft indices.
    #                    approach 0 for very high resolution but slow execution.
    #                    recommended range is 0.2 ~ 1.0
    #  - taper_a:        string - for example 'hanning' or 'Kaiser', for setting
    #                    tapering function
    #
    # returns: None
    #
    # this method allows the user to update some or all of the various
    # parameters represented in the arguments list. By design, it doesn't do
    # great error checking - for example, the user of this method must ensure that
    # 1) zero_padding_a <= nfft_a
    # 2) nfft_a and zero_padding_a are ints, ratio_a can be cast as float,
    # 3) taper_a is represented in the dict DemoDSPTaper.tapering_fctn_options
    #    (although upper-case doesn't matter)
    #
    def update_preferences(self,
                           nfft_a,
                           window_size_a,
                           ratio_a,
                           taper_a):
        self.nfft = nfft_a
        self.window_size = window_size_a

        # update self.zero_padding and tapering window
        #
        window_size_in_samples = int(self.window_size * self.samp_rate)
        self.zero_padding = self.nfft - window_size_in_samples
        self.taper.set_taper(window_size_in_samples)
        
        # we have got to update these buffer.indices to reflect
        # possible changes in nfft
        #
        self.set_freq_range(self.lower_freq_bound,
                            self.upper_freq_bound)

        # update_the buffer
        #
        self.buffer.set_performance_ratio(ratio_a)
        self.buffer.setup_for_make_window_array(self.time_scale,
                                                self.nfft,
                                                self.zero_padding)

        # update self.psd_calc with new nfft value
        #
        self.psd_calc.set_nfft(self.nfft)

        # update the plot
        #
        self.do_plot()
    #
    # end of function

    # method: get_mouse_in_secs_if_in_plot
    #
    # arguments: event_a:
    #
    # returns: None OR x coordinate in seconds (I belive found using self.t_data)
    #
    # this method will return None if event_a did not originate on
    # this class's screen real estate. However, it will return the
    # time in seconds if it does belong to this class. This allows
    # another class to loop over various plots (other signal views
    # have similar methods) and break when the correct view (and hence
    # correct x-coordinate) is found
    #
    def get_mouse_secs_if_in_plot(self,
                                     event_a):
        if self.img.boundingRect().contains(event_a):
            return event_a.x()

    # method: mouseMoveEvent
    #
    # arguments: event_a
    def mouseMoveEvent(self, event_a):
        qpoint = QtCore.QPoint(event_a.x(), event_a.y())
        self.sig_mouse_moved.emit(qpoint)
