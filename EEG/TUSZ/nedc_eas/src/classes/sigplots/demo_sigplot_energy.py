#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/sigplots/demo_sigplot_energy.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph import QtCore

import numpy as np

from classes.dsp.demo_dsp_buffer import DemoDSPBuffer
from .demo_plot_widget import DemoPlotWidget

# class: DemoEnergy
#
# this class is responsible for calculating and plotting the energy values
# of the edf file
#
class DemoEnergy(DemoPlotWidget):

    # signal to be emitted when the mouse moves
    #
    sig_mouse_moved=QtCore.Signal(object)

    # method: __init__
    #
    # arguments: dictionary of values from energy preferences
    #
    # returns: none
    #
    def __init__(self,
                 cfg_dict_a=None,
                 **kwargs):

        # **kwargs allows us to pass the axis from the calling function as:
        # axisItems={'bottom':axis}
        # allowing us to use the DemoTimeAxis module here
        #
        super(DemoEnergy, self).__init__(**kwargs)

        self.cfg_dict = cfg_dict_a

        # get window duration in seconds
        #
        self.win_duration = self.cfg_dict['window_duration']

        self.decimation_factor =  float(self.cfg_dict[
            'decimation_factor'])

        self.plot_scheme = self.cfg_dict['plot_scheme']

        self.max_value = self.cfg_dict['max_value']

        # disable some mouse activities of signalPlotWidget including
        # right-click and zoom.
        #
        self.setMenuEnabled(False)
        self.setMouseEnabled(False, False)
        self.setClipToView(clip=True)

        # change the color of signalPlotWidget to white.
        #
        self.setBackground('w')

        # hide the x-axis of signalPlotWidget
        #
        # self.plotItem.hideAxis('bottom')

        # hide the y-axis of signalPlotWidget
        #
        self.plotItem.hideAxis('left')

        self.signal_color_pen = self.cfg_dict['signal_color_pen']

        self.buffer = DemoDSPBuffer(self.decimation_factor,
                                    use_zero_padding=False)

    # method: update_preferences
    #
    # arguments: values from the Energy Preferences Tab
    #
    # returns: none
    #
    # this method is called when we apply the preferences, it allows
    # us to dynamically update the demo with new preferences value
    #
    def update_preferences(self,
                           factor_a,
                           window_duration_a,
                           color_pen_a,
                           plot_scheme_a,
                           max_value_a):

        self.decimation_factor = factor_a
        self.win_duration = window_duration_a
        self.signal_color_pen = color_pen_a
        self.plot_scheme = plot_scheme_a
        self.max_value = max_value_a

        self.set_sampling_rate(self.samp_rate)
        self.buffer.set_performance_ratio(self.decimation_factor)
        self.buffer.setup_for_make_window_array(self.time_scale,
                                                self.win_length)

    def set_sampling_rate(self,
                          sampling_rate_a):
        self.samp_rate = sampling_rate_a
        
        # convert duration to samples
        #
        self.win_length = int(self.win_duration * self.samp_rate)

    # method: set_x_range
    #
    # arguments:
    #  -left_index_a: index in self.t_data to start plot at
    #  -time_scale_a: time scale to used to determine how far to the
    #   right the right index should be
    #
    # return: None
    #
    # this method is similiar to the pyqtgraph.PlotWidget.setXRange() method
    # it informs this widget what parts of its signal buffer (self.y_data) to
    # use when creating the display
    #
    def set_x_range(self,
                    left_index_a,
                    time_scale_a):

        # update the time scale, and set the variable dist between rms wins
        # to correspond to this time_scale
        # if we computed the same number of rms for larger time scales, there
        # would be a major slowdown.
        #
        self.time_scale = float(time_scale_a)

        # left index marks the beginning of self.windowed_y_data_for_rms, and
        # happens to be the first index for which a rms will be computed
        # right index corresponds to the end of self.windowed_y_data_for_rms,
        # will be around the final index for which an rms will be computed
        #
        left_index = int(left_index_a)

        # left index marks the beginning of self.buffer.frame, and
        # happens to be the first index for which an rms will be computed
        # right index corresponds to the end of self.buffer.frame, and
        # will be around the final index for which an rms will be computed
        #
        right_index = self.get_right_index_and_end_of_eeg_test(left_index,
                                                               time_scale_a)

        self.t_range = right_index - left_index_a

        self.windowed_t_data = self.t_data[left_index_a:
                                           right_index]

        # what will be used to compute n rmss, where:
        # n = len(self.buffer.frame) / dist_between_rms_windows
        #
        self.buffer.set_frame(left_index_a,
                              right_index)

        self.buffer.setup_for_make_window_array(self.time_scale,
                                                self.win_length)
    #
    # end of method

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

    def set_signal_data(self,
                        t_data_a,
                        y_data_a):
        self.t_data = t_data_a
        self.buffer.set_data(y_data_a)

    def do_plot(self):

        # clear previously plotted curve
        #
        try:
            if hasattr(self, "curve"):
                self.curve.clear()
        except:
            pass
        self.buffer.square_all_points()

        # get all windows, according to the window duration
        #
        window_array = self.buffer.make_window_array()
        
        energy_data = np.zeros((len(self.windowed_t_data),))

        # get the stride size, this allows us to extrapolate what the points
        # inbetween each calculated point might be
        #
        i = self.buffer.stride_size - 1

        # get first window data
        #
        first_win = window_array[0]
        prev_energy_point = self.compute_energy_for_one_window(first_win)
        window_array = np.delete(window_array, 0, 0)

        # iterate over every window
        #
        for window in window_array:

            # calculate energy value for the window
            #
            energy_point = self.compute_energy_for_one_window(window)

            # calculate the slope based on the 2 most recent points,
            # divided by the length of our stride size
            #
            slope = (energy_point - prev_energy_point) / self.buffer.stride_size

            # fill in the points between the two calculated points based slope
            #
            for j in range(self.buffer.stride_size):
                energy_data[i - j] = energy_point - (slope * j)

            # store as prev point
            #
            prev_energy_point = energy_point

            # we will start next slope calculation after the next stride size
            #
            i += self.buffer.stride_size
        #
        # end of loop

        # we have chosen logarithmic in Energy Preferences
        #
        if self.plot_scheme == "Logarithmic":

            # iterate over every point
            #
            for i in range(energy_data.size):

                # try set to dB, unless plot is at 0, like at beginning of file
                #
                try:
                    energy_data[i] = 180*np.log10(energy_data[i])
                except Exception as e:
                    energy_data[i] = 0

        self.setYRange(0, self.max_value)

        self.curve = self.plot(self.windowed_t_data,
                               energy_data)

        # do the color style thing
        #
        self.curve.setPen(self.signal_color_pen)

    def compute_energy_for_one_window(self,
                                      window):

        sum_of_squares = 0
        for point in window:
            sum_of_squares += point

        averaged = sum_of_squares / self.win_length
        rms = averaged ** 0.5

        return rms

    def get_mouse_secs_if_in_plot(self,
                                     event_a):
        if self.plotItem.sceneBoundingRect().contains(event_a):
            return self.plotItem.vb.mapSceneToView(event_a).x()

    def mouseMoveEvent(self, event_a):
        qpoint = QtCore.QPoint(event_a.x(), event_a.y())
        self.sig_mouse_moved.emit(qpoint)
