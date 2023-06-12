#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/sigplots/demo_colorized_fft_plot.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from classes.dsp.demo_dsp_buffer import DemoDSPBuffer

import numpy as np

from .demo_plot_widget import DemoPlotWidget

class DemoColorizedFFTPlot(DemoPlotWidget):

    def __init__(self,
                 dict_rhythms,
                 **kwargs):

        super(DemoColorizedFFTPlot, self).__init__(**kwargs)

        self.dict_rhythms = dict_rhythms
        self.buffer = DemoDSPBuffer(1, False)

    def set_signal_data(self,
                        t_data,
                        y_data):
        self.t_data = t_data
        self.y_data = y_data
        self.buffer.set_data(y_data)

    def set_window_info(self,
                        slider_pos,
                        time_scale_a):
        self.slider_pos = slider_pos
        self.time_scale = time_scale_a

    def set_sampling_rate(self,
                          samp_rate):
        self.sampling_rate = samp_rate

    def do_plot(self):
        try:
            if hasattr(self, "curve"):
                self.curve.clear()
                self.clear()
        except:
            pass

        # get left and right indices of current time window
        #
        left_index = int(np.where(self.t_data==self.slider_pos)[0])
        right_index = int(left_index +
                          (self.time_scale * self.sampling_rate) - 1)

        # get the data we want to plot
        #
        y = self.y_data[left_index:right_index]

        # remove DC offset
        #
        y = y-np.mean(y)

        # Number of samples
        #
        N = int(self.sampling_rate * 2)
            
        # calculate fft
        #
        y_fft = np.fft.fft(y, n=N)

        # get magnitudes of fft
        #
        magnitudes = 2.0/N * np.abs(y_fft[:N//2])

        # get frequencies based on sampling rate
        #
        frequencies = []
        for i in np.arange(N/2):
            freq = i * self.sampling_rate / N
            frequencies.append(freq)

        # set range to reasonable values
        #
        self.setXRange(0, 30, 0)
        self.setYRange(0, 20, 0)

        # plot!
        #
        self.curve = self.plot(frequencies, magnitudes)
        self.curve.setPen((70,45,20,255))

        # get rhythms
        #
        delta = self.dict_rhythms['delta']
        theta = self.dict_rhythms['theta']
        alpha = self.dict_rhythms['alpha']
        beta = self.dict_rhythms['beta']

        # get the the middle frequency, where the center of the line will be
        #
        delta_line_center = (delta[1] - delta[0])/2.0 + delta[0]
        theta_line_center = (theta[1] - theta[0])/2.0 + theta[0]
        alpha_line_center = (alpha[1] - alpha[0])/2.0 + alpha[0]
        beta_line_center = (beta[1] - beta[0])/2.0 + beta[0]

        # get the width of the axis
        #
        bottom_axis = self.getAxis('bottom')
        axis_width = float(bottom_axis.geometry().width())

        # get percentage of the entire axis, based on filter ranges
        #
        delta_width = axis_width*(delta[1]-delta[0])/30.0*2
        theta_width = axis_width*(theta[1]-theta[0])/30.0*2
        alpha_width = axis_width*(alpha[1]-alpha[0])/30.0*2
        beta_width = axis_width*(beta[1]-beta[0])/30.0*2

        # set pens with the widths from above
        #
        delta_pen = pg.mkPen(tuple((255, 0, 0, 20)), width=delta_width)
        theta_pen = pg.mkPen(tuple((255, 255, 0, 20)), width=theta_width)
        alpha_pen = pg.mkPen(tuple((0, 255, 0, 20)), width=alpha_width)
        beta_pen = pg.mkPen(tuple((0, 0, 255, 20)), width=beta_width)
        
        # draw lines
        #
        delta_line = pg.InfiniteLine(pos=delta_line_center,
                                     angle=90,
                                     pen=delta_pen,
                                     movable=False)
        
        theta_line = pg.InfiniteLine(pos=theta_line_center,
                                     angle=90,
                                     pen=theta_pen,
                                     movable=False)
        alpha_line = pg.InfiniteLine(pos=alpha_line_center,
                                     angle=90,
                                     pen=alpha_pen,
                                     movable=False)
        beta_line = pg.InfiniteLine(pos=beta_line_center,
                                    angle=90,
                                    pen=beta_pen,
                                    movable=False)

        # add to plot
        #
        self.addItem(delta_line)
        self.addItem(theta_line)
        self.addItem(alpha_line)
        self.addItem(beta_line)

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
