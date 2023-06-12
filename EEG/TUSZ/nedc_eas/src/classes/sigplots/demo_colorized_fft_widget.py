#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/sigplots/demo_colorized_fft_widget.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph import QtGui, QtCore, PlotWidget
import pyqtgraph as pg

import numpy as np

from .demo_time_axis import DemoTimeAxis
from .demo_colorized_fft_plot import DemoColorizedFFTPlot

class DemoColorizedFFTWidget(QtGui.QWidget):

    def __init__(self,
                 dict_rhythms,
                 montage_names_a):

        super(DemoColorizedFFTWidget, self).__init__()
        self.dict_rhythms = dict_rhythms
        self.montage_names = montage_names_a
        self.layout_grid = QtGui.QGridLayout(self)

        self.init_other_widgets()
        self.setFixedHeight(500)
        self.setFixedWidth(400)
        self._init_sigplot()
        self.layout_grid.addWidget(self.fft_plot, 1, 0, 1, 8)

        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    # method: _init_sigplot
    #
    # arguments: time_scale_a sets the initial time scale on first load
    #
    # return: None
    #
    # This method creates the main window in which the signals are
    # drawn for the page which deals with the waveform-only montage view
    #
    def _init_sigplot(self,
                      time_scale_a=None):

        bottom_axis = DemoTimeAxis(orientation='bottom', use_time=False)
        left_axis = DemoTimeAxis(orientation='left', use_time=False)

        # TODO: This is the width of the fixed window!! fix this
        #
        bottom_axis.setWidth(323)

        self.fft_plot = DemoColorizedFFTPlot(axisItems={'bottom': bottom_axis,
                                                        'left': left_axis},
                                             dict_rhythms=self.dict_rhythms)
        
        # set some stylistic parameters
        #
        font = QtGui.QFont()
        font.setPointSize(10)
        self.fft_plot.setFont(font)

        # change the color of signalPlotWidget to white.
        #
        self.fft_plot.setBackground('w')

        # we only want to see the vertical ticks
        #
        self.fft_plot.showGrid(x=True, y=True, alpha=250)

        # disable some mouse activities of signalPlotWidget including
        # right-click and zoom.
        #
        self.fft_plot.setMenuEnabled(False)
        self.fft_plot.setMouseEnabled(False, False)
        self.fft_plot.setClipToView(clip=True)

        p = pg.mkPen((30, 30, 10), width=1)
        self.fft_plot.getAxis('bottom').setPen(p)
        self.fft_plot.getAxis('bottom').setLabel("Frequency", "Hz")
        self.fft_plot.getAxis('bottom').setRange(0, 30)
    #
    # end of function

    def set_signal_data(self,
                        t_data_a,
                        y_data_a):

        # store data in this class, so when we update
        # y_data we can pass both to fft_plot
        #
        self.t_data = t_data_a
        self.y_data = y_data_a

        if not hasattr(self, 'channel_index'):
            index = 0
        else:
            index = self.channel_index
        data_channel = self.y_data[index]

        self.fft_plot.set_signal_data(t_data_a,
                                      data_channel)

    def set_window_info(self,
                        slider_pos,
                        time_scale_a):

        self.fft_plot.set_window_info(slider_pos,
                                      time_scale_a)
    def init_other_widgets(self):
        self.channel_label = QtGui.QLabel("Channel Selector")
        self.layout_grid.addWidget(self.channel_label,0,0,1,1)
        
        self.channel_dropdown = QtGui.QComboBox()
        self.layout_grid.addWidget(self.channel_dropdown,0,1,1,1)

        for key in self.montage_names:
            self.channel_dropdown.addItem(self.montage_names[key])

        self.channel_dropdown.currentIndexChanged.connect(self.update_signal_data)

    def update_signal_data(self,
                           index):
        self.channel_index = index
        self.set_signal_data(self.t_data,
                             self.y_data)
        self.fft_plot.do_plot()
        
