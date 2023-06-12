#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/sigplots/demo_sigplot_waveform.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from .demo_plot_widget import DemoPlotWidget
import numpy as np

class DemoWaveform(DemoPlotWidget):
    def __init__(self,
                 cfg_dict_a=None,
                 time_scale_=None,
                 **kwargs):

        # **kwargs allows us to pass the axis from the calling function as:
        # axisItems={'bottom':axis}
        # allowing us to use the DemoTimeAxis module here
        #
        super(DemoWaveform, self).__init__(**kwargs)
        self.cfg_dict = cfg_dict_a

        font = QtGui.QFont()
        font.setPointSize(10)
        self.setFont(font)

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

        # define a color for drawing the signals. This can be changed
        # as an option in menu/preference in future versions.
        # TODO: setup from config file
        #
        self.signal_color_pen = self.cfg_dict['signal_color_pen']

    def do_plot(self):

        self.clear()

        self.setYRange(-500, 500)
        self.curve = self.plot(self.t_data,
                               self.y_data)

        # do the color style thing
        #
        self.curve.setPen(self.signal_color_pen)

    def set_signal_data(self,
                        t_data_a,
                        y_data_a):

        self.t_data = t_data_a
        self.y_data = y_data_a

    def update_preferences(self,
                           signal_color_tuple_a):
        self.signal_color_pen = signal_color_tuple_a
        self.do_plot()

    def draw_zoom_to_timescale_line(self,
                                    pos):
        pen = pg.mkPen('r', width=1, style=QtCore.Qt.SolidLine)

        line = pg.InfiniteLine(pos=pos,
                               angle=90,
                               pen=pen,
                               movable=False)
        self.addItem(line)
