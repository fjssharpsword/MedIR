#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/sigplots/demo_sigplots_page_only_waveform.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph import QtGui, QtCore, PlotWidget
import pyqtgraph as pg
import numpy as np

from .demo_time_axis import DemoTimeAxis
from .demo_plot_widget import DemoPlotWidget

PADDING_CONSTANT = 0.0001

# TODO: TOO MANY MAGIC NUMBERS IN HERE

class DemoSigplotsPageOnlyWaveform(QtGui.QWidget):
    def __init__(self,
                 parent=None,
                 cfg_dict_a=None,
                 y_max_a=None,
                 initial_time_scale_a=None):
        QtGui.QWidget.__init__(self, parent=parent)

        self.y_max = y_max_a
        self.cfg_dict = cfg_dict_a
        # define a color for drawing the signals. This can be changed as an
        # option in menu/preference in future versions.
        # TODO: setup from config file
        #
        self.signal_color_pen = self.cfg_dict['signal_color_pen']

        self.layout = QtGui.QGridLayout(self)

        # set some style paramaters
        #
        self.layout.setMargin(0)
        self.layout.setSpacing(0)

        # initialize the widget which will show the labels on the signal plot
        # and add it to the page's layout
        #
        self._init_labels()
        self.layout.addWidget(self.label,
                              0, 0, 1, 1)

        # initialize the widget which will actually show the signals
        # and add it to the page's layout
        #
        self._init_sigplot(initial_time_scale_a)
        self.layout.addWidget(self.signal_plot,
                              0, 1, 1, 1)

        # show grids
        #
        self.signal_plot.showGrid(x=True,
                                  y=False,
                                  alpha=250)

    def _init_labels(self):
        self.label = PlotWidget(self)

        # set size (TODO: is this necessary?) 16777215 is 2^24
        #
        self.label.setMaximumSize(QtCore.QSize(70, 16777215))

        # set some stylistic parameters
        #
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                 "background-color: rgb(240, 240, 240);")

        # make background white
        #
        self.label.setBackground('w')

        # hide y-axis
        #
        self.label.plotItem.hideAxis('left')

        # hide x-axis by defining the x-axis as white. We don't hide
        # x-axis directly to have the equal height in both of the
        # signalPlotWidget and montageNamePlotWidget
        #
        p = pg.mkPen((255, 255, 255), width=1, style=QtCore.Qt.SolidLine)
        self.label.getAxis('bottom').setPen(p)

        # initialize the x-axis. The end of the labels will be plotted on x=1.
        #
        self.label.setXRange(0,
                             1,
                             0.001)

        # initialize the y-axis like page_waveform_only_signal_plot.
        #
        self.label.setYRange(0,
                             self.y_max,
                             padding=0.0001)

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

        axis = DemoTimeAxis(orientation='bottom')

        # initialize signal plot as a child of only_waveform
        #
        self.signal_plot = DemoPlotWidget(self,
                                          axisItems={'bottom': axis})

        # set some stylistic parameters
        #
        font = QtGui.QFont()
        font.setPointSize(10)
        self.signal_plot.setFont(font)

        # change the color of signalPlotWidget to white.
        #
        self.signal_plot.setBackground('w')

        # hide the y-axis of signalPlotWidget
        #
        self.signal_plot.plotItem.hideAxis('left')

        # define the width of every page
        #
        self.signal_plot.setXRange(0,
                                   time_scale_a,
                                   padding=PADDING_CONSTANT)

        # define the height of every page
        #
        self.signal_plot.setYRange(0,
                                   self.y_max,
                                   padding=PADDING_CONSTANT)

        # disable some mouse activities of signalPlotWidget including
        # right-click and zoom.
        #
        self.signal_plot.setMenuEnabled(False)
        self.signal_plot.setMouseEnabled(False, False)
        self.signal_plot.setClipToView(clip=True)

        # show grids on signal plot widget.
        # TODO: do we want to wait to load an edf to add grid?
        # used to be set after main's first_run
        # TODO: do we want to be able to set grid color via cfg file
        #
        p = pg.mkPen((30, 30, 10), width=1, style=QtCore.Qt.SolidLine)
        self.signal_plot.getAxis('bottom').setPen(p)
        self.signal_plot.showGrid(x=True, y=True, alpha=100)
    #
    # end of function

    def set_signal_data(self,
                        t_data_a,
                        y_data_a):
        self.t_data = t_data_a
        self.y_data = y_data_a
        self.number_signals = len(self.y_data)

    def do_plot(self):

        # reset plot
        #
        self.signal_plot.clear()

        # adjust the height of plot widget regarding the self.y_max.
        #
        self.signal_plot.setYRange(0,
                                    self.y_max,
                                    padding=PADDING_CONSTANT)

        vert_offsets = self.get_plotting_vert_offsets()

        # draw EEG signals according to number of signals read from montage.
        # reverse the range to start drawing from the top
        #
        for i in range(self.number_signals):

            # substract the DC offset, allows for more normalized plotting
            #
            self.y_data[i] -= np.mean(self.y_data[i])

            # draw the signal using data (as is obvious, self.t_data
            # and self.y_data must have read in from main file
            # previously to this plot)
            #
            curve = self.signal_plot.plot(self.t_data,
                                          self.y_data[i] + vert_offsets[i])

            # do the color style thing
            #
            curve.setPen(self.signal_color_pen)
        #
        # end of for

        # Draws an infinite line to separate labels annotations from signals
        #
        line = pg.InfiniteLine(  # @UndefinedVariable
            pos=(self.y_max - (self.y_max / (self.number_signals + 2))),
            angle=0,
            pen=self.signal_color_pen,
            movable=False)
        self.signal_plot.addItem(line)
    #
    # end of function

    def do_label(self,
                 number_signals_a,
                 montage_names_a):

        self.num_sigs = number_signals_a

        # delete any previous labels (from previously loaded edfs
        #
        self.label.clear()

        # iterate over all the channels
        #
        vert_offsets = self.get_label_vert_offsets()
        for i in range(0, self.num_sigs):

            # set text style and channel name 'suffix' (ie FP1-F7)
            #
            channel_name = montage_names_a[i]
            htmltext = '<div style="text-align: ">\
                <span style="color: #000; font-size: 10pt;">' + \
                channel_name + '</span></div>'

            # set text using pytograph's text item module
            #
            text = pg.TextItem(html=htmltext,
                               anchor=(1, 0),
                               border=None,
                               fill=None)

            # add text to item
            #
            self.label.addItem(text)

            # set item in place
            #
            vert_offset = vert_offsets[i]
            text.setPos(1, vert_offset)
        #
        # end of for

        # TODO: move somehere more sensible
        #
        self.offsets_for_annotations = self.get_plotting_vert_offsets()

        # in the case where we load a blank montage, skip declaring this variable
        # (it will get reinstantiated later)
        #
        try:
            self.height_difference_between_channels = \
                self.offsets_for_annotations[0] - self.offsets_for_annotations[1]
        except:
            self.height_difference_between_channels = 0

        self.label.setMouseEnabled(False, False)
    #
    # end of function

    def get_rectangle_in_channel_coordinates(self,
                                             position_bottom,
                                             position_top):

        # ensure that position_top is always higher than position_bottom
        #
        if position_top < position_bottom:
            temp = position_bottom
            position_bottom = position_top
            position_top = temp

        # initialize some variables to iterate through channels offsets
        # we will look for the top channel first
        # once we have found the top, we will look for the bottom
        #
        channel_count = 0
        looking_for_top = True
        looking_for_bottom = False
        offset = 0
        channel_low = 0
        channel_high = 0
        
        # iterate through channel offsets to find matches for top and bottom
        #
        for offset in self.offsets_for_annotations:

            # if we are still looking for matches to either position, continue
            #
            if looking_for_bottom or looking_for_top:

                # the first case: have we reached the top position yet?
                #
                if looking_for_top:

                    # if offset is greater than or equal to position_top,
                    # we have found the top position
                    #
                    if offset <= position_top:

                        # set channel_high (variable to be returned) to
                        # channel_count, and update the flow control variables
                        #
                        channel_high = channel_count
                        looking_for_top = False
                        looking_for_bottom = True


                # the second case: we have reached the top position.
                # but have we reached the bottom position yet?
                #
                if looking_for_bottom:

                    # if offset is greater than or equal to position_bottom,
                    # we have found the bottom position
                    #
                    if offset <= position_bottom:

                        # set channel_low (variable to be returned) to
                        # channel_count, and update the flow control variables
                        #
                        channel_low = channel_count
                        looking_for_bottom = False

            channel_count += 1

        # deal with the case where the top of the rectangle is outside
        # the range of offsets
        #
        if offset > position_bottom:
            channel_low = channel_count

        # reverse the direction from which we count channels
        #
        # channel_high = len(self.offsets_for_annotations) - channel_high
        # channel_low = len(self.offsets_for_annotations) - channel_low
        return (channel_low, channel_high)

    def get_plotting_vert_offsets(self):

        # TODO: fix magic numbers
        #
        vert_scale_factor = 0.875
        pos_bottom = 400
        pos_top = pos_bottom + (self.y_max * vert_scale_factor)

        return np.linspace(pos_top,
                           pos_bottom,
                           self.num_sigs)

    def get_label_vert_offsets(self):

        # TODO: fix magic numbers
        #
        vert_scale_factor = 0.875
        bottom = 525
        top = bottom + (self.y_max * vert_scale_factor)

        return np.linspace(top,
                           bottom,
                           self.num_sigs)

    def update_preferences(self,
                           signal_color_tuple):

        self.signal_color_pen = signal_color_tuple
        self.do_plot()

    def draw_zoom_to_timescale_line(self,
                                    pos):

        pen = pg.mkPen('r', width=1, style=QtCore.Qt.SolidLine)
        
        line = pg.InfiniteLine(pos=pos,
                               angle=90,
                               pen=pen,
                               movable=False)
        self.signal_plot.addItem(line)

