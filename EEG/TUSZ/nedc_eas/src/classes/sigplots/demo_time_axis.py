#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/sigplots/demo_time_axis.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
import pyqtgraph
import time
from datetime import timedelta


# This class is used to change the time axis format from seconds to HH:MM:SS.
# It inherits from AxisItem in PyQtGraph.
# Note: Windows can't handle dates before 1970
#
class DemoTimeAxis(pyqtgraph.AxisItem):  # @UndefinedVariable
    def __init__(self, orientation, use_time=True):
        super(DemoTimeAxis, self).__init__(orientation)
        self.check = 0
        self.use_time = use_time

    # replace the built in method to assign strings to ticks with one that
    # marks standard time divisions
    #
    def tickStrings(self, values, scale, spacing):
        strns = []
        if self.use_time:
            string = '%02d:%02d:%02d'
            for x in values:
                d = timedelta(seconds=x)
                hours = d.days * 24 + d.seconds // 3600
                minutes = (d.seconds % 3600) // 60
                seconds = (d.seconds % 3600 % 60)
                try:
                    strns.append((string % (hours, minutes, seconds)))
                except ValueError:
                    strns.append('')
                    self.check += 1
        else:
            for x in values:
                strns.append(str(x))

        return strns
