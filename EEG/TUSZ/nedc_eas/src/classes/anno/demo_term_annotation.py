#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/anno/demo_term_annotation.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------

import pyqtgraph as pg

class DemoTermAnnotation(pg.LinearRegionItem):
    def __init__(self,
                 start,
                 end,
                 brush,
                 annotation_string):

        values = [start, end]
        bounds = [0, 10000]
        
        pg.LinearRegionItem.__init__(self,
                                     values,
                                     bounds=bounds,
                                     movable=False)
        
        self.setBrush(brush)

        htmltext = (
            '<div style="text-align: center"> \
            <span style="color: #000; font-size: 10pt;">' +
             annotation_string + '</span></div>')

        self.label = pg.TextItem(html=htmltext,
                                 anchor=(0, 0),
                                 border=None,
                                 fill=None)
