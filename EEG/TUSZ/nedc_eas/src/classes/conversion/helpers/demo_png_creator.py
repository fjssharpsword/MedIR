#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/conversion/helpers/demo_png_creator.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------

from pyqtgraph import QtGui
from pyqtgraph.Qt import QtWidgets
#---------------------------------------------------------------------
#
# file: DemoPngCreator
#
# this class creates temporary png files to be added to a pdf file
#
class DemoPngCreator():

    # method: __init__
    #
    # arguments:
    #  -paging_function_a: function argument that when called, pages
    #                        the edf file by the timescale length.
    #
    # returns: none
    #
    # this method initializes DemoPngCreator and it's attributes
    #
    def __init__(self,
                 paging_function_a):
        
        self.paging_function = paging_function_a
        self.page_counter = 0

    # method: create_png_files
    #
    # arguments:
    #  -widget_to_convert_a: the actual widget to be converted
    #  -start_time_a: start time of edf conversion
    #  -end_time_a: end time of edf conversion
    #  -time_scale_a: sets time scale of converted edf
    #  -tmp_dir_a: directory file path of /tmp/
    #
    # returns: none
    #
    # this method creates and adds png files to /tmp/
    #
    def create_png_files(self,
                         widget_to_convert_a,
                         start_time_a,
                         end_time_a,
                         time_scale_a,
                         tmp_dir_a):

        # create png file for every page in range
        #
        for plot_pos in range(start_time_a,
                              end_time_a - 1,
                              time_scale_a):
            
            try:
                self.paging_function(plot_pos)
            except:
                pass
            
            png_file = tmp_dir_a + str(self.page_counter) + ".png"
            
            pixmap_of_png = QtGui.QPixmap.grabWidget(widget_to_convert_a.winId())

            pixmap_of_png.save(png_file, "png")

            self.page_counter += 1
    #
    # end of method
