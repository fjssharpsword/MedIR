#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/conversion/helpers/demo_json_creator.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph import QtGui

import json

#--------------------------------------------------------------------------
#
# file: DemoJsonCreator
#
# this class calls functions to set and write a json matrix of a
# specified converted edf file
#
class DemoJsonCreator():

    # method: __init__
    #
    # arguments:
    #  -paging_function_a: function argument that when called, pages
    #                        the edf file by the timescale length.
    #
    # returns: none
    #
    # this method initializes DemoJsonCreator and it's attributes
    #
    def __init__(self,
                 paging_function_a):

        self.paging_function = paging_function_a

    # method: create_json_file
    #
    # arguments:
    #  -widget_to_convert_a: the actual widget to be converted
    #  -start_time_a: start time of edf conversion
    #  -end_time_a: end time of edf conversion
    #  -time_scale_a: sets time scale of converted edf
    #  -edf_file_name_a: file path and name of the edf file to be converted
    #
    # returns: none
    #
    # this method gets pixel information of QPixmap, sets information
    # into a json matrix, and writes to a file
    #
    def create_json_file(self,
                         widget_to_convert_a,
                         start_time_a,
                         end_time_a,
                         time_scale_a,
                         edf_file_name_a):

        json_list = []
        pixel_x_position = 0

        # gets the widget plot for one page of the edf file
        #
        for plot_pos in range(start_time_a,
                              end_time_a - 1,
                              time_scale_a):

            try:
                self.paging_function(plot_pos)
            except:
                pass

            # creates QPixmap and QImage of widget
            #
            pixmap_of_png = QtGui.QPixmap.grabWidget(widget_to_convert_a)
    
            image_for_json = pixmap_of_png.toImage()

            end_of_page = pixel_x_position + image_for_json.width()

            # iterates through all x pixels on one page
            #
            for i in range(pixel_x_position,
                           end_of_page):

                # creates a matrix of empty lists
                #
                json_list.append([])

                # iterates through all y pixels
                for j in range(image_for_json.height()):
                    pixel_rgb_value =  image_for_json.pixel(i - pixel_x_position,
                                                            j)
                    pixel_gray_value = pixel_rgb_value % 256

                    json_list[i].append(pixel_gray_value)

            # set new pixel_x_position to the last pixel on the page
            #
            pixel_x_position = i

        json_file_name = edf_file_name_a.rsplit(".", 1)[0] + ".json"

        # dumps the list matrix of pixel information into json_file_name
        #
        json.dump({edf_file_name_a:json_list}, open(json_file_name, 'w'))
    #
    # end of method
