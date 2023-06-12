#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/conversion/demo_converter.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from .helpers.demo_pdf_creator import DemoPdfCreator
from .helpers.demo_png_creator import DemoPngCreator
from .helpers.demo_json_creator import DemoJsonCreator

import os

#--------------------------------------------------------------------
#
# file: DemoConverter
#
# this class holds the methods to convert edf files. It serves as
# a wrapper for the more functional classes initialized in __init__
#
class DemoConverter():

    # method: __init__
    #
    # arguments:
    #  -paging_function_a: function argument that when called, pages
    #                        the edf file by the timescale length.
    #  -demo_directory_a: string holding directory file path
    #
    # returns: none
    #
    # this method initializes DemoConverter, as well as the other classes
    # it utilizes: DemoPngCreator, DemoPdfcreator, DemoJsoncreator
    #
    def __init__(self,
                 paging_function_a,
                 demo_directory_a):

        self.paging_function = paging_function_a
        self.demo_directory = demo_directory_a
        
        self.d_sep_char = os.sep
        self.tmp_dir = self.demo_directory + \
                       "tmp" + self.d_sep_char
        self.png_creator = DemoPngCreator(self.paging_function)

        self.pdf_creator = DemoPdfCreator()

        self.json_creator = DemoJsonCreator(self.paging_function)
    #
    # end of method

    # method: convert_edf_file
    #
    # arguments:
    #  -make_json_a: True if a json should be created in place of a pdf
    #  -start_time_a: start time of edf conversion
    #  -end_time_a: end time of edf conversion
    #  -time_scale_a: sets time scale of converted edf
    #  -widget_to_convert_a: the actual widget to be converted
    #  -edf_file_name_a: file path and name of the edf file to be converted
    #
    # returns: none
    #
    # this method calls methods to either create a json file, order
    # create a tmp directory, create tmp png files, and create a pdf
    #
    def convert_edf_file(self,
                         make_json_a,
                         start_time_a,
                         end_time_a,
                         time_scale_a,
                         widget_to_convert_a,
                         edf_file_name_a):

        if make_json_a is True:
            self.json_creator.create_json_file(widget_to_convert_a,
                                               start_time_a,
                                               end_time_a,
                                               time_scale_a,
                                               edf_file_name_a)
            
        else:

            # try to make /tmp/
            #
            try:
                os.mkdir(self.tmp_dir)

            # if /tmp/ already exists:
            # remove all files and directories found in /tmp/
            #
            except OSError as e:
                print (e)
                print ("removing all tmp files...")
                for tmp_dir_path, tmp_directories, tmp_files in os.walk(self.tmp_dir):
                    for png_file in tmp_files:
                        os.remove(tmp_dir_path + png_file)

            self.png_creator.create_png_files(widget_to_convert_a,
                                          start_time_a,
                                          end_time_a,
                                          time_scale_a,
                                          self.tmp_dir)

            self.pdf_creator.create_pdf_file(edf_file_name_a,
                                             self.tmp_dir)

        # after edf conversion, remove all files in
        # tmp directory and delete tmp directory
        #
        for tmp_dir_path, tmp_directories, tmp_files in os.walk(self.tmp_dir):
            for png_file in tmp_files:
                os.remove(tmp_dir_path + png_file)
            try:
                os.rmdir(tmp_dir_path)
            except:
                pass
    #
    # end of method
