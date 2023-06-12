#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/conversion/helpers/demo_pdf_creator.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from reportlab.lib.pagesizes import landscape, A3
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

import os

#--------------------------------------------------------------------------
#
# file: DemoPdfCreator
#
# this class takes in the tmp png files created, and sets them
# into a pdf format
#
class DemoPdfCreator():

    # method: __init__
    #
    # arguments: none
    #
    # returns: none
    #
    # this method initializes DemoPdfCreator and it's attributes
    #
    def __init__(self):

        self.vertical_offset_in_inches = 1 * inch
        self.horizontal_offset_in_inches = 1 * inch
        self.layout = landscape(A3)

        self.pdf_width = 1000
        self.pdf_height = 700

    # method: create_pdf_file
    #
    # arguments:
    #  -edf_file_name_a: file path and name of the edf file to be converted
    #  -tmp_dir_a: directory file path of /tmp/
    #
    # returns: none
    #
    # this method uses reportlab to setup a pdf file and append
    # the already created png files
    #
    def create_pdf_file(self,
                        edf_file_name_a,
                        tmp_dir_a):

        pdf_file_name = edf_file_name_a.rsplit(".", 1)[0] + ".pdf"

        # sets up pdf file
        #
        pdf_file = canvas.Canvas(pdf_file_name,
                                 self.layout)

        # iterates over all files in tmp directory
        #
        for root,dirs,files in os.walk(tmp_dir_a):
            for png_file in files:
        
                if png_file.endswith('.png'):
                    png_file = tmp_dir_a + png_file
                    pdf_file.drawImage(png_file,
                                       self.horizontal_offset_in_inches,
                                       self.vertical_offset_in_inches,
                                       self.pdf_width,
                                       self.pdf_height)
                
                    pdf_file.showPage()
        
        pdf_file.save()
    #
    # end of method
