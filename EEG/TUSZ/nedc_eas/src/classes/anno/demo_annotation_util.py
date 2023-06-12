#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/anno/demo_annotation_util.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
# ------------------------------------------------------------------------------
import os
from pyqtgraph.Qt import QtGui

from .demo_label_utils import LabelConverterUtils

# these constants are defined elsewhere to:
# 1) document the data structure
# 2) to avoid hard-coding
# 3) allow for sharing these constants between modules
#
from .demo_annotation_data_representation import *

PREF_ANNO_TYPE = ['csv', 'xml', 'tse', 'lbl']

# class: DemoAnnotatorUtil
#
# this class handles the reading and writing of annotation files,
# utilizes nedc_label_utils methods to achieve this
#
class DemoAnnotationUtil:

    # method: __init__
    #
    # arguments:
    #  -dict_*: dictionaries required to construct LabelConverterUtils
    #
    # returns: none
    #
    # this method constructs DemoAnnotationUtil, and sets it's attributes
    #
    def __init__(self,
                 dict_labels_priority_a,
                 dict_event_map_a,
                 dict_priority_map_a,
                 ann_map_map_file_a,
                 xml_schema):

        self.dict_labels_priority = dict_labels_priority_a
        self.dict_event_map = dict_event_map_a
        self.dict_priority_map = dict_priority_map_a

        self.edf_file_path = ""

        self.lcu = LabelConverterUtils(self.dict_labels_priority,
                                       self.dict_event_map,
                                       self.dict_priority_map,
                                       ann_map_map_file_a,
                                       xml_schema)

        self.channels_to_ignore = [22]  # EKG

    # method: read_file
    #
    # arguments:
    #  file_a: file to read
    #
    # returns: annotations (raw list of annotations)
    #
    # this method sets the internal variable anno_file_name and
    # reads the uses LabelConverterUtils to read the label file
    #
    def read_file(self,
                  file_a):

        self.anno_file_name = None
        self.edf_file_path = os.path.abspath(file_a)
        name_no_extension = os.path.splitext(file_a)[0]

        for extension in PREF_ANNO_TYPE:
            potential_file = name_no_extension + "." + extension
            if os.path.isfile(potential_file):
                self.extension_type = extension
                self.anno_file_name = potential_file
                break

        if self.anno_file_name is not None:
            read_annotations = self.lcu.nedc_read_labels(self.anno_file_name)
            
        else:
            read_annotations = []

        annotations = []

        for channel in read_annotations:
            for annotation in read_annotations[channel]:

                left_bound = annotation[0]
                right_bound = annotation[1]
                type = annotation[2]

                if type != 0:
                    annotations.append([channel,
                                        left_bound,
                                        right_bound,
                                        type])
        return annotations

    # method: write_annotations_to_file
    #
    # arguments:
    #  -dict_annotation_a: dictionary of all annotations stored in DemoAnnotator
    #
    # returns: none
    #
    # this method is called when the user clicks 'save' in the menu bar,
    # calls NedcLabelUtil methods to write to file
    #
    def write_annotations_to_file(self,
                                  dict_annotations_a,
                                  ext):

        # checks to see if an annotation file exists
        #
        if hasattr(self, 'anno_file_name'):

            # modify extension of file coresspoindingly to ext user chooses
            # make sure that it is not None before proceeding
            #
            try:
                fname_no_extension, anno_ext = os.path.splitext(
                    self.anno_file_name)

                if (anno_ext != "." + ext):
                    self.anno_file_name = fname_no_extension + "." + ext

                    # check if file exists, if not, then save as
                    #
                    if not os.path.isfile(self.anno_file_name):
                        self.save_as_annotations_to_file(dict_annotations_a, ext)

                elif self.anno_file_name != None:
                    annotations_to_save = self.prepare_annotations_for_writing(
                        dict_annotations_a)
                    if self.anno_file_name.endswith(".csv"):
                        self.lcu.nedc_write_csv(self.anno_file_name,
                                                annotations_to_save)
                    if self.anno_file_name.endswith(".xml"):
                        self.lcu.nedc_write_xml(self.anno_file_name,
                                                annotations_to_save)
                    if self.anno_file_name.endswith(".lbl"):
                        self.lcu.nedc_write_lbl(self.anno_file_name,
                                                annotations_to_save)
                    if self.anno_file_name.endswith(".lab"):
                        self.lcu.nedc_write_lab(self.anno_file_name,
                                                annotations_to_save)
                else:
                    mb = QtGui.QMessageBox()
                    mb.setIcon(mb.Icon.Warning)
                    mb.setText(".csv file is not found")
                    mb.exec_()
            except:
                self.save_as_annotations_to_file(dict_annotations_a, ext)

        # else call save as method
        #
        else:
            self.save_as_annotations_to_file(dict_annotations_a, ext)

    # method: prepare_annotations_for_writing
    #
    # arguments:
    #  -dict_annotations_a: dictionary of annotations stored in DemoAnnotator
    #
    # returns: dictionary of annotations to be saved
    #
    # this method checks whether an annotation should be written, and prepares
    # the data to be inputted into NedcLabelUtil
    #
    def prepare_annotations_for_writing(self,
                                        dict_annotations_a):
        list_annotations_to_save = []
        for anno in dict_annotations_a.values():
            channel = anno[CHAN_IND]
            anno_type = anno[TYPE_IND]

            # add annotation to list if the annotation is not on the EKG channel
            # and if the channel is not null
            #
            if channel not in self.channels_to_ignore:
                if anno_type != 0:
                    list_annotations_to_save.append(anno)

        # annotation[1][0] says to sort by the start of annotation
        # example annotation: [channel, [start, end, type]]
        #                                  ^
        #                            annotation[1][0]
        #
        list_annotations_to_save.sort(
            key=lambda annotation: annotation[L_BOUND_IND])

        dict_annotations_to_save = {}
        for anno in list_annotations_to_save:
            channel = anno[CHAN_IND]

            start_time = anno[L_BOUND_IND]
            end_time = anno[R_BOUND_IND]
            annotation_type = anno[TYPE_IND]
            annotation_value = [start_time, end_time, annotation_type]

            dict_annotations_to_save. \
                setdefault(channel, []).append(annotation_value)

        return dict_annotations_to_save

    # method: save_as_annotations_to_file
    #
    # arguments:
    #  -dict_annotations_a: dictionary of annotations stored in DemoAnnotator
    #
    # returns: none
    #
    # this method performs the same actions as write_annotations_to_file,
    # however also gets the file name to be written to by the user
    #
    def save_as_annotations_to_file(self,
                                    dict_annotations_a, ext):

        annotations_to_save = self.prepare_annotations_for_writing(
            dict_annotations_a)

        anno_file_path = self.edf_file_path.replace(".edf", ".csv")

        input_file_name_as_QString, _ = QtGui.QFileDialog.getSaveFileName(
            None,
            "Save File",
            anno_file_path,
            "CSV files (*.csv);;XML files (*.xml);;LBL files (*.lbl);;TSE files (*.tse)")

        self.anno_file_name = str(input_file_name_as_QString)
        if self.anno_file_name.endswith(".csv"):
            self.lcu.nedc_write_csv(self.anno_file_name,
                                    annotations_to_save)
        if self.anno_file_name.endswith(".xml"):
            self.lcu.nedc_write_xml(self.anno_file_name,
                                    annotations_to_save)
        if self.anno_file_name.endswith(".tse"):
            self.lcu.nedc_write_tse(self.anno_file_name,
                                    annotations_to_save)
        if self.anno_file_name.endswith(".lbl"):
            self.lcu.nedc_write_lbl(self.anno_file_name,
                                    annotations_to_save)

        # does not support .lbl and .lab yet
        #
        """
        if self.anno_file_name.endswith(".lbl"):
            self.lcu.nedc_write_lbl(self.anno_file_name,
                                    annotations_to_save)
        if self.anno_file_name.endswith(".lab"):
            self.lcu.nedc_write_lab(self.anno_file_name,
                                    annotations_to_save)
        """
