#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/anno/demo_annotation_plotter.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph.Qt import QtCore, QtGui

from .demo_annotation import DemoAnnotation
from .demo_term_annotation import DemoTermAnnotation

# these constants are defined elsewhere to:
# 1) document the data structure
# 2) to avoid hard-coding
# 3) allow for sharing these constants between modules
#
from .demo_annotation_data_representation import *

MOCK_TERM = False
MOCK_TERM_ANNOTATIONS = {
    9990999: ["term",   0,   1,  3],
    9991999: ["term",   1,   2,  4],
    9992999: ["term",   2, 3.5, 14],
    9993999: ["term", 6.7, 9.3,  7],
}
MOCK_TERM_KEYS = list(MOCK_TERM_ANNOTATIONS.keys())


# class: DemoAnnotationPlotter
#
# this class plots and contains the annotations in the current window.
# this class holds the DemoAnnotation objects, while DemoAnnotator holds
# the actual values to each annotation
#
class DemoAnnotationPlotter:

    # method: __init__
    #
    # arguments:
    #  -sigplots_widget_a: widget to be able to generate attributes
    #  -dict_*: dictionaries used to plot annotations
    #  -time_scale_a: time scale of demo in seconds
    #
    # returns: none
    #
    # this method constructs DemoAnnotationPlotter, and set it's attributes
    #
    def __init__(self,
                 sigplots_widget_a,
                 dict_event_map_a,
                 cfg_dict_single_annotation_a,
                 dict_anno_colors_a,
                 montage_names_a,
                 cfg_dict_map_a,
                 time_scale_a):

        self.sigplots_widget = sigplots_widget_a
        self.page_1_parent = \
            self.sigplots_widget.page_only_waveform.signal_plot.plotItem

        self.dict_event_map = dict_event_map_a

        self.cfg_dict_single_annotation = cfg_dict_single_annotation_a
        self.dict_anno_colors = dict_anno_colors_a

        self.time_scale = int(time_scale_a)

        self.set_montage(montage_names_a)

        self.anno_objects = {}

    # method: set_montage
    #
    # arguments:
    #  -montage_names_a: dictionary of channel names
    #
    # returns: none
    #
    # this method sets attributes, such as montage_names and
    # height_between_channels
    #
    def set_montage(self,
                    montage_names_a):

        self.montage_names = montage_names_a

        # update this information because it is possible that the
        # number of channels has changed, and so it is possible that
        # the displayed distance between channels has changed
        #
        page_1_offsets = self.sigplots_widget. \
                 page_only_waveform.offsets_for_annotations

        self.height_between_channels = \
            self.sigplots_widget.page_only_waveform. \
            height_difference_between_channels

        self.page_1_height_offsets = page_1_offsets \
                                     - (self.height_between_channels / 2)

    # method: plot_for_current_time_window:
    #
    # arguments:
    #  -slider_current_pos_a: current position of slider
    #  -annotations_a: annotations from DemoAnnotator
    #  -pre_computed_views_a: pre_computed_view from DemoAnnotator
    #
    # returns: none
    #
    # this method sets `self.axis_left_bound` and `self.axis_right_bound`,
    # and then iterates over `self.annotations` to find and plot any
    # annotations within the window described by these bounds
    #
    def plot_for_current_time_window(self,
                                     slider_current_pos_a,
                                     annotations_a,
                                     pre_computed_views_a,
                                     annotations_to_ignore):

        self.axis_left_bound = int(slider_current_pos_a)
        self.axis_right_bound = int(slider_current_pos_a + self.time_scale)
        self.annotations = annotations_a
        self.pre_computed_views = pre_computed_views_a

        # if there were annotations plotted on screen, find any annotations
        # that are now off screen, remove them
        #
        if len(self.anno_objects) > 0:
            keys_to_remove=[]

            for key in self.anno_objects:

                p1_anno = self.anno_objects[key][PLOT_IND].page_1_roi
                self.page_1_parent.removeItem(p1_anno)
                keys_to_remove.append(key)

            for k in keys_to_remove:
                del self.anno_objects[k]

        # if there are annotations in pre_computed_views
        #
        if len(self.pre_computed_views) > 0:

            # get annotations to be displayed on screen
            #
            view_set = self.pre_computed_views[int(self.axis_left_bound)]

            # TODO: delete once term annotations are "plugged in"
            #
            if MOCK_TERM is True:
                self.annotations.update(MOCK_TERM_ANNOTATIONS)
                view_set = view_set + MOCK_TERM_KEYS
                
            for key in view_set:
                annotation = self.annotations[key]
                anno_type_number = annotation[TYPE_IND]
         
                # check for null event, other filtered annotations:
                #
                if (anno_type_number != 0 and
                    anno_type_number not in annotations_to_ignore):

                    channel_number = annotation[CHAN_IND]

                    if channel_number != "term":
                        annotation_left_bound = annotation[L_BOUND_IND]
                        annotation_right_bound = annotation[R_BOUND_IND]

                        self.plot_anno(channel_number,
                                       annotation_left_bound,
                                       annotation_right_bound,
                                       anno_type_number,
                                       unique_key=key)
                    else:
                        self.plot_term_annotation(annotation)

    def plot_term_annotation(self,
                             annotation_a):
        annotation_left_bound = annotation_a[L_BOUND_IND]
        annotation_right_bound = annotation_a[R_BOUND_IND]
        annotation_type_number = annotation_a[TYPE_IND]

        annotation_string = self.dict_event_map[annotation_type_number]
        r, g, b, alpha = self.dict_anno_colors[annotation_string]
        brush = QtGui.QBrush(QtGui.QColor(r, g, b, alpha))

        annotation = DemoTermAnnotation(annotation_left_bound,
                                        annotation_right_bound,
                                        brush,
                                        annotation_string)
        self.page_1_parent.addItem(annotation)
        self.page_1_parent.addItem(annotation.label)

        # place the name of annotations labels approximately in
        # center of every annotations.
        #
        label_position_horizontal = (
            annotation_left_bound
            + (annotation_right_bound - annotation_left_bound) / 2)

        label_position_vertical = 10000

        annotation.label.setPos(label_position_horizontal,
                                label_position_vertical)

    # method: plot_anno
    #
    # arguments: all of the data required to create an annotation
    #
    # returns: none
    #
    # this method plots a single annotation to the window. adds
    # this annotation self.anno_objects
    #
    def plot_anno(self,
                  channel_number_a,
                  x_position_start_a,
                  x_position_end_a,
                  type_number=0,
                  is_selected=False,
                  unique_key=None):

        # get annotation attributes
        #
        height = self.page_1_height_offsets[channel_number_a]

        position = [x_position_start_a, height]

        size = [x_position_end_a - x_position_start_a,
                self.height_between_channels]
        bounds = QtCore.QRectF(0,
                               height,
                               1000000,
                               self.height_between_channels)

        annotation_name = self.dict_event_map[type_number]

        channel_name = self.montage_names[channel_number_a]

        # generate DemoAnnotation object
        #
        annotation_plot =  DemoAnnotation(
            position,
            channel_name,
            annotation_name,
            self.cfg_dict_single_annotation,
            self.dict_anno_colors,
            self.page_1_parent,
            unique_key,
            size=size,
            maxBounds=bounds)

        annotation_plot.set_style(is_selected=is_selected)

        # store DemoAnnotation object
        #
        self.anno_objects[unique_key] = [channel_number_a,
                                           x_position_start_a,
                                           x_position_end_a,
                                           type_number,
                                           annotation_plot]

    # method: remove_annotation_object
    #
    # arguments:
    #  -key: key of annotation to be removed
    #
    # returns: none
    #
    # this method is called when a user wants to remove an annotation,
    # this method only handles self.annotation_object removal
    #
    def remove_annotation_object(self,
                                 key):
        annotation_plot = self.anno_objects[key][PLOT_IND]
        annotation_plot.remove()

        self.anno_objects.pop(key)

    # method: edit_annotation_object
    #
    # arguments:
    #  -key: key of annotation to be editted
    #
    # returns: none
    #
    # this method edits a single plotted annotation
    #
    def edit_annotation_object(self,
                               key):
        annotation_plot = self.anno_objects[key][PLOT_IND]
        annotation_plot.set_style(is_selected=True)

    # method: get_keys_of_annotation_objects_in_rect
    #
    # arguments:
    #  -x_position_start_a: left bound of rectangle in seconds
    #  -x_position_end_a: right bound of rectangle in seconds
    #  -channel_low_a: channel_number of upper bound of rectangle
    #  -channel_high_a: channel_number of lower bound of rectangle
    #
    # returns: keys of annotations inside the rectangle
    #
    # this method finds and returns the annotations found inside the rectangle
    # drawn by the user
    #
    def get_keys_of_annotation_objects_in_rect(self,
                                               x_position_start_a,
                                               x_position_end_a,
                                               channel_low_a,
                                               channel_high_a):
        selected_channels = range(channel_low_a, channel_high_a)

        keys_for_annotations_within_rectangle = []

        for key in self.anno_objects:
            annotation_object = self.anno_objects[key]
            channel_number = annotation_object[CHAN_IND]

            if channel_number in selected_channels:

                anno_left_bound = annotation_object[L_BOUND_IND]
                anno_right_bound = annotation_object[R_BOUND_IND]

                if (anno_left_bound <= x_position_end_a
                    and anno_right_bound >= x_position_start_a):

                    keys_for_annotations_within_rectangle.append(key)

        return keys_for_annotations_within_rectangle

    # method: process_selected_annotation_objects
    #
    # arguments:
    #  -keys_for_selected_annotation_objects: keys of selected annotations
    #
    # returns: none
    #
    # this method sets the annotation objects to selected
    #
    def process_selected_annotation_objects(self,
                                            keys_for_selected_annotation_objects):
        for key in keys_for_selected_annotation_objects:
            annotation_object = self.anno_objects[key][PLOT_IND]
            annotation_object.set_style(is_selected=True)

    # method: deal_with_type_selection_annotation_object
    #
    # arguments:
    #  -key_a: key of annotation_object to editted
    #  -type_number_a: annotation type to be changed to
    #
    # returns: none
    #
    # this method edits the annotation object to a specific type
    #
    def deal_with_type_selection_annotation_object(self,
                                                   key_a,
                                                   type_number_a):

        annotation_plot = self.anno_objects[key_a][PLOT_IND]

        # get a string with which to label annotation_plot
        #
        annotation_string = self.dict_event_map[type_number_a]
        annotation_plot.set_type(annotation_string)

        self.anno_objects[key_a][TYPE_IND] = type_number_a

    # method: deselect_annotation_object
    #
    # arguments:
    #  -key: key of annotation_object to be deselected
    #
    # returns: none
    #
    # this method deselects a single annotation object
    #
    def deselect_annotation_object(self,
                                   key):
        annotation_plot = self.anno_objects[key][PLOT_IND]
        annotation_plot.set_style(is_selected=False)

        key_for_color = annotation_plot.anno_name
        annotation_plot.set_type(key_for_color)

    # method: process_channel_selector_annotation_object
    #
    # arguments:
    #  -channel_dict: dictionary of selected channels
    #  -key: key of annotation object to be displayed
    #
    # returns: none
    #
    # this method removes annotation objects from the screen if
    # the channel is not selected in the channel selector menu
    #
    def process_channel_selector_annotation_object(self,
                                                   channel_dict,
                                                   key):
        # get object and associated channel_number
        #
        annotation_object = self.anno_objects[key]
        channel_number = annotation_object[CHAN_IND]

        # If a channel / key combination is found where the channel
        # dictionary returns a corresponding False value, then we must
        # remove the old object, and delete records of it
        #
        if channel_dict.get(channel_number) is False:
            self.remove_annotation_object(key)

    # method: translate_annotation_object
    #
    # arguments:
    #  -new_left_bound: left bound to move to
    #  -new_right_bound: right bound to move to
    #  -key_a: key of annotation object to be moved
    #  -annotation_overlapping_a: boolean, true if annotations are overlapping
    #
    # returns: none
    #
    # this method moves an annotation object to a desired left and right bound
    #
    def translate_annotation_object(self,
                                    new_left_bound,
                                    new_right_bound,
                                    key_a,
                                    annotation_overlapping_a):

        moving_annotation_object = self.anno_objects[key_a]

        # if there are no overlapping annotations, translate annotation object
        #
        if annotation_overlapping_a is False:

            moving_annotation_object[L_BOUND_IND] = new_left_bound
            moving_annotation_object[R_BOUND_IND] = new_right_bound

        # if there are overlapping annotations, remove previous annotation
        # object and create a new annotation object at new bounds
        #
        else:

            channel_number = moving_annotation_object[CHAN_IND]
            type_number = moving_annotation_object[TYPE_IND]

            self.remove_annotation_object(key_a)
            self.plot_anno(channel_number,
                           new_left_bound,
                           new_right_bound,
                           type_number=type_number,
                           unique_key=key_a)
