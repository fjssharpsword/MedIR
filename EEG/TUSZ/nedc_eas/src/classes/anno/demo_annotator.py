#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/anno/demo_annotator.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from distutils.command.config import config
import os
import re

from pyqtgraph.Qt import QtCore

from .demo_annotation_selection_menu import DemoAnnotationSelectionMenu
from .demo_annotation_util import DemoAnnotationUtil
from .demo_annotation_plotter import DemoAnnotationPlotter

# these constants are defined elsewhere to:
# 1) document the data structure
# 2) to avoid hard-coding
# 3) allow for sharing these constants between modules
#
from .demo_annotation_data_representation import *

# regex expression for map file
#
MAP_REGEX = re.compile("(.+?(?==))=\((\d+),(\d+),(\(\d+,\d+,\d+,\d+\))\)", 
    re.IGNORECASE)


# method: _make_map_dictionary
#
# arguments:
#  -map_file: the map file
#
# returns: parsed dictionaries from preferences file.
#
# this method parses the argument dictionary into smaller,
# more specific dictionaries.
#
def _make_map_dictionary(map_file):
    
    map_dictionary  = {}
    
    with open(map_file) as file:

        for ind, line in enumerate(file):

            line = line.strip().replace(" ", "")

            if len(line) == 0 or line.startswith("#") or line.startswith("[") or line.startswith("symbols"):
                continue
            
            # pattern matching
            #
            result = re.findall(MAP_REGEX, line)[0]

            if len(result) < 4:
                raise Exception(f"Map File Configuration invalid on line {ind + 1}")

            key, mapping, priority, rgb_val = result[0], int(result[1]), int(result[2]), eval(result[3])

            map_dictionary[key] = (mapping, priority, rgb_val)
    
    return map_dictionary

# class: DemoAnnotator
#
# This class acts as a the main class for annotations. It stores the
# annotations that are plotted, and calls methods to generate
# annotations and process the mouse interactions
#
# Example annotation:
# --Setting:
# key = self.generate_unique_key() <- 9999
# channel = 1
# start_time = 77.7
# end_time = 88.8
# anno_type = 11
# self.annotations[key] = [channel, start_time, end_time, anno_type]
# -- Reading:
# channel = self.annotations[9999][0]       <- 1
# start_time = self.annotation[9999][1]     <- 77.7
# end_time = self.annotation[9999][2]       <- 88.8
# anno_type = self.annotation[9999][3]      <- 11
#
class DemoAnnotator:

    # method: __init__
    #
    # arguments:
    #  -sigplots_widget_a: signal plotting widget that allows for connections
    #  -cfg_dict_*: dictionary of values from config file
    #  -time_scale_a: time scale of demo
    #  -montage_names_a: names of montage from config file
    #
    def __init__(self,
                 sigplots_widget_a,
                 ann_map_file,
                 cfg_dict_single_annotation_a,
                 time_scale_a,
                 montage_names_a,
                 main_window_widget_a,
                 ann_chan_map_file_a,
                 xml_schema):

        # we need to be able to address the signal plot widget from
        # within this class, as this is where all the annotations are
        # selected and plotted.  This is defined in
        # $(NEDC_NFC)/src/classes/sigplots/demo_sigplots_widget.py
        # This is called from
        # $(NEDC_NFC)/src/classes/demo_event_loop.py
        #
        self.sigplots_widget = sigplots_widget_a

        self.main_window_widget = main_window_widget_a

        self.time_scale = int(time_scale_a)

        # used to control individual annotations
        # (passed at __init__ of DemoAnnotation)
        #
        self.cfg_dict_single_anno = cfg_dict_single_annotation_a
        self.ann_map_file = _make_map_dictionary(ann_map_file)

        self.dict_anno_colors,     \
        self.dict_event_map,       \
        self.dict_labels_priority, \
        self.dict_priority_map = self._parse_configuration(self.ann_map_file)

        # initialize the pop-up menu for selection of annotation type_number
        #
        self.annotation_selection_menu = DemoAnnotationSelectionMenu(self,
            self.ann_map_file)

        self.set_montage_for_channel_selector(montage_names_a)

        # connect the various signals that enable region selection and
        # hooks to process "OK" and "Remove" in self.annotation_selection_menu
        #
        self._init_connect_signals()

        # initialize some empty dictionaries
        #
        self.annotations = {}
        self.pre_computed_views = {}

        self.keys_for_selected_annotation_objects = []

        # this is used to assign unique ids to each annotation
        #
        self.annotation_count = 0

        # this is used to ignore certain anno types
        #
        self.ignore_annotations = []

        # this is used to store the slider position, received from event loop
        # in plot_annotations_for_current_time_window
        #
        self.previous_slider_pos = 0

        self.anno_util = DemoAnnotationUtil(self.dict_labels_priority,
                                            self.dict_event_map,
                                            self.dict_priority_map,
                                            ann_map_file,
                                            xml_schema)

        self.anno_plot = DemoAnnotationPlotter(self.sigplots_widget,
                                               self.dict_event_map,
                                               self.cfg_dict_single_anno,
                                               self.dict_anno_colors,
                                               montage_names_a,
                                               self.ann_map_file,
                                               time_scale_a)


    # method: _parse_configuration
    #
    # arguments:
    #  -cfg_dict_map_a: dictionary of values for each type_number of annotation
    #
    # returns: parsed dictionaries from preferences file.
    #
    # this method parses the argument dictionary into smaller,
    # more specific dictionaries.
    #
    def _parse_configuration(self,
                             cfg_dict_map_a):

        # initialize a bunch of empty dictionaries to store
        # mappings extracted from cfg_dict_map_a
        #
        # cfg_dict_map_a contains entries of this form:
        #
        #   KEY   | MAPPING     PRIORITY    RGB_COLOR_SCHEME
        # --------+-------------------------------------------------------
        #   SPSW: |  ( 1,          0,       (128, 128, 255,  35))
        #
        # we make the following dictionaries from this passed in dictionary:

        # maps annotation id strings to rgb values.
        # This class makes significant use of this.
        # LabelConverterUtils.__init__() doesn't care.
        # Probably should use mapping number instead of
        # string as key, but havent done this yet.
        #   (e.g. SPWS -> (128, 128, 255, 35))
        #
        dict_anno_colors = {}

        # maps annotation id numbers to label strings.
        # This class makes significant use of this.
        # LabelConverterUtils.__init__() requires it.
        #   (e.g. 1 -> SPSW)
        #
        dict_event_map = {}

        # decides which annotation to choose if overlap
        # not used here except for LabelConverterUtils.
        # LabelConverterUtils.__init__() requires it.
        #   (e.g. SPSW -> [1, 0])
        #
        dict_labels_priority = {}

        # maps annotation id numbers to priorities
        # not used here except for LabelConverterUtils.
        # LabelConverterUtils.__init__() requires it.
        #   (e.g. 1 -> 0)
        #
        dict_priority_map = {}

        for key in cfg_dict_map_a:

            # break out one entry of the config dictionary into its components
            #
            annotation_name = key
            label_id = int(cfg_dict_map_a[key][0])
            priority = int(cfg_dict_map_a[key][1])
            color = cfg_dict_map_a[key][2]

            dict_labels_priority[key] = [label_id, priority]
            dict_priority_map[label_id] = priority
            dict_event_map[label_id] = annotation_name

            # create annotation color dictionary
            #
            dict_anno_colors[key] = color

        return (dict_anno_colors,
                dict_event_map,
                dict_labels_priority,
                dict_priority_map)

    # method: write_annotation_to_file
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called when the user selects save from the menu bar
    #
    def write_annotations_to_file(self, ext):
        self.anno_util.write_annotations_to_file(self.annotations, ext)

    # method: save_as_annotations_to_file
    #
    # arguments:
    #
    # returns: none
    #
    # this method is called when user selects save as from the menu bar
    #
    def save_as_annotations_to_file(self, ext):
        self.anno_util.save_as_annotations_to_file(self.annotations, ext)

    # method: _init_connect_signals
    #
    # arguments: none
    #
    # returns: none
    #
    # this method acts as a constructor to the selection menu signals,
    # connects them various methods in this class
    #
    def _init_connect_signals(self):

        # when button "OK" is selected from self.annotation_selected_menu,
        # call `self.process_type_selection`
        #
        self.annotation_selection_menu. \
            signal_return_ok.connect(self.process_type_selection)

        # when button "Remove" is selected from self.annotation_selected_menu,
        # call `self.process_menu_remove`
        #
        self.annotation_selection_menu. \
            signal_return_remove.connect(self.process_menu_remove)

        # when button "Adjust" is selected from self.annotation_selection_menu,
        # call `adjust_annotations`
        #
        self.annotation_selection_menu. \
            signal_adjust_pressed.connect(self.adjust_annotations)

        # when the "x" is selected, do nothing but unselect annotations
        #
        self.annotation_selection_menu. \
            signal_return_no_ok.connect(self.clear_selected_annotations)

        # when button "OK" is selected from the channel_selector,
        # call self.process_channel_selector_selection
        #
        self.annotation_selection_menu.channel_selector. \
            signal_return_ok.connect(self.process_channel_selector_selection)

        # call self.process_channel_selector_selection whenever a choice in
        # channel selector has changed
        #
        self.annotation_selection_menu.channel_selector. \
            signal_update_channels.connect(
                self.process_channel_selector_selection)

        # call self.process_type_selector_selection whenever a checkbox in
        # anno type selector has changed
        #
        self.main_window_widget.anno_type_selector. \
            signal_update_selections.connect(
                self.process_type_selector_selection)

        # when (rectangular) region is selected in
        # `self.sigplots_widget.page_only_waveform.signal_plot.plotItem.vb`
        # a signal `signal_region_selected` is emitted. When this is
        # emitted, call `self.process_region_selection`
        #
        self.sigplots_widget.page_only_waveform.signal_plot.plotItem.vb. \
            signal_region_selected.connect(self.process_region_selection)

    # method: set_total_time_recording
    #
    # arguments:
    #  -total_time_recording_a: total time of EEG record
    #
    # returns: none
    #
    # this method sets attribute self.total_time_recording when called
    #
    def set_total_time_recording(self,
                                 total_time_recording_a):
        self.total_time_recording = int(total_time_recording_a)
        self.pre_computed_views, self.sorted_annos = self.pre_compute_view()

    def set_time_scale(self,
                       time_scale_a):
        self.time_scale = int(time_scale_a)
        self.pre_computed_views, self.sorted_annos = self.pre_compute_view()

    def set_montage_for_channel_selector(self,
                                         montage_names_a):

        self.annotation_selection_menu.channel_selector.set_montage(
            montage_names_a)

    # method: setup_on_opening_new_edf
    #
    # arguments:
    #  -file_a: string containing edf file path
    #
    # returns: none
    #
    # this method sets path to edf file, also checks if a rec
    # file exists for the current edf file
    #
    def setup_on_opening_new_edf(self,
                                 file_a):

        self.clear_all_annotations()
        if (os.path.isfile(file_a)):

            annotations = self.anno_util.read_file(file_a)

            # iterate over read in annotations
            #
            for annotation in annotations:

                channel_number = annotation[CHAN_IND]
                x_pos_start    = annotation[L_BOUND_IND]
                x_pos_end      = annotation[R_BOUND_IND]
                type_number    = annotation[TYPE_IND]

                self.make_annotation(channel_number,
                                     x_pos_start,
                                     x_pos_end,
                                     type_number)

    # method: clear_all_annotations
    #
    # arguments: none
    #
    # returns: none
    #
    # this method removes all plotted and stored annotations and
    # resets the annotation counter to 0
    #
    def clear_all_annotations(self):
        self.annotations = {}
        self.anno_plot.anno_objects = {}
        self.annotation_count = 0

    # method: plot_annotations_for_current_time_window
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called when eeg_plotter is called, plots annotations that will
    # occur in the current time window
    #
    def plot_annotations_for_current_time_window(self,
                                                 slider_current_pos_a):
    
        self.anno_plot.plot_for_current_time_window(slider_current_pos_a,
                                                    self.annotations,
                                                    self.pre_computed_views,
                                                    self.ignore_annotations)

        # connect all plotted annotations to signals
        #
        for key in self.anno_plot.anno_objects:
            self.connect_annotation_to_signals(key)

        # store value for plotting annos within this class, not from event loop
        #
        self.previous_slider_pos = slider_current_pos_a

    # method: connect_annotation_to_signal
    #
    # arguments:
    #  -key: key of self.anno_plot.annotation_plot to be connected
    #
    # returns: none
    #
    # this method connects a plotted annotation to the signals below
    #
    def connect_annotation_to_signals(self,
                                      key):

        annotation_plot = self.anno_plot.anno_objects[key][PLOT_IND]

        # calls self.deal_with_annotation_movement when user clicks
        # and drags annotation_plot
        #
        annotation_plot.signal_region_changed.connect(
            self.deal_with_annotation_movement)

        # calls remove_single_annotation when user clicks on an annotation_plot
        # and selects 'remove'
        #
        annotation_plot.signal_remove_unique_key.connect(
            self.remove_single_annotation)

        # calls edit_single_annotation when user clicks on an annotation_plot
        # and selects 'edit'
        #
        annotation_plot.signal_edit_unique_key.connect(
            self.edit_single_annotation)

    # method: process_type_selection
    #
    # arguments:
    #  -annotation_type_number_a: type_number of annotation to be created
    #
    # returns: none
    #
    # this method is called when the "Ok" button in
    # `self.annotation_selection_menu` is selected. It is responsible
    # for labeling the event and recording it within self.annotations
    #
    def process_type_selection(self,
                               annotation_type_number_a):

        # iterate over channels in selected annotations, and then
        # iterate over annotations within those channels
        #
        for key in self.keys_for_selected_annotation_objects:

            self.deal_with_type_selection_one_annotation(
                key,
                annotation_type_number_a)

        self.deselect_annotations()

        self.enable_all_user_interaction()

    # method: deal_with_type_selection_one_annotation
    #
    # arguments:
    #  -key_a: key of annotation to be editted
    #  -type_number_a: type_number of annotation to be changed todo
    #
    # returns: none
    #
    # this method calls method in DemoAnnotationPlotter and updates
    # self.annotations with new type_number_a
    #
    def deal_with_type_selection_one_annotation(self,
                                                key_a,
                                                type_number_a):

        self.anno_plot.deal_with_type_selection_annotation_object(
            key_a,
            type_number_a)
        self.annotations[key_a][TYPE_IND] = type_number_a

    # method: process_menu_remove
    #
    # arguments: none
    #
    # returns: none
    #
    # this method removes selected annotations when user selects 'remove'
    # from selection menu
    #
    def process_menu_remove(self):

        for key in self.keys_for_selected_annotation_objects:

            self.anno_plot.remove_annotation_object(key)
            self.annotations.pop(key)

        self.keys_for_selected_annotation_objects = []

        self.pre_computed_views, self.sorted_annos = self.pre_compute_view()

    # method: process_region_selection
    #
    # arguments:
    #  -x_pos_start: left bound in seconds
    #  -y_position_top: lower bound in pixels
    #  -x_pos_end: right bound in seconds
    #  -y_position_bottom: upper bound in pixels
    #
    # returns: None
    #
    # this method is called whenever a (rectanglular) region is
    # selected via the mouse.
    #
    # If there are already annotations in this rectangle:
    #   it selects these annotations and collects their keys
    #   in `self.keys_for_selected_annotation_objects'
    #
    # Otherwise,
    #   it maps the y position arguments to a range of channels,
    #   draws annotations on these channels and collects their keys
    #   in `self.keys_for_selected_annotation_objects'
    #
    # either way, it finishes up by prompting the user to select a
    # label type_number for the annotations whose keys are selected in
    # `self.keys_for_selected_annotation_objects'.
    #
    # the button that opens the channel selector widget is only shown
    # if the annotations are freshly drawn (there is not an obvious
    # behavior for the channel selector widget if the selected
    # annotations have diverse start and start times)
    #
    def process_region_selection(self,
                                 x_pos_start,
                                 y_position_top,
                                 x_pos_end,
                                 y_position_bottom):

        self.deselect_annotations()
        channel_bottom, channel_top = \
            self.sigplots_widget.page_only_waveform. \
            get_rectangle_in_channel_coordinates(y_position_bottom,
                                                 y_position_top)

        # call method in DemoAnnotationPlotter to get the keys of annotations
        # within drawn rectangle, if any
        #
        self.keys_for_selected_annotation_objects = \
            self.anno_plot.get_keys_of_annotation_objects_in_rect(
                x_pos_start,
                x_pos_end,
                channel_top,
                channel_bottom)

        # if there are annotations in the rectangle drawn by user's mouse,
        # then collect them in self.selected_annotations.
        #
        if self.keys_for_selected_annotation_objects:
            self.annotation_selection_menu.select_channels_button.hide()
            self.annotation_selection_menu.adjust_button.show()

            self.anno_plot.process_selected_annotation_objects(
                self.keys_for_selected_annotation_objects)

        # if no annotations are in the rectangle drawn by user's mouse,
        # then make some new annotations.
        #
        else:
            self.annotation_selection_menu.adjust_button.hide()
            self.annotation_selection_menu.select_channels_button.show()
            
            self.add_new_rectangle_of_annotations(
                x_pos_start,
                x_pos_end,
                channel_top,
                channel_bottom)

            self.annotation_selection_menu.channel_selector. \
                set_channels_selected(channel_top,
                                      channel_bottom)
        

        self.show_annotation_selection_menu()

    # method: process_channel_selector_selection
    #
    # arguments:
    #  -channel_dict: dictionary of selected channels in channel_selector
    #
    # returns: none
    #
    # this method calls plotting methods to plot annotations on
    # selected channels
    #
    def process_channel_selector_selection(self,
                                           channel_dict):

        # iterate over old keys for selected annotation objects. The
        # strategy here is to compare the channels associated with
        # these keys to the True / False checkbox values in the the
        # checkbox channel dictionary arg. The [-1::-1] thing: we
        # must iterate "backwards" over this list because we remove
        # from the list during iteration, and this will mess up the
        # iteration unless we go backwards
        #
        for key in self.keys_for_selected_annotation_objects[-1::-1]:

            # get object and associated channel_number
            #
            annotation_plot = self.anno_plot.anno_objects[key]
            channel_number = annotation_plot[CHAN_IND]

            # If a channel / key combination is found where the channel
            # dictionary returns a corresponding False value, then we must
            # remove the old object, and delete records of it
            #
            if channel_dict.get(channel_number) is False:

                # remove the object from GUI
                #
                self.anno_plot.remove_annotation_object(key)

                # remove all records of the object
                #
                self.annotations.pop(key)
                self.keys_for_selected_annotation_objects.remove(key)

            # we have now processed the channel
            #
            channel_dict.pop(channel_number)

        example_key = self.keys_for_selected_annotation_objects[0]
        example_anno_plot = self.anno_plot.anno_objects[example_key][PLOT_IND]
        left_bound = example_anno_plot.left_bound
        right_bound = example_anno_plot.right_bound

        for channel_number in channel_dict:

            channel_is_selected =  channel_dict[channel_number]

            if channel_is_selected:
                self.make_annotation(channel_number,
                                     left_bound,
                                     right_bound,
                                     do_plot=True,
                                     is_selected=True)

        self.pre_computed_views, self.sorted_annos = self.pre_compute_view()

    # method: process_type_selector_selection
    #
    # arguments:
    #  -type_dict: dictionary of selected annotation types in anno_type_selector
    #
    # returns: none
    #
    # this method checks to see if an annotation type should be plotted, also
    # checks types to see if we should navigate to the annotation
    #
    def process_type_selector_selection(self,
                                        type_dict):

        # iterate over all checkboxes
        #
        for key in type_dict:

            # get the mapped type number of key, e.g. 'null' -> 0
            #
            mapped_key = self.ann_map_file[key][0]
            
            # if key is in ignore annotations
            #
            if mapped_key in self.ignore_annotations:

                # if checkbox is clicked
                #
                if type_dict[key] is True:
                    self.ignore_annotations.remove(mapped_key)

            # if key is not in ignore annotations
            #
            else:

                # if checkbox is not clicked
                #
                if type_dict[key] is False:
                    self.ignore_annotations.append(mapped_key)

        self.pre_computed_views, self.sorted_annos = self.pre_compute_view()

        self.plot_annotations_for_current_time_window(self.previous_slider_pos)

    # method: clear_selected_annotations
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called when the 'x' is selected in selection menu,
    # deselects all annotations
    #
    def clear_selected_annotations(self):

        for key in self.keys_for_selected_annotation_objects:
            annotation_type = self.annotations[key][TYPE_IND]

            # if the annotation is not null, deselect
            #
            if annotation_type != 0:
                self.anno_plot.deselect_annotation_object(key)

            # if annotation is null, delete annotation
            #
            else:
                self.anno_plot.remove_annotation_object(key)
                self.annotations.pop(key)

            self.keys_for_selected_annotation_objects = []
            self.pre_computed_views, self.sorted_annos = self.pre_compute_view()
        self.enable_all_user_interaction()

    # method: add_new_rectangle_of_annotations
    #
    # arguments:
    #  -x_pos_start: left bound of rectangle in seconds
    #  -x_pos_end: right bound of rectangle in seconds
    #  -channel_start: top channel number covered by rectangle
    #  -channel_end: bottom channel number covered by rectangle
    #
    # returns: none
    #
    # this method is called when a user draws a rectangle with no annotations
    # inside. generates a rectangle of annotations
    #
    def add_new_rectangle_of_annotations(self,
                                         x_pos_start,
                                         x_pos_end,
                                         channel_start,
                                         channel_end):

        for channel_number in range(channel_start,
                                    channel_end):

            self.make_annotation(channel_number,
                                 x_pos_start,
                                 x_pos_end,
                                 do_plot=True,
                                 is_selected=True)
        self.pre_computed_views, self.sorted_annos = self.pre_compute_view()

    # method: make_annotation
    #
    # arguments:
    #  -channel_number: which channel the annotation is on
    #  -left_bound:     beginning (in seconds)
    #  -right_bound:    end (in seconds)
    #  -type_number:           a number that is mapped to annotation type_number
    #  -do_plot:        should this annotation be plotted upon creation.
    #                   if not, just store a record of it
    #  -is_selected:    should the annotation be selected
    #                   (visually as well as should we remember it as selected
    #                   via self.keys_for_selected_annotation_objects)
    #
    # returns: none. It might be useful to make this return the unique
    #                key generated, but this is unnecessary at present
    #
    # this method is the only method for creating new annotation
    # records. all other methods that need to create new annotation
    # records should call this.
    #
    def make_annotation(self,
                        channel_number,
                        left_bound,
                        right_bound,
                        type_number=0,
                        do_plot=False,
                        is_selected=False):

        # create a unique by which to remember the annotation
        # not very sophisticated, but it works.
        #
        unique_key = self.annotation_count
        self.annotation_count += 1

        # create the annotation
        #
        self.annotations[unique_key] = [channel_number,
                                        left_bound,
                                        right_bound,
                                        type_number]

        if do_plot is True:
            self.anno_plot.plot_anno(channel_number,
                                     left_bound,
                                     right_bound,
                                     type_number=type_number,
                                     is_selected=is_selected,
                                     unique_key=unique_key)
            self.connect_annotation_to_signals(unique_key)

            if is_selected is True:
                self.keys_for_selected_annotation_objects.append(unique_key)

    # method: deselect_annotations
    #
    # arguments: none
    #
    # returns: none
    #
    # this method deselects all annotations, called when editting annotations
    #
    def deselect_annotations(self):
        for key in self.keys_for_selected_annotation_objects:
            self.anno_plot.deselect_annotation_object(key)
        self.keys_for_selected_annotation_objects = []

    # method: deal_with_annotation_movement
    #
    # arguments:
    #  -new_left_bound: left bound where user dropped the annotation in seconds
    #  -new_right_bound: right bound where user dropped the annotation in seconds
    #  -key_a: key of moving annotation
    #
    # returns: none
    #
    # this method is called when a user stops dragging the annotation, updates
    # self.annotations and self.anno_plot.anno_objects values
    #
    def deal_with_annotation_movement(self,
                                      new_left_bound,
                                      new_right_bound,
                                      moving_key):

        # this bool will become true if the annotation that moved
        # is overlapping another annotation
        #
        annotation_overlapping = False

        # store original length of annotation
        #
        annotation_length = new_right_bound - new_left_bound

        # channel number of moving annotation
        #
        moving_annotation = self.annotations[moving_key]
        moving_channel_number = moving_annotation[CHAN_IND]

        # store bounds before moving the annotation
        #
        prev_l_bound = moving_annotation[L_BOUND_IND]
        prev_r_bound = moving_annotation[R_BOUND_IND]

        for key in self.annotations:

            # make sure we don't check the moving annotation against itself
            #
            if key != moving_key:

                # if the moving annotation is able to overlap annotation[key] and if
                # the annotation is not null
                #
                if (self.annotations[key][CHAN_IND] == moving_channel_number and
                    self.annotations[key][TYPE_IND] != 0):

                    # get bounds of potential overlapping annotation
                    #
                    existing_l_bound = self.annotations[key][L_BOUND_IND]
                    existing_r_bound = self.annotations[key][R_BOUND_IND]

                    # case where new_left_bound is overlapping annotation
                    #
                    if existing_l_bound < new_left_bound < existing_r_bound:

                        # new_left_bound pops to right end of overlapping annotation
                        #
                        new_left_bound = existing_r_bound

                        # retain original length of annotation
                        #
                        new_right_bound = new_left_bound + annotation_length

                        annotation_overlapping = True

                    # case where new_right_bound is overlapping annotation
                    #
                    if existing_l_bound < new_right_bound < existing_r_bound:

                        # new_right_bound pops to left end of overlapping annotation
                        #
                        new_right_bound = existing_l_bound

                        # retain original length of annotation
                        #
                        new_left_bound = new_right_bound - annotation_length

                        annotation_overlapping = True

                    # case where entire moving_annotation is overlapping another
                    #
                    if (new_left_bound < existing_l_bound and
                        new_right_bound > existing_r_bound):

                        # new_right_bound pops to left end of overlapping annotation
                        #
                        new_right_bound = existing_l_bound

                        # retain original length of annotation
                        #
                        new_left_bound = new_right_bound - annotation_length

                        annotation_overlapping = True
        #
        # end of for loop

        # translate annotation to new bounds
        #
        moving_annotation[L_BOUND_IND] = new_left_bound
        moving_annotation[R_BOUND_IND] = new_right_bound

        self.anno_plot.translate_annotation_object(new_left_bound,
                                                   new_right_bound,
                                                   moving_key,
                                                   annotation_overlapping)

        # if we moved an annotation that is selected
        #
        if moving_key in self.keys_for_selected_annotation_objects:

            # remove this key from selected keys
            #
            self.keys_for_selected_annotation_objects.remove(moving_key)

            # iterate over all other selected annotations
            #
            for key in self.keys_for_selected_annotation_objects:

                # reference for selected annotation
                #
                anno = self.annotations[key]

                # if we resized with the right handle
                #
                if new_left_bound == prev_l_bound:

                    # get the distance the annotation has been stretched
                    #
                    distance = new_right_bound - prev_r_bound
                    stretched_r_bound = anno[R_BOUND_IND] + distance

                    # stretch selected annotation by the same length
                    #
                    anno[R_BOUND_IND] = stretched_r_bound
                    self.anno_plot.translate_annotation_object \
                        (anno[L_BOUND_IND],
                         stretched_r_bound,
                         key,
                         annotation_overlapping)

                # if we resized with the left handle
                #
                elif new_right_bound == prev_r_bound:

                    # get the distance annotation has been stretched
                    #
                    distance = new_left_bound - prev_l_bound
                    stretched_l_bound = anno[L_BOUND_IND] + distance

                    # stretch selected annotation by the same length
                    #
                    anno[L_BOUND_IND] = stretched_l_bound
                    self.anno_plot.translate_annotation_object \
                        (stretched_l_bound,
                         anno[R_BOUND_IND],
                         key,
                         annotation_overlapping)

                # if we moved the entire annotation
                #
                else:

                    # get distance each bound moved
                    #
                    distance = new_left_bound - prev_l_bound
                    stretched_l_bound = anno[L_BOUND_IND] + distance

                    distance = new_right_bound - prev_r_bound
                    stretched_r_bound = anno[R_BOUND_IND] + distance

                    # update both bounds
                    #
                    anno[L_BOUND_IND] = stretched_l_bound
                    anno[R_BOUND_IND] = stretched_r_bound
                    self.anno_plot.translate_annotation_object(stretched_l_bound,
                                                               stretched_r_bound,
                                                               key,
                                                               annotation_overlapping)

            # clear annotations after we're done moving
            #
            self.clear_selected_annotations()

        # TODO: is this connect_annotation_to_signals necessary?
        #
        #self.connect_annotation_to_signals()

        self.pre_computed_views, self.sorted_annos = self.pre_compute_view()

        self.plot_annotations_for_current_time_window(self.previous_slider_pos)

    # method: remove_single_annotation
    #
    # arguments:
    #  -unique_key: key of annotation to be removed
    #
    # returns: none
    #
    # this method is called when the user clicks on an annotation, selects 'remove'
    #
    def remove_single_annotation(self,
                                 unique_key):
        self.anno_plot.remove_annotation_object(unique_key)
        self.annotations.pop(unique_key)

        self.keys_for_selected_annotation_objects = []
        self.pre_computed_views, self.sorted_annos = self.pre_compute_view()

    # method: edit_single_annotation
    #
    # arguments:
    #  -unique_key: key of annotation to edit
    #
    # returns: none
    #
    # this method is called when the suer clicks on an annotation, selects 'edit'
    #
    def edit_single_annotation(self,
                               unique_key):
        self.anno_plot.edit_annotation_object(unique_key)
        self.keys_for_selected_annotation_objects = [unique_key]
        self.annotation_selection_menu.select_channels_button.hide()
        self.annotation_selection_menu.adjust_button.show()
        self.show_annotation_selection_menu()

    # method: show_annotation_selection_menu
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called when a user wants to edit annotations,
    # disables user interaction with the demo until selection menu is closed
    #
    def show_annotation_selection_menu(self):
        self.disable_all_user_interaction_other_than_through_anno_select_menu()
        self.annotation_selection_menu.show()

    # method: disable_all_user_interaction_other_than_through_anno_select_menu
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called when selection menu is open, this prevents any
    # unwanted interaction with the demo while selected annotations
    #
    def disable_all_user_interaction_other_than_through_anno_select_menu(self):
        self.sigplots_widget.page_only_waveform.signal_plot.plotItem.vb. \
            signal_region_selected.disconnect()
        for annotation in self.anno_plot.anno_objects.values():
            annotation_plot = annotation[PLOT_IND]
            annotation_plot.page_1_roi.translatable = False
            annotation_plot.signal_remove_unique_key.disconnect()
            annotation_plot.signal_edit_unique_key.disconnect()

    # method: enable_all_user_interaction
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called once selection menu is closed, restores
    # all functionality to the demo
    #
    def enable_all_user_interaction(self):

        # when (rectangular) region is selected in
        # `self.sigplots_widget.page_only_waveform.signal_plot.plotItem.vb`
        # a signal `signal_region_selected` is emitted. When this is
        # emitted, call `self.process_region_selection`
        #
        self.sigplots_widget.page_only_waveform.signal_plot.plotItem.vb. \
            signal_region_selected.connect(self.process_region_selection)
        for annotation in self.anno_plot.anno_objects.values():
            annotation_plot = annotation[PLOT_IND]
            annotation_plot.page_1_roi.translatable = True
            annotation_plot.signal_remove_unique_key.connect(
                self.remove_single_annotation)
            annotation_plot.signal_edit_unique_key.connect(
                self.edit_single_annotation)

    # method: pre_compute_view
    #
    # arguments: None
    #
    # returns:
    #  - pre_compute_views:  indexed sets of views for speeding up navigation
    #  - sorted_annos: ordered list of annotations for nav by anno
    #
    # this method creates an set of indexes for quick resolution of
    # annotations. Otherwise navigation gets very slow for edfs with
    # large numbers of annotations
    #
    def pre_compute_view(self):

        pre_computed_views = {}
        sorted_annos = []

        dt = self.time_scale

        # create a sequence of integers corresponding to all seconds in edf
        # make this sequence into a dict of the form:
        # {0: [], 1: [], 2: [], ... N: []}
        #
        keys = range(0, self.total_time_recording + 1)
        pre_computed_views = dict.fromkeys(keys)
        for view in pre_computed_views:
            pre_computed_views[view]=[]
        # iterate over all annotations and associate them with each view
        #
        for key in self.annotations:
            
            # establish the beginning and end of annotation
            #
            x_pos_start = int(self.annotations[key][L_BOUND_IND])
            x_pos_end = int(self.annotations[key][R_BOUND_IND])
    
            # only add to sorted annos if the type is not in ignore annotations
            # this is changed via anno type selector
            #
            if self.annotations[key][TYPE_IND] not in self.ignore_annotations:
                
                # accumulate annotations for later sorting by x_pos_start time
                # this is _only_ for get_*_annotation
                #
                sorted_annos.append([key, x_pos_start])

            # find the first and last view that this particular
            # annotation should be associated with
            #
            first_view = max(0, x_pos_start - 1 - self.time_scale)
            last_view = x_pos_end + self.time_scale

            # append this annotation to each view in appropriate range
            #
            for view in range(first_view, last_view):
         
                # some tests to determine if the annotation belongs in
                # current view in iteration
                #
                anno_starts_in_view = (x_pos_start >= view
                                       and x_pos_start <= view + dt)
                anno_ends_in_view = (x_pos_end >= view
                                     and x_pos_end <= view + dt)
        
                # if the annotation passes the test, append it to the list for
                # that view (stored in the dict pre_computed_views)
                #
                if (anno_starts_in_view or anno_ends_in_view):
                    try:
                        pre_computed_views[view].append(key)
                    except KeyError:
                        pass
                    
            # occasionally there are annotations that are longer than self.time_scale.
            # this makes sure that this annotation is associated with these views
            #     <--- annotation --->
            #        <--- view --->
            # we need to make sure that this situation is accounted for
            #
            anno_size_larger_than_time_scale = \
                x_pos_end - x_pos_start > dt
            if anno_size_larger_than_time_scale:
                for view in pre_computed_views:
                    if (x_pos_start < view
                        and x_pos_end > view + dt):
                        if key not in pre_computed_views[view]:
                            pre_computed_views[view].append(key)

        # sort annotations. This allows for quick get_*_annotation
        #
        sorted_annos.sort(key=lambda tup:tup[1])

        return (pre_computed_views,
                sorted_annos)

    def adjust_annotations(self):
        self.enable_all_user_interaction()

    # method: get_previous_annotation
    #
    # arguments: none
    #
    # returns: off screen annotation
    #
    # this method is called when user selects one of the arrows in the ui
    # all of these methods allow for navigation through record by annotations
    #
    def get_previous_annotation(self,
                                slider_current_pos_a):
        axis_left_bound = int(slider_current_pos_a)
        return next(x[1] for x in reversed(
            self.sorted_annos) if x[1] < axis_left_bound)

    def get_next_annotation(self,
                            slider_current_pos_a):
        axis_right_bound = int(slider_current_pos_a) + self.time_scale
        return next(x[1] for x in \
                    self.sorted_annos if x[1] >= axis_right_bound)

    def get_first_annotation(self):
        return self.sorted_annos[0][1]

    def get_last_annotation(self):
        last_annotation_index = len(self.sorted_annos)
        return self.sorted_annos[last_annotation_index - 1][1]

    # method: update_annotation_preferences
    #
    # arguments: values from DemoPreferences and beyond
    #
    # returns: none
    #
    # this method is called when 'apply' is pressed in the DemoPreferencesWidget,
    # updates values in cfg_dict_single_anno to the values entered by user
    # in the preferences window.
    #
    def update_annotation_preferences(self,
                                      handle_color_a,
                                      handle_size_a,
                                      default_border_w_a,
                                      selected_border_w_a,
                                      label_color_a,
                                      label_font_size_a):

        self.cfg_dict_single_anno['pen_handle'] = handle_color_a
        self.cfg_dict_single_anno['handle_size'] = handle_size_a
        self.cfg_dict_single_anno['border_width_default'] = default_border_w_a
        self.cfg_dict_single_anno['border_width_selected'] = selected_border_w_a
        self.cfg_dict_single_anno['lbl_color'] = label_color_a
        self.cfg_dict_single_anno['lbl_font_size'] = label_font_size_a
