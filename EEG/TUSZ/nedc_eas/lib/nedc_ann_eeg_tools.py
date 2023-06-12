#!/usr/bin/env python
#
# file: $NEDC_NFC/class/python/nedc_ann_tools/nedc_ann_tools.py
#
# revision history:
# 20220514 (JP): updated some things to fix bugs in the eeg scoring software
# 20220307 (PM): configured the CSV Class to support the new CSV format
# 20210201 (TC): added XML and CSV
# 20200610 (LV): refactored code
# 20200607 (JP): refactored code
# 20170728 (JP): added compare_durations and load_annotations
# 20170716 (JP): upgraded to use the new annotation tools
# 20170714 (NC): created new class structure
# 20170709 (JP): refactored the code
# 20170612 (NC): added parsing and displaying methods
# 20170610 (JP): initial version
#
# This class contains a collection of methods that provide
# the infrastructure for processing annotation-related data.
#------------------------------------------------------------------------------

# import reqired system modules
#
import copy
from operator import itemgetter
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from lxml import etree
from pathlib import Path
from pprint import pprint
from xml.dom import minidom

# import required NEDC modules
#
import nedc_debug_tools as ndt
import nedc_file_tools as nft
import nedc_eval_common as nec


#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define a data structure that encapsulates all file types:
#  we use this data structure to access lower-level objects. the key
#  is the type name, the first value is the magic sequence that should
#  appear in the file and the second value is the name of class data member
#  that is used to dynamically bind the subclass based on the type.
#
#  we also need a list of supported versions for utilities to use.
#
FTYPE_LBL = 'lbl_v1.0.0'
FTYPE_TSE = 'tse_v1.0.0'
FTYPE_CSV = 'csv_v1.0.0'
FTYPE_XML = '1.0'

FTYPES = {'lbl': [FTYPE_LBL, 'lbl_d'], 'tse': [FTYPE_TSE, 'tse_d'],
          'csv': [FTYPE_CSV, 'csv_d'], 'xml': [FTYPE_XML, 'xml_d']}
VERSIONS = [FTYPE_LBL, FTYPE_TSE, FTYPE_CSV, FTYPE_XML]

# define numeric constants
#
DEF_CHANNEL = int(-1)
PRECISION = int(4)

# define the string to check the files' header
#
DEF_CSV_HEADER = "# version = csv_v1.0.0"
DEF_CSV_LABELS = "channel,start_time,stop_time,label,confidence"

# define keys for header information specific to annotations
#
CKEY_VERSION = 'version'
CKEY_BNAME = 'bname'
CKEY_DURATION = 'duration'
CKEY_MONTAGE_FILE = 'montage_file'
CKEY_ANNOTATION_LABEL_FILE = 'annotation_label_file'

# define constants for term based representation in the dictionary
#
DEF_TERM_BASED_IDENTIFIER = "TERM"

# define a default montage file
#
DEFAULT_MAP_FNAME = "$NEDC_NFC/lib/nedc_eas_default_map.txt"

DEFAULT_MONTAGE_FNAME = "$NEDC_NFC/lib/nedc_eas_default_montage.txt"

# define symbols that appear as keys in an lbl file
#
DELIM_LBL_MONTAGE = 'montage'
DELIM_LBL_NUM_LEVELS = 'number_of_levels'
DELIM_LBL_LEVEL = 'level'
DELIM_LBL_SYMBOL = 'symbols'
DELIM_LBL_LABEL = 'label'

# define Csv Class labels
#
DEF_CSV_ANNOTATION = "annotation"

# define a list of characters we need to parse out
#
REM_CHARS = [nft.DELIM_BOPEN, nft.DELIM_BCLOSE, nft.DELIM_NEWLINE,
             nft.DELIM_SPACE, nft.DELIM_QUOTE, nft.DELIM_SEMI,
             nft.DELIM_SQUOTE]

# define constants associated with the Xml class
#
DEF_XML_CHANNEL_PATH = "label/montage_channels/channel"
DEF_XML_EVENT = "event"
DEF_XML_ROOT = "root"
DEF_XML_BNAME = "bname"
DEF_XML_DURATION = "duration"
DEF_XML_LABEL = "label"
DEF_XML_NAME = "name"
DEF_XML_PROBABILITY = "probability" 
DEF_XML_ENDPOINTS = "endpoints"
DEF_XML_CHANNEL = "channel"
DEF_XML_MONTAGE_CHANNELS = "montage_channels"
DEF_XML_ANNOTATION_LABEL_FILE = "annotation_label_file"
DEF_XML_MONTAGE_FILE = "montage_file"

# define the location of the schema file:
#  note that this is version specific since the schema file will evolve
#  over time.
#
DEFAULT_XML_SCHEMA_FILE = "$NEDC_NFC/lib/nedc_eeg_xml_schema_v00.xsd"

# define types check
#
PARENT_TYPE = 'parent'
EVENT_TYPE = 'event'
LIST_TYPE = 'list'
DICT_TYPE = 'dict'

# define montage regex 
#
DEF_REGEX_MONTAGE_FILE = re.compile(r'(-?\d+(?=,)),(\w+(?:-?)+\w+(?=:))', 
                         re.IGNORECASE)
DEF_NEDC_EAS_MAP_REGEX = re.compile("(.+?(?==))=\((\d+),(\d+),(\(\d+,\d+,\d+,\d+\))\)", 
    re.IGNORECASE)

#------------------------------------------------------------------------------
#
# functions listed here
#
#------------------------------------------------------------------------------

# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()

# function: get_unique_events
#
# arguments:
#  events: events to aggregate
#
# return: a list of unique events
#
# This method combines events if they are of the same start/stop times.
#
def get_unique_events(events):

    # list to store unique events
    #
    unique_events = []

    # make sure events_a are sorted
    #
    events = sorted(events, key=lambda x: (x[0], x[1]))

    # loop until we have checked all events_a
    #
    while len(events) != 0:

        # reset flag
        #
        is_unique = True
        n_start = int(-1)
        n_stop = int(-1)

        # get this event's start/stop times
        #
        start = events[0][0]
        stop = events[0][1]

        # if we are not at the last event
        #
        if len(events) != 1:

            # get next event's start/stop times
            #
            n_start = events[1][0]
            n_stop = events[1][1]

        # if this event's start/stop times are the same as the next event's,
        #  (only do this if we are not at the last event)
        #
        if (n_start == start) and (n_stop == stop) and (len(events) != 1):

            # combine this event's dict with the next event's symbol dict
            #
            for symb in events[1][2]:

                # if the symb is not found in this event's dict
                #
                if symb not in events[0][2]:

                    # add symb to this event's dict
                    #
                    events[0][2][symb] = events[1][2][symb]

                # else if the symb in the next event has a higher prob
                #
                elif events[1][2][symb] > events[0][2][symb]:

                    # update this event's symb with prob from the next event
                    #
                    events[0][2][symb] = events[1][2][symb]
                #
                # end of if/elif
            #
            # end of for

            # delete the next event, it is not unique
            #
            del events[1]
        #
        # end of if

        # loop over unique events
        #
        for unique in unique_events:

            # if the start/stop times of this event is found in unique events
            #
            if (start == unique[0]) and (stop == unique[1]):

                # combine unique event's dict with this event's dict:
                #  iterate over symbs in this event's dict
                #
                for symb in events[0][2]:

                    # if the symb is not found in the unique event's dict
                    #
                    if symb not in unique[2]:

                        # add symb to the unique event's dict
                        #
                        unique[2][symb] = events[0][2][symb]

                    # else if the symb in this event has a higher prob
                    #
                    elif events[0][2][symb] > unique[2][symb]:

                        # update unique event's symb with prob from this event
                        #
                        unique[2][symb] = events[0][2][symb]
                    #
                    # end of if/elif
                #
                # end of for

                # delete this event, it is not unique
                #
                del events[0]
                is_unique = False
                break
            #
            # end of if
        #
        # end of for

        # if this event is still unique
        #
        if is_unique is True:

            # add this event to the unique events
            #
            unique_events.append(events[0])

            # delete this event, it is now stored as unique
            #
            del events[0]
        #
        # end of if
    #
    # end of while

    # exit gracefully
    #
    return unique_events
#
# end of function

# function: compare_durations
#
# arguments:
#  l1: the first list of files
#  l2: the second list of files
#
# return: a boolean value indicating status
#
# This method goes through two lists of files and compares the durations
# of the annotations. If they don't match, it returns false.
#
def compare_durations(l1, l2):

    # display an informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: comparing durations of annotations" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # create an annotation object
    #
    ann = AnnEeg()
    
    # check the length of the lists
    #
    if len(l1) != len(l2):
        return False

    # loop over the lists together
    #
    for l1_i, l2_i in zip(l1, l2):

        # load the annotations for l1
        #
        if ann.load(l1_i) == False:
            print("Error: %s (line: %s) %s: error loading annotation (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, l1_i))
            return False
        
        # get the events for l1
        #
        events_l1 = ann.get()
        
        # sort the event
        #
        events_l1.sort(key = itemgetter(0))
        
        # fill in all the gap annotation with BCKG
        #
        events_l1 = nec.augment_annotation(events_l1, ann.get_file_duration())
        
        # join all the BCKG events together 
        #
        events_l1 = nec.remove_repeated_events(events_l1)

        if events_l1 == None:
            print("Error: %s (line: %s) %s: error getting annotation ((%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, l1_i))
            return False

        # load the annotations for l2
        #
        if ann.load(l2_i) == False:
            print("Error: %s (line: %s) %s: error loading annotation (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, l2_i))
            return False
        
        # get the events for l2
        #
        events_l2 = ann.get()

        # sort the event
        #
        events_l2.sort(key = itemgetter(0))
        
        # fill in all the gap annotation with BCKG
        #
        events_l2 = nec.augment_annotation(events_l2, ann.get_file_duration())
        
        # join all the BCKG events together 
        #
        events_l2 = nec.remove_repeated_events(events_l2)

        if events_l2 == None:
            print("Error: %s (line: %s) %s: error getting annotation: (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, l2_i))
            return False
        
        # check the durations
        #
        if round(events_l1[-1][1], ndt.MAX_PRECISION) != \
           round(events_l2[-1][1], ndt.MAX_PRECISION):
            print("Error: %s (line: %s) %s: durations do not match" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            print("\t%s (%f)" % (l1_i, events_l1[-1][1]))
            print("\t%s (%f)" % (l2_i, events_l2[-1][1]))
            return False

    # exit gracefully
    #
    return True
#
# end of function

# function: load_annotations
#
# arguments:
#  list: a list of filenames
#
# return: a list of lists containing all the annotations
#
# This method loops through a list and collects all the annotations.
#
def load_annotations(flist, level=int(0), sublevel=int(0),
                     channel=DEF_CHANNEL):

    # display an informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: loading annotations" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # create an annotation object
    #
    events = []
    ann = AnnEeg()

    # loop over the list
    #
    for fname in flist:

        # load the annotations
        #
        if ann.load(fname) == False:
            print("Error: %s (line: %s) %s: loading annotation for file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return None

        # get the events
        #
        events_tmp = ann.get(level, sublevel, channel)
        if events_tmp == None:
            print("Error: %s (line: %s) %s: error getting annotation (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return None

        events_tmp.sort(key = itemgetter(0))
        
        # fill in all the gap annotation with BCKG
        #
        events_tmp = nec.augment_annotation(events_tmp, ann.get_file_duration())
        
        # join all the BCKG events together 
        #
        events_tmp = nec.remove_repeated_events(events_tmp)

        events.append(events_tmp)

    # exit gracefully
    #
    return events

# function: parse_nedc_eas_map_to_montage_defintion
#
# arguments:
#  map_file: the nedc_eas map file configuation
#
# return: a dictionary to parse the map file
#
# note:
# ** NEDC_EAS MAP FILE CONFIGURATION ** 
# KEY   | MAPPING     PRIORITY    RGB_COLOR_SCHEME
# null = ( 0,          0,          (  0 ,  0,   0,  10))
#
# We want to extract only the KEY and MAPPING to build this dictionary:
#
# map_dictionary = { mapping: key}
# 
# Ex: map_dictionary = { 'null' : 0 }
#
def parse_nedc_eas_map_to_montage_defintion(map_file):
    
    map_dictionary  = {}
    
    with open(map_file) as file:

        for ind, line in enumerate(file):

            line = line.strip().replace(" ", "")

            if len(line) == 0 or line.startswith("#") or line.startswith("[") or line.startswith("symbols"):
                continue
            
            # pattern matching
            #
            result = re.findall(DEF_NEDC_EAS_MAP_REGEX, line)[0]

            if len(result) < 4:
                raise Exception(f"Map File Configuration invalid on line {ind + 1}")

            key, mapping, priority, rgb_val = result[0], int(result[1]), int(result[2]), eval(result[3])

            map_dictionary[mapping] = key
    
    return map_dictionary
#
# end of function

#------------------------------------------------------------------------------
#
# + Classes are listed here:
#  There are six classes in this file arranged in this hierarchy
#   AnnGrEeg -> {Tse, Lbl, Csv, Xml} -> AnnEeg
#
# + Breakdown of Ann_EEG_Tools:
#   
#   AnnGrEeg : The basic data structure that this whole library uses
#   Tse      : The class that deals with Tse files
#   Lbl      : The class that deals with Lbl files
#   Csv      : The class that deals with Csv files
#   Xml      : The class that deals with Xml files
#   AnnEeg   : This is a wrapper for all the other classes. 
#              You would ONLY need to instantiate this class. 
#
#   Between the four classes {Tse, Lbl, Csv, Xml}, each of the classes share
#   a common method that has the same name (it is important that their name is
#   is the same for AnnEeg to work).
#  
#   Here is the common method:
#       + load()
#       + write()
#       + display()
#       + add()
#       + delete()
#       + get()
#       + get_graph()
#       + set_graph()
#
#   Nedc_ann_eeg_tools works by using the AnnEeg class to automatically call 
#   correct method for the correct file type. So DO NOT REMOVE any of the common
#   method pointed out above. 
#
# + Graphing Object Structure:
#
#   Below is the return Graphing Object Structure:
#
#       graph = { level { sublevel { channel_index : [[start_time, stop time, {'label': probability}],
#                                                      ...]}}}
#       level: int 
#       sublevel: int
#       channel_index: int
#       start_time: float
#       stop_time:  float
#       label: string
#       probability: float
#
#   Ex:
#       graph = { 0 { 0 { 2 : [[55.1234, 60.0000, {'elec': 1.0}],
#                              [65.1234, 70.0000, {'chew': 1.0}]
#                              ]}}}
#
#------------------------------------------------------------------------------

# class: AnnGrEeg
#
# This class implements the main data structure used to hold an annotation.
#
class AnnGrEeg:

    # method: AnnGrEeg::constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self):

        # set the class name
        #
        AnnGrEeg.__CLASS_NAME__ = self.__class__.__name__

        # declare a data structure to hold a graph
        #
        self.graph_d = {}
    #
    # end of method

    # method: AnnGrEeg::create
    #
    # arguments:
    #  lev: level of annotation
    #  sub: sublevel of annotation
    #  chan: channel of annotation
    #  start: start time of annotation
    #  stop: stop time of annotation
    #  symbols: dict of symbols/probabilities
    #
    # return: a boolean value indicating status
    #
    # This method create an annotation in the AG data structure
    #
    def create(self, lev, sub, chan, start, stop, symbols):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "creating annotation in AG data structure"))

        # try to access sublevel dict at level
        #

        try:
            self.graph_d[lev]

            # try to access channel dict at level/sublevel
            #
            try:
                self.graph_d[lev][sub]

                # try to append values to channel key in dict
                #
                try:
                    self.graph_d[lev][sub][chan].append([start, stop, symbols])

                # if appending values failed, finish data structure
                #
                except:

                    # create empty list at chan key
                    #
                    self.graph_d[lev][sub][chan] = []

                    # append values
                    #
                    self.graph_d[lev][sub][chan].append([start, stop, symbols])

            # if accessing channel dict failed, finish data structure
            #
            except:

                # create dict at level/sublevel
                #
                self.graph_d[lev][sub] = {}

                # create empty list at chan
                #
                self.graph_d[lev][sub][chan] = []

                # append values
                #
                self.graph_d[lev][sub][chan].append([start, stop, symbols])

        # if accessing sublevel failed, finish data structure
        #
        except:

            # create dict at level
            #
            self.graph_d[lev] = {}

            # create dict at level/sublevel
            #
            self.graph_d[lev][sub] = {}

            # create empty list at level/sublevel/channel
            #
            self.graph_d[lev][sub][chan] = []

            # append values
            #
            self.graph_d[lev][sub][chan].append([start, stop, symbols])

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: AnnGrEeg::get
    #
    # arguments:
    #  level: level of annotations
    #  sublevel: sublevel of annotations
    #
    # return: events by channel at level/sublevel
    #
    # This method returns the events stored at the level/sublevel argument
    #
    def get(self, level, sublevel, channel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: getting events stored at level/sublevel" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # declare local variables
        #
        events = []
        
        # try to access graph at level/sublevel/channel
        #
        try:
            events = self.graph_d[level][sublevel][channel]

            # exit gracefully
            #
            return events

        # exit (un)gracefully: if failed, return False
        #
        except:
            return False
    #
    # end of method

    # method: AnnGrEeg::sort
    #
    # arguments: none
    #
    # return: a boolean value indicating status
    #
    # This method sorts annotations by level, sublevel,
    # channel, start, and stop times
    #
    def sort(self):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: %s %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "sorting annotations by",
                   "level, sublevel, channel, start and stop times"))

        # sort each level key by min value
        #
        self.graph_d = dict(sorted(self.graph_d.items()))

        # iterate over levels
        #
        for lev in self.graph_d:

            # sort each sublevel key by min value
            #
            self.graph_d[lev] = dict(sorted(self.graph_d[lev].items()))

            # iterate over sublevels
            #
            for sub in self.graph_d[lev]:

                # sort each channel key by min value
                #
                self.graph_d[lev][sub] = \
                    dict(sorted(self.graph_d[lev][sub].items()))

                # iterate over channels
                #
                for chan in self.graph_d[lev][sub]:

                    # sort each list of labels by start and stop times
                    #
                    self.graph_d[lev][sub][chan] = \
                        sorted(self.graph_d[lev][sub][chan],
                               key=lambda x: (x[0], x[1]))

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: AnnGrEeg::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur, sym, level, sublevel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: adding events of type sym" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # try to access level/sublevel
        #
        try:
            self.graph_d[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, AnnGrEeg.__CLASS_NAME__,
                   ndt.__NAME__, "level/sublevel not found", level, sublevel))
            return False

        # variable to store what time in the file we are at
        #
        mark = 0.0

        # make sure events are sorted
        #
        self.sort()

        # iterate over channels at level/sublevel
        #
        for chan in self.graph_d[level][sublevel]:

            # reset list to store events
            #
            events = []

            # iterate over events at each channel
            #
            for event in self.graph_d[level][sublevel][chan]:

                # ignore if the start or stop time is past the duration
                #
                if (event[0] > dur) or (event[1] > dur):
                    pass

                # ignore if the start time is bigger than the stop time
                #
                elif event[0] > event[1]:
                    pass

                # ignore if the start time equals the stop time
                #
                elif event[0] == event[1]:
                    pass

                # if the beginning of the event is not at the mark
                #
                elif event[0] != mark:

                    # create event from mark->starttime
                    #
                    events.append([mark, event[0], {sym: 1.0}])

                    # add event after mark->starttime
                    #
                    events.append(event)

                    # set mark to the stop time
                    #
                    mark = event[1]

                # if the beginning of the event is at the mark
                #
                else:

                    # store this event
                    #
                    events.append(event)

                    # set mark to the stop time
                    #
                    mark = event[1]
            #
            # end of for

            # after iterating through all events, if mark is not at dur
            #
            if mark != dur:

                # create event from mark->dur
                #
                events.append([mark, dur, {sym: 1.0}])

            # store events as the new events in self.graph_d
            #
            self.graph_d[level][sublevel][chan] = events
        #
        # end of for

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: AnnGrEeg::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym
    #
    def delete(self, sym, level, sublevel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: deleting events of type sym" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # try to access level/sublevel
        #
        try:
            self.graph_d[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, AnnGrEeg.__CLASS_NAME__,
                   ndt.__NAME__, "level/sublevel not found", level, sublevel))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in self.graph_d[level][sublevel]:

            # get events at chan
            #
            events = self.graph_d[level][sublevel][chan]

            # keep only the events that do not contain sym
            #
            events = [e for e in events if sym not in e[2].keys()]

            # store events in self.graph_d
            #
            self.graph_d[level][sublevel][chan] = events
        #
        # end of for

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: AnnGrEeg::get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method returns the entire graph, instead of a
    # level/sublevel/channel.
    #
    def get_graph(self):
        return copy.deepcopy(self.graph_d)
    #
    # end of method

    # method: AnnGrEeg::set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph.
    #
    def set_graph(self, graph):
        self.graph_d = graph
        self.sort()
        return True
    #
    # end of method

    # method: AnnGrEeg::delete_graph
    #
    def delete_graph(self):
        self.graph_d  = {}
        return True
#
# end of class

# class: Tse
#
# This class contains methods to manipulate time-synchronous event files.
#
class Tse:

    # method: Tse::constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self):

        # set the class name
        #
        Tse.__CLASS_NAME__ = self.__class__.__name__

        # declare Graph object, to store annotations
        #
        self.graph_d = AnnGrEeg()
    #
    # end of method

    # method: Tse::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: loading annotation from file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # open file
        #
        with open(fname, nft.MODE_READ_TEXT) as fp:

            # loop over lines in file
            #
            for line in fp:

                # clean up the line
                #
                line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                           .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)
                check = line.replace(nft.DELIM_SPACE, nft.DELIM_NULL)

                # throw away commented, blank lines, version lines
                #
                if check.startswith(nft.DELIM_COMMENT) or \
                   check.startswith(nft.DELIM_VERSION) or \
                   len(check) == 0:
                    continue

                # split the line
                #
                val = {}
                parts = line.split()

                try:
                    # loop over every part, starting after start/stop times
                    #
                    for i in range(2, len(parts), 2):

                        # create dict with label as key, prob as value
                        #
                        val[parts[i]] = float(parts[i+1])

                    # create annotation in AG
                    #
                    self.graph_d.create(int(0), int(0), int(-1),
                                        float(parts[0]), float(parts[1]), val)
                except:
                    print("Error: %s (line: %s) %s::%s %s (%s)" %
                          (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__,
                           ndt.__NAME__, "invalid annotation", line))
                    return False

        # make sure graph is sorted after loading
        #
        self.graph_d.sort()

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Tse::get
    #
    # arguments:
    #  level: level of annotations to get
    #  sublevel: sublevel of annotations to get
    #
    # return: events at level/sublevel by channel
    #
    # This method gets the annotations stored in the AG at level/sublevel.
    #
    def get(self, level, sublevel, channel):
        events = self.graph_d.get(level, sublevel, channel)
        return events
    #
    # end of method

    # method: Tse::display
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a boolean value indicating status
    #
    # This method displays the events from a flat AG.
    #
    def display(self, level, sublevel, fp=sys.stdout):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: displaying events from flag AG" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__, ndt.__NAME__,
                   "level/sublev not in graph", level, sublevel))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in graph[level][sublevel]:

            # iterate over events for each channel
            #
            for event in graph[level][sublevel][chan]:
                start = event[0]
                stop = event[1]

                # create a string with all symb/prob pairs
                #
                pstr = ""
                for symb in event[2]:
                    pstr += f" {symb:>8} {event[2][symb]:10.{PRECISION}f}"

                # display event
                #
                fp.write(f"{'ALL':>10}: {start:10.{PRECISION}f} {stop:10.{PRECISION}f}{pstr}\n")

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Tse::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method writes the events to a .tse file
    #
    def write(self, ofile, level, sublevel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: writing events to .tse file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # make sure graph is sorted
        #
        self.graph_d.sort()

        # get graph
        #
        graph = self.get_graph()

        # try to access the graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__,
                   ndt.__NAME__, "level/sublevel not in graph",
                   level, sublevel))
            return False

        # list to collect all events
        #
        events = []

        # iterate over channels at level/sublevel
        #
        for chan in graph[level][sublevel]:

            # iterate over events for each channel
            #
            for event in graph[level][sublevel][chan]:

                # store every channel's events in one list
                #
                events.append(event)

        # remove any events that are not unique
        #
        events = get_unique_events(events)

        # open file with write
        #
        with open(ofile, nft.MODE_WRITE_TEXT) as fp:

            # write version
            #
            fp.write("version = %s\n" % FTYPES['tse'][0])
            fp.write(nft.DELIM_NEWLINE)

            # iterate over events
            #
            for event in events:

                # create symb/prob string from dict
                #
                pstr = ""
                for symb in event[2]:
                    pstr += f" {symb} {event[2][symb]:.{PRECISION}f}"

                # write event
                #
                fp.write(f"{event[0]:.{PRECISION}f} {event[1]:.{PRECISION}f}{pstr}\n")

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Tse::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur, sym, level, sublevel):
        return self.graph_d.add(dur, sym, level, sublevel)
    #
    # end of method

    # method: Tse::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym.
    #
    def delete(self, sym, level, sublevel):
        return self.graph_d.delete(sym, level, sublevel)
    #
    # end of method

    # method: Tse::get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()
    #
    # end of method

    # method: Tse::set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph.
    #
    def set_graph(self, graph):
        return self.graph_d.set_graph(graph)
    #
    # end of method

#
# end of class

# class: Lbl
#
# This class implements methods to manipulate label files.
#
class Lbl:

    # method: Lbl::constructor
    #
    # arguments: none
    #
    # return: none
    #
    # This method constructs Ag
    #
    def __init__(self):

        # set the class name
        #
        Lbl.__CLASS_NAME__ = self.__class__.__name__

        # declare variables to store info parsed from lbl file
        #
        self.chan_map_d = {DEF_CHANNEL: DEF_TERM_BASED_IDENTIFIER}
        self.montage_lines_d = []
        self.symbol_map_d = {}
        self.num_levels_d = int(1)
        self.num_sublevels_d = {int(0): int(1)}

        # declare AG object to store annotations
        #
        self.graph_d = AnnGrEeg()
    #
    # end of method

    # method: Lbl::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: loading annotation from file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # open file
        #
        fp = open(fname, nft.MODE_READ_TEXT)

        # loop over lines in file
        #
        for line in fp:

            # clean up the line
            #
            line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                       .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)

            # parse a single montage definition
            #
            if line.startswith(DELIM_LBL_MONTAGE):
                try:
                    chan_num, name, montage_line = \
                        self.parse_montage(line)
                    self.chan_map_d[chan_num] = name
                    self.montage_lines_d.append(montage_line)
                except:
                    print("Error: %s (line: %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing montage", line))
                    fp.close()
                    return False

            # parse the number of levels
            #
            elif line.startswith(DELIM_LBL_NUM_LEVELS):
                try:
                    self.num_levels_d = self.parse_numlevels(line)
                except:
                    print("Error: %s (line: %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing number of levels",
                           line))
                    fp.close()
                    return False

            # parse the number of sublevels at a level
            #
            elif line.startswith(DELIM_LBL_LEVEL):
                try:
                    level, sublevels = self.parse_numsublevels(line)
                    self.num_sublevels_d[level] = sublevels

                except:
                    print("Error: %s (line: %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing num of sublevels",
                           line))
                    fp.close()
                    return False

            # parse symbol definitions at a level
            #
            elif line.startswith(DELIM_LBL_SYMBOL):
                try:
                    level, mapping = self.parse_symboldef(line)
                    self.symbol_map_d[level] = mapping
                except:
                    print("Error: %s (line %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing symbols", line))
                    fp.close()
                    return False

            # parse a single label
            #
            elif line.startswith(DELIM_LBL_LABEL):
                lev, sub, start, stop, chan, symbols = \
                    self.parse_label(line)
                try:
                    lev, sub, start, stop, chan, symbols = \
                        self.parse_label(line)
                except:
                    print("Error: %s (line %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing label", line))
                    fp.close()
                    return False

                # create annotation in AG
                #
                status = self.graph_d.create(lev, sub, chan,
                                             start, stop, symbols)

        # close file
        #
        fp.close()

        # sort labels after loading
        #
        self.graph_d.sort()

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: Lbl::get
    #
    # arguments:
    #  level: level value
    #  sublevel: sublevel value
    #
    # return: events by channel from AnnGrEeg
    #
    # This method returns the events at level/sublevel
    #
    def get(self, level, sublevel, channel):

        # get events from AG
        #
        events = self.graph_d.get(level, sublevel, channel)

        # exit gracefully
        #
        return events
    #
    # end of method

    # method: Lbl::display
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a boolean value indicating status
    #
    # This method displays the events from a flat AG
    #
    def display(self, level, sublevel, fp=sys.stdout):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: displaying events from flat AG" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get graph
        #
        graph = self.get_graph()

        # try to access level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            sys.stdout.write("Error: %s (line: %s) %s::%s: %s (%d/%d)" %
                             (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                              ndt.__NAME__, "level/sublevel not found",
                              level, sublevel))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in graph[level][sublevel]:

            # iterate over events at chan
            #
            for event in graph[level][sublevel][chan]:

                # find max probability
                #
                max_prob = max(event[2].values())

                # iterate over symbols in dictionary
                #
                for symb in event[2]:

                    # if the value of the symb equals the max prob
                    #
                    if event[2][symb] == max_prob:

                        # set max symb to this symbol
                        #
                        max_symb = symb
                        break

                # display event
                #
                fp.write(f"{self.chan_map_d[chan]:>10}: \
                            {event[0]:10.{PRECISION}f} \
                            {event[1]:10.{PRECISION}f} {max_symb:>8} \
                            {max_prob:10.{PRECISION}f}\n")

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Lbl::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method writes events to a .lbl file.
    #
    def write(self, ofile, level, sublevel):

        # make sure graph is sorted
        #
        self.graph_d.sort()

        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s: %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "level/sublevel not found", level, sublevel))
            return False

        # open file with write
        #
        with open(ofile, nft.MODE_WRITE_TEXT) as fp:

            # write version
            #
            fp.write(nft.DELIM_NEWLINE)
            fp.write("version = %s\n" % FTYPES['lbl'][0])
            fp.write(nft.DELIM_NEWLINE)

            # if montage_lines is blank, we are converting from tse to lbl.
            #
            # create symbol map from tse symbols
            #
            if len(self.montage_lines_d) == 0:

                # variable to store the number of symbols
                #
                num_symbols = 0

                # create a dictionary at level 0 of symbol map
                #
                self.symbol_map_d[int(0)] = {}

                # iterate over all events stored in the 'all' channels
                #
                for event in graph[level][sublevel][int(-1)]:

                    # iterate over symbols in each event
                    #
                    for symbol in event[2]:

                        # if the symbol is not in the symbol map
                        #
                        if symbol not in self.symbol_map_d[0].values():

                            # map num_symbols interger to symbol
                            #
                            self.symbol_map_d[0][num_symbols] = symbol

                            # increment num_symbols
                            #
                            num_symbols += 1

            # write montage lines
            #
            for line in self.montage_lines_d:
                fp.write("%s\n" % line)

            fp.write(nft.DELIM_NEWLINE)

            # write number of levels
            #
            fp.write("number_of_levels = %d\n" % self.num_levels_d)
            fp.write(nft.DELIM_NEWLINE)

            # write number of sublevels
            #
            for lev in self.num_sublevels_d:
                fp.write("level[%d] = %d\n" %
                         (lev, self.num_sublevels_d[lev]))
            fp.write(nft.DELIM_NEWLINE)

            # write symbol definitions
            #
            for lev in self.symbol_map_d:
                fp.write("symbols[%d] = %s\n" %
                         (lev, str(self.symbol_map_d[lev])))
            fp.write(nft.DELIM_NEWLINE)

            # iterate over channels at level/sublevel
            #
            for chan in graph[level][sublevel]:

                # iterate over events in chan
                #
                for event in graph[level][sublevel][chan]:

                    # create string for probabilities
                    #
                    pstr = f"{nft.DELIM_OPEN}"

                    # iterate over symbol map
                    #
                    for symb in self.symbol_map_d[level].values():

                        # if the symbol is found in the event
                        #
                        if symb in event[2]:
                            pstr += (str(event[2][symb]) + nft.DELIM_COMMA +
                                     nft.DELIM_SPACE)
                        else:
                            pstr += '0.0' + nft.DELIM_COMMA + nft.DELIM_SPACE

                    # remove the ', ' from the end of pstr
                    #
                    pstr = pstr[:len(pstr) - 2] + f"{nft.DELIM_CLOSE}{nft.DELIM_BCLOSE}"

                    # write event
                    #
                    fp.write(f"label = {level}, {sublevel}, {event[0]:.{PRECISION}f}, {event[1]:.{PRECISION}f}, {chan}, {pstr}\n") 

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Lbl::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym
    #
    def add(self, dur, sym, level, sublevel):
        return self.graph_d.add(dur, sym, level, sublevel)

    # method: Lbl::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym
    #
    def delete(self, sym, level, sublevel):
        return self.graph_d.delete(sym, level, sublevel)

    # method: Lbl::get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()
    #
    # end of method

    # method: Lbl::set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph
    #
    def set_graph(self, graph):
        return self.graph_d.set_graph(graph)
    #
    # end of method

    # method: Lbl::parse_montage
    #
    # arguments:
    #  line: line from label file containing a montage channel definition
    #
    # return:
    #  channel_number: an integer containing the channel map number
    #  channel_name: the channel name corresponding to channel_number
    #  montage_line: entire montage def line read from file
    #
    # This method parses a montage line into it's channel name and number.
    # Splitting a line by two values easily allows us to get an exact
    # value/string from a line of definitions
    #
    def parse_montage(self, line):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: parsing montage by channel name, number" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # split between '=' and ',' to get channel number
        #
        channel_number = int(
            line.split(nft.DELIM_EQUAL)[1].split(nft.DELIM_COMMA)[0].strip())

        # split between ',' and ':' to get channel name
        #
        channel_name = line.split(
            nft.DELIM_COMMA)[1].split(nft.DELIM_COLON)[0].strip()

        # remove chars from montage line
        #
        montage_line = line.strip().strip(nft.DELIM_NEWLINE)

        # exit gracefully
        #
        return [channel_number, channel_name, montage_line]
    #
    # end of method

    # method: Lbl::parse_numlevels
    #
    # arguments:
    #  line: line from label file containing the number of levels
    #
    # return: an integer containing the number of levels defined in the file
    #
    # This method parses the number of levels in a file.
    #
    def parse_numlevels(self, line):

        # split by '=' and remove extra characters
        #
        return int(line.split(nft.DELIM_EQUAL)[1].strip())
    #
    # end of method

    # method: Lbl::parse_numsublevels
    #
    # arguments:
    #  line: line from label file containing number of sublevels in level
    #
    # return:
    #  level: level from which amount of sublevels are contained
    #  sublevels: amount of sublevels in particular level
    #
    # This method parses the number of sublevels per level in the file
    #
    def parse_numsublevels(self, line):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: parsing number of sublevels per level" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # split between '[' and ']' to get level
        #
        level = int(line.split(
            nft.DELIM_OPEN)[1].split(nft.DELIM_CLOSE)[0].strip())

        # split by '=' and remove extra characters
        #
        sublevels = int(line.split(nft.DELIM_EQUAL)[1].strip())

        # exit gracefully
        #
        return [level, sublevels]
    #
    # end of method

    # method: Lbl::parse_symboldef
    #
    # arguments:
    #  line: line from label fiel containing symbol definition for a level
    #
    # return:
    #  level: an integer containing the level of this symbol definition
    #  mappings: a dict containing the mapping of symbols for this level
    #
    # This method parses a symbol definition line into a specific level,
    # the corresponding symbol mapping as a dictionary.
    #
    def parse_symboldef(self, line):

        # split by '[' and ']' to get level of symbol map
        #
        level = int(line.split(nft.DELIM_OPEN)[1].split(nft.DELIM_CLOSE)[0])

        # remove all characters to remove, and split by ','
        #
        syms = ''.join(c for c in line.split(nft.DELIM_EQUAL)[1]
                       if c not in REM_CHARS)

        symbols = syms.split(nft.DELIM_COMMA)

        # create a dict from string, split by ':'
        #   e.g. '0: seiz' -> mappings[0] = 'seiz'
        #
        mappings = {}
        for s in symbols:
            mappings[int(s.split(':')[0])] = s.split(':')[1]

        # exit gracefully
        #
        return [level, mappings]
    #
    # end of method

    # method: Lbl::parse_label
    #
    # arguments:
    #  line: line from label file containing an annotation label
    #
    # return: all information read from .ag file
    #
    # this method parses a label definition into the values found in the label
    #
    def parse_label(self, line):

        # dict to store symbols/probabilities
        #
        symbols = {}

        # remove characters to remove, and split data by ','
        #
        lines = ''.join(c for c in line.split(nft.DELIM_EQUAL)[1]
                        if c not in REM_CHARS)

        data = lines.split(nft.DELIM_COMMA)

        # separate data into specific variables
        #
        level = int(data[0])
        sublevel = int(data[1])
        start = float(data[2])
        stop = float(data[3])

        # the channel value supports either 'all' or channel name
        #
        try:
            channel = int(data[4])
        except:
            channel = int(-1)

        # parse probabilities
        #
        probs = lines.split(
            nft.DELIM_OPEN)[1].strip(nft.DELIM_CLOSE).split(nft.DELIM_COMMA)

        # set every prob in probs to type float
        #
        probs = list(map(float, probs))

        # convert the symbol map values to a list
        #
        map_vals = list(self.symbol_map_d[level].values())

        # iterate over symbols
        #
        for i in range(len(self.symbol_map_d[level].keys())):

            if probs[i] > 0.0:

                # set each symbol equal to the corresponding probability
                #
                symbols[map_vals[i]] = probs[i]

        # exit gracefully
        #
        return [level, sublevel, start, stop, channel, symbols]
    #
    # end of method

    # method: Lbl::update_montage
    #
    # arguments:
    #  montage_file: montage file
    #
    # return: a boolean value indicating status
    #
    # this method updates a montage file to class value
    #
    def update_montage(self, montage_file, from_nedc_eas = False):

        # update new montage file, if input montage file is None, update
        # with the default montage
        #
        if montage_file is None or montage_file == "None":
            self.montage_fname_d = nft.get_fullpath(DEFAULT_MONTAGE_FNAME)
        else:
            self.montage_fname_d = nft.get_fullpath(montage_file)

        
        if from_nedc_eas:
            
            # TODO: look into a way of how we don't hardcode the 0
            #
            line = f"{DELIM_LBL_SYMBOL}[0] = {str(parse_nedc_eas_map_to_montage_defintion(montage_file))}"
            line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                        .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)
            try:
                level, mapping = self.parse_symboldef(line)
                self.symbol_map_d[level] = mapping
            except:
                print("Error: %s (line %s) %s::%s: %s (%s)" %
                    (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                    ndt.__NAME__, "error parsing montage", line))
                return False
        
        else:
            montage_fp = open(nft.get_fullpath(self.montage_fname_d),
                            nft.MODE_READ_TEXT)
            # loop over lines in file
            #
            lines = montage_fp.readlines()

            for line in lines:
                # clean up the line
                #
                line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                        .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)

                # parse a single montage definition
                #
                if line.startswith(DELIM_LBL_MONTAGE):
                    try:
                        chan_num, name, montage_line = \
                            self.parse_montage(line)
                        self.chan_map_d[chan_num] = name
                        self.montage_lines_d.append(montage_line)
                    except:
                        print("Error: %s (line %s) %s::%s: %s (%s)" %
                            (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                            ndt.__NAME__, "error parsing montage", line))
                        montage_file.close()
                        return False

                # parse symbol definitions at a level
                #
                elif line.startswith(DELIM_LBL_SYMBOL):
                    try:
                        level, mapping = self.parse_symboldef(line)
                        self.symbol_map_d[level] = mapping
                    except:
                        print("Error: %s (line %s) %s::%s: %s (%s)" %
                            (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                            ndt.__NAME__, "error parsing symbols", line))
                        montage_fp.close()
                        return False
#
# end of class

# class: Csv
#
# This class contains methods to manipulate comma seperated value files.
#
class Csv:

    # method: Csv::constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self, montage_f = DEFAULT_MONTAGE_FNAME) -> None:

        Csv.__CLASS_NAME__ = self.__class__.__name__
        
        self.montage_f = montage_f
        self.channel_map_label = {DEF_CHANNEL:DEF_TERM_BASED_IDENTIFIER}
        self.file_duration = 0
        
        # parse the default montage file to ensure that
        # the channel dictionary isn't empty
        #
        self.parse_montage(nft.get_fullpath(self.montage_f))
        self.graph_d = AnnGrEeg()

    # method: Csv::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: loading annotation from file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        with open(fname, nft.MODE_READ_TEXT) as fp:

            for line in fp:

                # remove space, "\n" and "\r" just case in it is written on a 
                # a window operating machine
                #
                line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                           .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL) \
                           .replace(nft.DELIM_SPACE, nft.DELIM_NULL)
                
                # # get the montage_file only if the user didn't specify anything
                # # take the user's input as the priority montage file
                # #
                # if line.startswith(nft.DELIM_COMMENT + "montage") and self.montage_f == DEFAULT_MONTAGE_FNAME:
                #     self.montage_f = line.split(nft.DELIM_EQUAL)[int(-1)]
                    
                # parse montage file
                #
                # if not self.parse_montage(nft.get_fullpath(self.montage_f)):
                #     print("Error: %s (line: %s) %s::%s %s (%s)" %
                #         (__FILE__, ndt.__LINE__, Csv.__CLASS_NAME__,
                #         ndt.__NAME__, "unable to parse montage file", line))
                #     return False

                # get the duration
                #
                if line.startswith(nft.DELIM_COMMENT + "duration"):
                    self.file_duration = line.replace("secs", nft.DELIM_NULL) \
                                             .split(nft.DELIM_EQUAL)[int(-1)]

                # ignore comments, blank line, csv header
                #
                if line.startswith(nft.DELIM_COMMENT) or \
                    DEF_CSV_LABELS in line or \
                    len(line) == 0 :
                    continue
                
                # get the annotation label file for each line
                # 
                channel, start_time, stop_time, label, confidence = \
                    line.split(nft.DELIM_COMMA)
                
                # If the annotation is term base
                # then we should handle it 
                if channel == DEF_TERM_BASED_IDENTIFIER:
                    
                    # uses the index of -1 if it is a term based event
                    #
                    self.graph_d.create(int(0), int(0), int(-1),
                            float(start_time), float(stop_time),
                            {label:float(confidence)})

                else:

                    # get the correct index for the channel
                    #
                    for ind, channel_lb in self.channel_map_label.items():
                        if channel_lb == channel:
                            channel_ind = ind

                    self.graph_d.create(int(0), int(0), channel_ind,
                                float(start_time), float(stop_time), 
                                {label:float(confidence)})

        self.graph_d.sort()

        # exit gracefully 
        #
        return True
    
    # method: Csv::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method writes the events to a .csv file
    #
    def write(self, ofile, level, sublevel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: writing events to .csv file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # sort the graph 
        # just in case it was set without using the load method
        #
        self.graph_d.sort()

        # get the graph
        #
        graph = self.get_graph()

        # try to access the graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__,
                   ndt.__NAME__, "level/sublevel not in graph",
                   level, sublevel))
            return False
        
        # make the directory if a path is passed
        #
        if len(ofile.split(nft.DELIM_SLASH)) > 1:
            os.makedirs(os.path.dirname(ofile), exist_ok=True)

        # open file to write
        #
        with open(ofile, nft.MODE_WRITE_TEXT, newline="\n") as fp:

            # write version
            #
            fp.write(f"# version = {FTYPES['csv'][0]}\n")

            # write the bname
            #
            fp.write("# bname = %s\n" %
                     os.path.splitext(os.path.basename(ofile))[0])

            # write the duration
            #
            fp.write(f"# duration = {self.file_duration} secs\n")
            
            # write the montage file
            #
            fp.write(f"# montage_file = {self.montage_f}\n")
            fp.write(nft.DELIM_COMMENT)
            fp.write(nft.DELIM_NEWLINE)

            # write the csv header
            #
            fp.write(f"{DEF_CSV_LABELS}\n")
            
            # add all the event from the graphing object
            #
            for channel_ind, events in graph[level][sublevel].items():
                
                for event in events:
                    start_time, stop_time = event[0], event[1]
                    
                    # takes the form {'label':confidence}
                    # then unzip it then cast it to the variable where it gets
                    # unzip again as the first unzip makes it turn into a tuple
                    #
                    [*label], [*confidence] = zip(*event[-1].items())

                    fp.write(f"{self.channel_map_label[channel_ind]},"
                             f"{start_time:.{PRECISION}f},"
                             f"{stop_time:.{PRECISION}f},"
                             f"{label[0]},"
                             f"{confidence[0]:.{PRECISION}f}\n")
                    

        # exit gracefully
        #
        return True
    
    # method: Csv::display
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a boolean value indicating status
    #
    # This method displays the events from a flat AG.
    #
    def display(self, level, sublevel, fp=sys.stdout):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: displaying events from flag AG" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__, ndt.__NAME__,
                   "level/sublev not in graph", level, sublevel))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in graph[level][sublevel]:

            # iterate over events for each channel
            #
            for event in graph[level][sublevel][chan]:

                # find max probability
                #
                max_prob = max(event[2].values())

                # iterate over symbols in dictionary
                #
                for symb in event[2]:

                    # if the value of the symb equals the max prob
                    #
                    if event[2][symb] == max_prob:

                        # set max symb to this symbol
                        #
                        max_symb = symb
                        break

                # display event
                #
                if max_prob is not None:
                    fp.write(f"{self.chan_map_d[chan]:>10}: \
                            {event[0]:10.{PRECISION}f} \
                            {event[1]:10.{PRECISION}f} \
                            {max_symb:>8} \
                            {max_prob:10.{PRECISION}f}\n")
                else:
                    fp.write(f"{self.chan_map_d[chan]:>10}: \
                                {event[0]:10.{PRECISION}f} \
                                {event[1]:10.{PRECISION}f} \
                                {max_symb:>8}\n")
        # exit gracefully
        #
        return True
    #
    # end of method

    # method:: Csv::validate
    #
    # arguments:
    #  fname: the file name
    #
    # return: a boolean value indicating status
    #
    # This method returns True if the metadata is a valid csv header.
    #
    def validate(self, fname):

        # display debugging information
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: checking for csv (%s)" %
                  (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                   ndt.__NAME__, fname))

        # open the file
        #
        fp = open(fname, nft.MODE_READ_TEXT)
        if fp is None:
            print("Error: %s (line: %s) %s::%s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                   ndt.__NAME__, fname))
            return False

        # read the first line in the file
        #
        header = fp.readline()
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: header (%s)" %
                  (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                   ndt.__NAME__, header))
        fp.close()

        # exit gracefully:
        #
        if DEF_CSV_HEADER in header.strip():
            return True
        else:
            if dbgl > ndt.BRIEF:
                print("Error: %s (line: %s) %s::%s: Not a valid CSV (%s)" %
                      (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                       ndt.__NAME__, fname))
            return False
    #
    # end of method
    
    # method: Csv::parse_montage
    #
    # argument:
    #  montage_f: a montage file 
    #
    # return: a boolean value indicating status
    #
    # This method updates the channel_map_label variable based on the inputted
    # montage file
    #
    # Note:
    #  Please expand any environment variable before passing it into the function
    #  since the function does not expand it for you.
    #
    def parse_montage(self, montage_f):
        
        montage_fp = open(montage_f, nft.MODE_READ_TEXT)
        if montage_fp is None:
            print("Error: %s (line: %s) %s::%s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, Csv.__CLASS_NAME__,
                   ndt.__NAME__, montage_f))
            return False
        
        # check if the dictionary has been populated once 
        # this will be true when the dictionary is not-empty 
        #
        if len(self.channel_map_label) > 1:
            self.channel_map_label.clear()

        self.channel_map_label = {DEF_CHANNEL:DEF_TERM_BASED_IDENTIFIER}

        for line in montage_fp:

            line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                        .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL) \
                        .replace(nft.DELIM_SPACE, nft.DELIM_NULL)
            
            # ignore the montage header and header 
            # Ex: [Montage] 
            #
            if line.startswith(nft.DELIM_OPEN) or line.startswith(nft.DELIM_COMMENT) \
                or len(line) == 0:
                continue
            
            # extract the information
            # 
            channel_number, channel_name = re.findall(DEF_REGEX_MONTAGE_FILE, line).pop()

            # append to the channel_map dictionary to create
            # the corresponding hannel number and name
            #
            self.channel_map_label[int(channel_number)] = channel_name
        
        # exit gracefully 
        #
        return True
    # 
    # end of method

    # method: Csv::set_file_duration
    #
    # arguments: 
    #   dur: duration of the file
    #
    # return: None
    # 
    # This method allows us to set the file duration for the whole
    # csv file
    #
    def set_file_duration(self,dur):
        self.file_duration = dur
        return

    # method: Csv::get_file_duration
    #
    # arguments: none
    #
    # return:
    #  duration: the file whole file duration (float)
    # 
    # This method returns the file duration for the whole
    # csv file
    #
    def get_file_duration(self):
        return float(self.file_duration)
 
    # method: Csv::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur, sym, level, sublevel):
        return self.graph_d.add(dur, sym, level, sublevel)
    #
    # end of method
    
    # method: Csv::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym.
    #
    def delete(self,sym, level, sublevel):
        return self.graph_d.delete(sym, level, sublevel)
    #
    # end of method

    # method: Csv::get
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  channel: the channel 
    #
    # return: 
    #
    #
    def get(self, level, sublevel, channel):
        events = self.graph_d.get(level, sublevel, channel)
        return events
    #
    # end of method

    # method: Csv::get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()
    #
    # end of method

    # method: Csv::set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph.
    #
    def set_graph(self, graph):
        return self.graph_d.set_graph(graph)
    #
    # end of method

# class: Xml
#
class Xml:

    # method: Xml::constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self, montage_f = DEFAULT_MONTAGE_FNAME, schema = DEFAULT_XML_SCHEMA_FILE) -> None:

        # set the class name
        #
        Xml.__CLASS_NAME__ = self.__class__.__name__ 
        
        self.montage_f = montage_f
        self.schema = schema
        self.channel_map_label = {DEF_CHANNEL:DEF_TERM_BASED_IDENTIFIER}
        self.file_duration = 0

        # parse the default montage file to ensure that
        # the channel dictionary isn't empty
        #
        self.parse_montage(nft.get_fullpath(self.montage_f))
        
        self.graph_d = AnnGrEeg()
    
    # method: Xml::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname):
        
        status = self.validate(fname)
        if not status:
            print("Error: %s (line: %s) %s: invalid xml file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False
        else:
            
            root = ET.parse(fname).getroot()

            xml_dict = self.tree_to_dict(root)

            # note: This XML format assumes that we are doing annotations on 
            #       0,0 level only. Chances are, we would need to revisit the 
            #       XML if we want it to support multi-level annotations.
            #
            graph = {0: {0: dict(xml_dict)}}

            # set the graphing object to be the newly parsed XML 
            #
            self.graph_d.graph_d = graph
        
        # exit gracefully
        #
        return True

    # method: Xml::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method writes the events to a .xml file
    #
    def write(self, ofile, level, sublevel):
        
        # sort the graph
        #
        self.graph_d.sort()

        # get graph
        #
        graph = self.get_graph()

        # TODO: Need to look into TERM BASED ANNOTATIONS XML format

        # local variables
        #
        file_start_time, file_end_time = float("inf"), float("-inf")
        channels = set()

        # get the durations, end points and channels
        #
        for channel_index, data in graph[0][0].items():

            channels.add(self.channel_map_label[channel_index])

            for start, stop, _ in data:
                file_start_time = min(file_start_time, start)
                file_end_time = max(file_end_time, stop)

        # set up the root
        #
        root = ET.Element(DEF_XML_ROOT)

        # add the bname
        #
        bname = ET.SubElement(root, DEF_XML_BNAME)
        bname.text = Path(ofile).stem

        # add the duration
        #
        duration = ET.SubElement(root, DEF_XML_DURATION)
        duration.text = f"{self.file_duration} secs"

        # add the montage file
        #
        montage_file = ET.SubElement(root, DEF_XML_MONTAGE_FILE)
        montage_file.text = self.montage_f

        # set up the label
        #
        label = ET.SubElement(root, DEF_XML_LABEL, name= Path(ofile).stem, dtype=PARENT_TYPE)

        # add the endpoints 
        #
        endpoints = ET.SubElement(label, DEF_XML_ENDPOINTS, name=DEF_XML_ENDPOINTS, dtype= LIST_TYPE)
        endpoints.text = f"[{file_start_time:.{PRECISION}f}, {file_end_time:.{PRECISION}f}]"
        
        # add the montage_channels
        #
        montage_channels = ET.SubElement(label, DEF_XML_MONTAGE_CHANNELS, name=DEF_XML_MONTAGE_CHANNELS, dtype=PARENT_TYPE)
        
        # add all the channels to the xml
        #
        for channel in channels:
            montage_channels.append(ET.Element(DEF_XML_CHANNEL, name=channel, dtype="*"))

        # writes the start time and end time of each event under the correct channels
        #
        for channel_index, data in graph[0][0].items():

            parent_channel = label.find('montage_channels/channel[@name=\'%s\']' % (self.chan_map_d[channel_index]))

            for start, stop, tag_probability in data:

                event_tag, event_probability = next(iter(tag_probability.items()))

                tag = ET.SubElement(parent_channel, DEF_XML_EVENT, name=str(event_tag), dtype=PARENT_TYPE)

                endpoint = ET.SubElement(tag, DEF_XML_ENDPOINTS, name=DEF_XML_ENDPOINTS, dtype= LIST_TYPE)
                endpoint.text = f"[{start:.{PRECISION}f}, {stop:.{PRECISION}f}]"

                probability = ET.SubElement(tag, DEF_XML_PROBABILITY, name=DEF_XML_PROBABILITY, dtype= LIST_TYPE)
                probability.text = f"[{float(event_probability):.{PRECISION}f}]"
     
        # convert the tree to a string
        #
        xmlstr = ET.tostring(root, encoding=nft.DEF_CHAR_ENCODING)

        # convert the string to a pretty print
        #
        reparsed = minidom.parseString(
            xmlstr).toprettyxml(indent=nft.DELIM_SPACE)
        
        # open the output file to write
        #
        with open(ofile, nft.MODE_WRITE_TEXT) as writer:

            # write the xml file
            #
            writer.write(reparsed)
        
        # exit gracefully
        #
        return True
    
    # method: Xml::display
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a boolean value indicating status
    #
    # This method displays the events from a flat AG.
    #
    def display(self, level, sublevel, fp=sys.stdout):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: displaying events from flag AG" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Xml.__CLASS_NAME__, ndt.__NAME__,
                   "level/sublev not in graph", level, sublevel))
            return False

        for chan in graph[level][sublevel]:
            # iterate over events for each channel
            #
            for event in graph[level][sublevel][chan]:
                start = event[0]
                stop = event[1]

                # create a string with all symb/prob pairs
                #
                pstr = ""
                for symb in event[2]:
                    pstr += f" {symb:>8} {event[2][symb]:10.{PRECISION}f}"

                if chan != -1:
                    chan_a = chan
                else:
                    chan_a = -1
                # display event
                #
                fp.write(f"{self.chan_map_d[chan_a]:>10}: \
                            {start:10.{PRECISION}f} \
                            {stop:10.{PRECISION}f}{pstr}\n") 
        # exit gracefully
        #
        return True
    #
    # end of method
    
    # arguments:
    #  root (xml.etree.ElementTree.root): root of the xml file
    #
    # return:
    #  treedict: dictionary equivalent of xml tree
    # 
    # This method converts the given xml files into the graphing 
    # object format which allows us to set the graph directly using the return
    # dictionary
    #
    def tree_to_dict(self, root):
        
        # local variables
        #
        treedict = defaultdict(list)

        # access the duration
        #
        duration = root.find(DEF_XML_DURATION).text
        self.file_duration = duration.replace(" secs", nft.DELIM_NULL)

        # access all the channel
        #
        for montage_channel in root.findall(DEF_XML_CHANNEL_PATH):
            
            # set the channel num so that we can use it to index
            # our dictionary with the corresponding channel number
            #
            for num, channel in self.chan_map_d.items():
                if channel == montage_channel.attrib[DEF_XML_NAME]:
                    channel_num = num

            # iterate through all the event in that channel 
            #
            for event in montage_channel.findall(DEF_XML_EVENT):
                tag = event.attrib[DEF_XML_NAME]
                probability = event.find(DEF_XML_PROBABILITY).text.strip(nft.DELIM_OPEN) \
                                                                  .strip(nft.DELIM_CLOSE)
                start_time, end_time = \
                    event.find(DEF_XML_ENDPOINTS).text.strip(nft.DELIM_OPEN) \
                                                      .strip(nft.DELIM_CLOSE) \
                                                      .strip() \
                                                      .split(",")
                # append to the correct channel index
                #
                treedict[channel_num].append([float(start_time), float(end_time), {tag: float(probability)}])

        # exit gracefully
        #
        return treedict

    # method: Xml::validate
    #
    # arguments:
    #  fname: filename to be validated
    #
    # return: a boolean value indicating status
    #
    # This method validates xml file with a schema
    #
    def validate(self, fname, xml_schema = DEFAULT_XML_SCHEMA_FILE):

        # parse an XML file
        #
        try:
            # turn a file to XML Schema validator
            #
            self.schema = etree.XMLSchema(file=nft.get_fullpath(xml_schema))
            xml_file = etree.parse(fname)

        # check for a syntax error
        #
        except etree.XMLSyntaxError:
            if dbgl > ndt.NONE:
                print("Error: %s (line: %s) %s: xml syntax error (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False

        # check if there was an OSerror (e.g,, file doesn't exist)
        #
        except OSError:
            if dbgl > ndt.NONE:
                print("Error: %s (line: %s) %s: xml file doesn't exist (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False

        # validate the schema
        #
        status = self.schema.validate(xml_file)
        if status == False:
            try:
                self.schema.assertValid(xml_file)
            except etree.DocumentInvalid as errors:
                print("Error: %s (line: %s) %s: %s (%s)" %
                      (__FILE__, ndt.__LINE__, ndt.__NAME__, errors, fname))

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: Xml::parse_montage
    #
    # argument:
    #  montage_f: a montage file 
    #
    # return: a boolean value indicating status
    #
    # This method updates the channel_map_label variable based on the inputted
    # montage file
    #
    # Note:
    #  Please expand any environment variable before passing it into the function
    #  since the function does not expand it for you.
    #
    def parse_montage(self, montage_f):
        
        montage_fp = open(montage_f, nft.MODE_READ_TEXT)
        if montage_fp is None:
            print("Error: %s (line: %s) %s::%s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, Csv.__CLASS_NAME__,
                   ndt.__NAME__, montage_f))
            return False
        
        # check if the dictionary has been populated once 
        # this will be true when the dictionary is not-empty 
        #
        if len(self.channel_map_label) > 1:
            self.channel_map_label.clear()

        for line in montage_fp:

            line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                        .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL) \
                        .replace(nft.DELIM_SPACE, nft.DELIM_NULL)
            
            # ignore the montage header and header 
            # Ex: [Montage] 
            #
            if line.startswith(nft.DELIM_OPEN) or line.startswith(nft.DELIM_COMMENT) \
                or len(line) == 0:
                continue
            
            # extract the information
            # 
            channel_number, channel_name = re.findall(DEF_REGEX_MONTAGE_FILE, line).pop()

            # append to the channel_map dictionary to create
            # the corresponding hannel number and name
            #
            self.channel_map_label[int(channel_number)] = channel_name
        
        # exit gracefully 
        #
        return True
    # 
    # end of method
    
    # method: Xml::set_file_duration
    #
    # arguments: 
    #   dur: duration of the file
    #
    # return: None
    # 
    # This method allows us to set the file duration for the whole
    # xml file
    #
    def set_file_duration(self,dur):
        self.file_duration = dur
        return

    # method: Xml::get_file_duration
    #
    # arguments: 
    #
    # return: 
    #  duration: the file duration (float)
    # 
    # This method returns the file duration for the whole
    # xml file
    #
    def get_file_duration(self):
        return float(self.file_duration)

    # method: Xml::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur, sym, level, sublevel):

        # check that this is a valid add, by checking if
        # the sym specified is listed in the class mapping
        #
        class_mapping = self.get_valid_sub(self.schema_fname_d)

        # initialize variables to hold child and parent values
        #
        child = None
        parent = None
        Status = False

        # traverse dictionary holding possible sym parent keys and
        # child values
        #
        for key in class_mapping:

            # if the sym specified is found to be a key
            #
            if (sym == key):

                # store parent value
                #
                parent = key
                Status = True
                break

            # get list of children for each parent sym
            #
            value_list = class_mapping.get(key)

            # iterate through list of children
            #
            for value in value_list:

                # if sym is found in value list
                #
                if (sym == value):

                    # store child value
                    #
                    status = True
                    child = value
                    parent = key
                    break

        # if label is not valid
        #
        if status is False:
            print("Error: %s (line: %s) %s: invalid label (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, sym))

        # if the sym to add is a child we make sure that the parent
        # label already exists in the duration provided
        #
        if (child != None):

            graph = self.get_graph()

            # obtain parent dictionary which is one sublevel below child
            #
            chan_dict = graph[level][sublevel-1]

            # check if parent sym exists at duration
            #
            add_parent = True

            for key in chan_dict:

                # list of events
                #
                value_list = chan_dict.get(key)

                # for each event
                # example of event: [0.0, 10.2787, {'bckg': 1.0}]
                #
                for event in value_list:

                    # get label of event for example ['bckg'] or ['seiz']
                    #
                    event_key = list(event[2].keys())

                    # if the parent event exists at duration
                    #
                    if ((event[1] == dur) and (parent == event_key[0])):

                        Add_parent = False

            # if parent label was not found at duration
            #
            if (add_parent == True):

                # add parent label
                #
                self.graph_d.add(dur, parent, level, sublevel - 1)

        return self.graph_d.add(dur, sym, level, sublevel)

    #
    # end of method
        
    
    # method: Xml::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym.
    #
    def delete(self, sym, level, sublevel):
        return self.graph_d.delete(sym, level, sublevel)
    #
    # end of method

    # method: Xml::get
    #
    # arguments:
    #  level: level of annotations to get
    #  sublevel: sublevel of annotations to get
    #
    # return: events at level/sublevel by channel
    #
    # This method gets the annotations stored in the AG at level/sublevel.
    #
    def get(self, level, sublevel, channel):
        events = self.graph_d.get(level, sublevel, channel)
        return events
    #
    # end of method

    # method: Xml::get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()
    # 
    # end of method

    # method: Xml::set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph
    #
    def set_graph(self, graph):
        return self.graph_d.set_graph(graph)
    #
    # end of method
#
# end of class

# class: Ann
#
# This class is the main class of this file. It contains methods to
# manipulate the set of supported annotation file formats including
# label (.lbl) and time-synchronous events (.tse) formats.
#
class AnnEeg:

    # method: AnnEeg::constructor
    #
    # arguments: none
    #
    # return: none
    #
    # This method constructs AnnEeg
    #
    def __init__(self):

        # set the class name
        #
        AnnEeg.__CLASS_NAME__ = self.__class__.__name__

        # declare variables for each type of file:
        #  these variable names must match the FTYPES declaration.
        #
        self.tse_d = Tse()
        self.lbl_d = Lbl()
        self.csv_d = Csv()
        self.xml_d = Xml()

        # declare variable to store type of annotations
        #
        self.type_d = None
    #
    # end of method

    # method: AnnEeg::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname, schema=DEFAULT_XML_SCHEMA_FILE, montage_f=DEFAULT_MONTAGE_FNAME):
        
        # reinstantiate objects, this removes the previous loaded annotations
        #
        self.lbl_d = Lbl()
        self.tse_d = Tse()
        self.csv_d = Csv(montage_f)
        self.xml_d = Xml(montage_f, schema)

        # determine the file type
        #
        magic_str = nft.get_version(fname)
    
        self.type_d = self.check_version(magic_str)

        if self.type_d == None or self.type_d == False:
            print("Error: %s (line: %s) %s: unknown file type (%s: %s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname, magic_str))
            return False

        # load the specific type
        #
        return getattr(self, FTYPES[self.type_d][1]).load(fname)

    #
    # end of method

    # method: AnnEeg::get
    #
    # arguments:
    #  level: the level value
    #  sublevel: the sublevel value
    #
    # return:
    #  events: a list of ntuples containing the start time, stop time,
    #          a label and a probability.
    #
    # This method returns a flat data structure containing a list of events.
    #
    def get(self, level=int(0), sublevel=int(0), channel=int(-1)):

        if self.type_d is not None:
    
            events = getattr(self,
                             FTYPES[self.type_d][1]).get(level, sublevel,
                                                         channel)
        else:
            print("Error: %s (line: %s) %s: no annotation loaded" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            return False
        
        # exit gracefully
        #
        return events
    #
    # end of method

    # method: AnnEeg::display
    #
    # arguments:
    #  level: level value
    #  sublevel: sublevel value
    #  fp: a file pointer (default = stdout)
    #
    # return: a boolean value indicating status
    #
    # This method displays the events at level/sublevel.
    #
    def display(self, level=int(0), sublevel=int(0), fp=sys.stdout):

        if self.type_d is not None:

            # display events at level/sublevel
            #
            status = getattr(self,
                             FTYPES[self.type_d][1]).display(level,
                                                             sublevel, fp)

        else:
            sys.stdout.write("Error: %s (line: %s) %s %s" %
                             (ndt.__NAME__, ndt.__LINE__, ndt.__NAME__,
                              "no annotations to display"))
            return False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of annotation to write
    #  sublevel: sublevel of annotation to write
    #
    # return: a boolean value indicating status
    #
    # This method writes annotations to a specified file.
    #
    def write(self, ofile, level=int(0), sublevel=int(0)):

        # write events at level/sublevel
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[self.type_d][1]).write(ofile,
                                                                 level,
                                                                 sublevel)
        else:
            sys.stdout.write("Error: %s (line: %s) %s: %s" %
                             (__FILE__, ndt.__LINE__, ndt.__NAME__,
                              "no annotations to write"))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::add
    #
    # arguments:
    #  dur: duration of file
    #  sym: symbol of event to be added
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events to the current events based on args.
    #
    def add(self, dur, sym, level, sublevel):

        # add labels to events at level/sublevel
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[self.type_d][1]).add(dur,
                                                               sym,
                                                               level,
                                                               sublevel,)
        else:
            print("Error: %s (line: %s) %s: no annotations to add to" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::delete
    #
    # arguments:
    #  sym: symbol of event to be deleted
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes all events of type sym
    #
    def delete(self, sym, level, sublevel):

        # delete labels from events at level/sublevel
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[self.type_d][1]).delete(sym,
                                                                  level,
                                                                  sublevel)
        else:
            print("Error: %s (line: %s) %s: no annotations to delete" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::validate
    #
    # arguments:
    #  type: the type of ann object to set
    #
    # return: a boolean value indicating status
    #
    # This method validate the file
    #
    def validate(self, fname, xml_schema = DEFAULT_XML_SCHEMA_FILE):

        # validate file
        #
        status = self.csv_d.validate(fname) or \
                 self.xml_d.validate(fname, xml_schema)

        if not status:
            if dbgl > ndt.BRIEF:
                print("Error: %s (line: %s) %s: cannot validate file type" %
                        (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::set_type
    #
    # arguments:
    #  type: the type of ann object to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the type and graph in type from self.type_d
    #
    def set_type(self, ann_type):

        # set the graph of ann_type to the graph of self.type_d
        #
        if self.type_d is not None:
            status = getattr(self,
                             FTYPES[ann_type][1]).set_graph(
                                 getattr(self,
                                         FTYPES[self.type_d][1]).get_graph())
            self.type_d = ann_type

        else:
            print("Error: %s (line: %s) %s: no graph to set" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::set_graph
    #
    # arguments:
    #  type: type of ann object to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the type and graph in type from self.type_d
    #
    def set_graph(self, graph):
        status = getattr(self, FTYPES[self.type_d][1]).set_graph(graph)
        return status

    # method: AnnEeg::set_file_duration
    #
    # arguments: 
    #   dur: duration of the file
    #
    # return: None
    # 
    # This method allows us to set the file duration for csv and xml
    # files. TSE and LBL files does not have durations within them. 
    #
    def set_file_duration(self,dur):

        # display debugging information
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: setting file durations (%s)" %
                  (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                   ndt.__NAME__, dur))
        
        self.csv_d.set_file_duration(dur)
        self.xml_d.set_file_duration(dur)
        
        return

    # method: AnnEeg::get_file_duration
    #
    # arguments: 
    #
    # return: None
    # 
    # This method returns the file duration for csv and xml files. 
    # TSE and LBL files does not have durations within them. 
    #
    def get_file_duration(self):

        # display debugging information
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: getting file durations" %
                  (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                   ndt.__NAME__))
        
        if self.type_d == "csv" or self.tse_d == "xml":

            duration = getattr(self, FTYPES[self.type_d][1]).get_file_duration()
       
        else:
            print("Error: %s (line: %s) %s: invalid type" %
        (__FILE__, ndt.__LINE__, ndt.__NAME__))

        return duration


    # method: Anneeg:: delete_graph
    #
    #
    def delete_graph(self):
        getattr(self, FTYPES[self.type_d][1]).graph_d.delete_graph()
        return True

    # method: AnnEeg::get_graph
    #
    # arguments: none
    #
    # return: the entire annotation graph
    #
    # This method returns the entire stored annotation graph
    #
    def get_graph(self):

        # if the graph is valid, get it
        #
        if self.type_d is not None:
            graph = getattr(self, FTYPES[self.type_d][1]).get_graph()
        else:
            print("Error: %s (line: %s) %s: no graph to get" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            graph = None

        # exit gracefully
        #
        return graph
    #
    # end of method

    # method: AnnEeg::check_version
    #
    # arguments:
    #  magic: a magic sequence
    #
    # return: a character string containing the name of the type
    #
    def check_version(self, magic):

        # check for a match
        #
        for key in FTYPES:
            if FTYPES[key][0] == magic:
                return key

        # exit (un)gracefully:
        #  if we get this far, there was no match
        #
        return False
    #
    # end of method



#
# end of class

#
# end of file
