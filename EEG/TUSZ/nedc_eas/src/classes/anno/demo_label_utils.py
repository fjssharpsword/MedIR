#!/usr/bin/env python

# file: $nedc_nfc/class/python/nedc_label_utils/nedc_label_utils.py
#
# revision history:
#  20161020 (JM): initial version
#
# usage:
#  import nedc_label_utils as nlu
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
# ------------------------------------------------------------------------------

# import required modules
#
import pprint
import sys
import os
from operator import itemgetter

# import nedc modules
#
import nedc_file_tools as nft
import nedc_ann_eeg_tools as nae

# ------------------------------------------------------------------------------
#
# globals are listed here
#
# ------------------------------------------------------------------------------

# level and sublevel to read segments from lbl file and characters to be
# stripped from lines
#
NEDC_LBL_READ_LEVEL = 0
NEDC_LBL_READ_SUB = 1
NEDC_LBL_STRIP = ['{', '}', '[', ']', ';']
NEDC_LBL_ALL_CHAN = "all"

# defaults for arguments
#
NEDC_DEF_CFMT = "_ch%0.3d"
NEDC_DEF_EPOCH_WDUR = 1
NEDC_DEF_EPOCH_FDUR = 1
NEDC_DEF_FDUR = 0.1
NEDC_DEF_NUM_CH = 22

# define the default event labels for the lbl files
#
NEDC_LBL_EVENTS = ['bckg', 'pled', 'gped', 'eyebl', 'artf', 'bckg', 'seiz']
NEDC_BCKG_EVENT_LABEL = "BCKG"

NEDC_CSV_LABELS = "channel,start_time,stop_time,label,confidence"

# extensions for label files
#
NEDC_CSV_EXT = "csv"
NEDC_LAB_EXT = "lab"
NEDC_OVR_LAB_EXT = "lab_ov"
NEDC_LBL_EXT = "lbl"
NEDC_LST_EXT = "list"
NEDC_XML_EXT = "xml"
NEDC_TSE_EXT = "tse"

NEDC_EXT_LIST = [NEDC_CSV_EXT, NEDC_XML_EXT, NEDC_LBL_EXT, NEDC_TSE_EXT]

# define symbols that appear as keys in an lbl file
#
DELIM_LBL_MONTAGE = 'montage'
DELIM_LBL_NUM_LEVELS = 'number_of_levels'
DELIM_LBL_LEVEL = 'level'
DELIM_LBL_SYMBOL = 'symbols'
DELIM_LBL_LABEL = 'label'

# queries to be replaced in the template lbl file
#
NEDC_LBL_REPL_MAP = "$LEVEL_MAP"
NEDC_LBL_REPL_FNAME = "$NAME"

# items per label file type
#
NEDC_REC_SEG_LEN = 4
NEDC_LAB_SEG_LEN = 3

# HTK constants
#
HTK_FDUR = 0.001
HTK_TIME_SCALE = 1.0 / 100e-9

# null label variables
#
NULL_EVENT_LABEL = "null"
NULL_EVENT_ID = 0
NULL_EVENT_PRIORITY = 0
NULL_SEGMENT = [0, 1, NULL_EVENT_ID]

# character and string constants
#
DOT = "."
NULL = ""
NEDC_DSK_RAID = "/dsk0_raid10"

# ------------------------------------------------------------------------------
#
# classes are listed here
#
# ------------------------------------------------------------------------------

# class: LabelConverterUtils
#
# This class contains methods to read multiple types of label files and covert
# them.
#


class LabelConverterUtils():

    # method: Constructor
    #
    # arguments:
    #  fdur: frame duration in seconds
    #  epoch_fdur: overlap between epochs in seconds
    #  epoch_wdur: the amount of time in epoch is defined in seconds
    #  num_ch: number of expected channels in the corresponding EEGs
    #  cfmt: channel modifier
    #
    # return: none
    #
    # The constructor initializes key values for the class. If no values are
    # passed, defaults are used
    #
    # ELK: changed arguments for compatability with demo. Should be
    # fairly easy to allow both versions.
    #
    def __init__(self,
                 dict_labels_priority_a,
                 dict_event_map_a,
                 dict_priority_map_a,
                 ann_map_file_a,
                 xml_schema,
                 fdur_a=None, epoch_fdur_a=None,
                 epoch_wdur_a=None, num_ch_a=None, cfmt_a=None):

        # initialize class variables
        #  if no arguments are passed to the constructor, use defaults
        #
        if fdur_a is None:
            self.fdur = NEDC_DEF_FDUR
        else:
            self.fdur = fdur_a

        if epoch_fdur_a is None:
            self.epoch_fdur = NEDC_DEF_EPOCH_FDUR
        else:
            self.epoch_fdur = epoch_fdur_a

        if epoch_wdur_a is None:
            self.epoch_wdur = NEDC_DEF_EPOCH_WDUR
        else:
            self.epoch_wdur = epoch_wdur_a

        if num_ch_a is None:
            self.num_ch = NEDC_DEF_NUM_CH
        else:
            self.num_ch = num_ch_a

        if cfmt_a is None:
            self.cfmt = NEDC_DEF_CFMT
        else:
            self.cfmt = cfmt_a

        self.bckg_id = 0

        # ELK: call self.set_demo_config instead of self.nedc_read_param
        #

        # use the demo's dictionaries to set parameters
        #
        self.set_demo_config(dict_labels_priority_a,
                             dict_event_map_a,
                             dict_priority_map_a)

        # # read the parameter file
        # #
        # self.nedc_read_param(cmap_dict_a)

        # get channel map using nedc_ann_eeg_tools
        #
        self.ann = nae.AnnEeg()
        self.chan_map_d = self.ann.csv_d.channel_map_label

        # get symbol map using lbl class
        #
        self.ann.lbl_d.update_montage(ann_map_file_a, from_nedc_eas = True)
        self.symbol_map_d = self.ann.lbl_d.symbol_map_d
        self.chan_lbl_map_d = self.ann.lbl_d.chan_map_d
        self.montage_lines_d = []
        self.num_levels_d = int(1)
        self.num_sublevels_d = {int(0): int(1)}
        self.ann.xml_d.schema_fname_d = xml_schema

    #
    # end of constructor

    # --------------------------------------------------------------------------
    #
    # methods for reading label files
    #
    # --------------------------------------------------------------------------

    # method: nedc_read_labels
    #
    # arguments:
    #  fname: the name of the input file to be read
    #
    # return: the segments read from the file
    #
    # This method takes in an input file name and uses the informational
    # methods to check its type. It then reads the segments accordingly
    #
    def nedc_read_labels(self, fname_a):

        # check the type of the file and read the segments accordingly
        #
        if self.nedc_is_csv(fname_a):
            segments = self.nedc_read_csv(fname_a)
        elif self.nedc_is_xml(fname_a):
            segments = self.nedc_read_xml(fname_a)
        elif self.nedc_is_tse(fname_a):
            segments = self.nedc_read_tse(fname_a)
        elif self.nedc_is_lbl(fname_a):
            segments = self.nedc_read_lbl(fname_a)

        # if the file type is not supported, print an error message and exit
        #
        else:
            print("LabelConverterUtils.nedc_read_labels(): %s %s"
                  % ("the file type is not supported for file", fname_a))
            return False

        # exit gracefully
        #
        return segments
    #
    # end of method

    # method: nedc_read_csv
    #
    # arguments:
    #  fname_a: filename of the csv file
    #
    # return:
    #  segments: the labels of the file in a general form
    #
    # This method takes an input .csv file and segments each channel of the
    # .csv file into an array. This is done by segmenting the .csv file line by
    # line.
    #
    def nedc_read_csv(self, fname_a):

        # declare local variables
        #
        segments = {}

        # calculate the frame rate ratio to get the segments in a general form
        #
        fr_ratio = HTK_FDUR / self.fdur

        # open the file for reading
        #
        # try:
        #     lines = [line.strip() for line in open(fname_a)]
        # except:
        #     print("LabelConverterUtils.nedc_read_rec(): %s %s."
        #           % ("Error opening rec file", fname_a))
        #     return False

        # open file
        #
        with open(fname_a, 'r') as fp:

            segments = {}

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
                   len(check) == 0 or \
                   NEDC_CSV_LABELS in line:
                    continue

                # split the line
                #
                parts = [part.strip() for part in line.split(nft.DELIM_COMMA)]

                try:
                    # loop over every part, starting after start/stop times
                    #
                    for i in range(3, len(parts), 3):

                        # create dict with label as key, prob as value
                        #
                        label = next(key for key, value
                                     in self.symbol_map_d[0].items()
                                     if value == parts[i])

                    # get chan idx
                    #
                    chan = parts[0].replace(nft.DELIM_QUOTE, nft.DELIM_NULL)

                    # add annotation to all channels if there is "__ALL__"
                    # else find chan index before add to graph
                    #
                    if chan == "__ALL__":
                        chan = int(-1)
                    else:
                        chan = next(key for key, value
                                    in self.chan_map_d.items()
                                    if value == chan)

                    # append annotation to segments
                    #
                    if chan not in segments:
                        segments[chan] = []

                    segments[chan].append(
                        [float(parts[1]), float(parts[2]), label])
                except:
                    pass

        # if there is annotation in key __ALL__ in segments, append to all channels
        #
        if (-1 in segments):
            anns = segments[-1]

            for ch in range(self.num_ch):
                if ch not in segments:
                    segments[ch] = []
                for ann in anns:
                    segments[ch].append(ann)

            # delete key __ALL__ after
            #
            del segments[-1]

        # check if each channel has a signal
        #
        for ch in range(self.num_ch):

            # if the segments are missing a channel, add a null one
            #
            if ch not in segments:
                null_channel = {ch: [NULL_SEGMENT]}
                segments.update(null_channel)

        # sort the segments of each channel
        #
        for channel in segments:
            segments[channel] = sorted(segments[channel], key=itemgetter(0))

        # collapse the events in each channel to prevent overlaps of
        # the same event
        #
        tmp_segs = {}
        for channel in segments:
            seg_length = len(segments[channel])
            if channel not in tmp_segs:
                tmp_segs[channel] = []

            for seg_num in range(seg_length):
                current_seg = segments[channel][seg_num]

                if seg_num <= seg_length - 2:
                    next_seg = segments[channel][seg_num + 1]
                    if ((current_seg[1] >= next_seg[0])
                            and (current_seg[2] == next_seg[2])):
                        tmp_segs[channel].append([current_seg[0], next_seg[0],
                                                  current_seg[2]])
                    else:
                        tmp_segs[channel].append(current_seg)

                else:
                    tmp_segs[channel].append(current_seg)

        segments = tmp_segs

        # exit gracefully
        #
        return segments
    #
    # end of method

    # method: nedc_read_xml
    #
    # arguments:
    #  fname_a: filename of the csv file
    #
    # return:
    #  segments: the labels of the file in a general form
    #
    # This method takes an input .xml file and segments each channel of the
    # .xml file into an array. This is done by segmenting the .xml file line by
    # line.
    #
    def nedc_read_xml(self, fname_a):

        # declare local variables
        #
        segments = {}

        # calculate the frame rate ratio to get the segments in a general form
        #
        fr_ratio = HTK_FDUR / self.fdur

        try:

            if bool(self.ann.xml_d.graph_d.graph_d) == False:

                # load file if graph is empty
                #
                self.ann.xml_d.load(fname_a)

            graph = self.ann.xml_d.get_graph()

            segments = {}

            for sublev in graph[0]:
                for chan in graph[0][sublev]:

                    # append annotation to segments
                    #
                    if chan not in segments:
                        segments[chan] = []

                    for ann in graph[0][sublev][chan]:
                        start, stop = ann[0:2]

                        label = next(key for key, value
                                     in self.symbol_map_d[0].items()
                                     if value == list(ann[2].keys())[0])
                        segments[chan].append(
                            [float(start), float(stop), label])
        except:
            pass

        # sort the segments of each channel and find index of duplicate label
        #
        idx_to_remove = {}
        for channel in segments:
            segments[channel] = sorted(segments[channel], key=itemgetter(0))
            check_dup = [item[0:2] for item in segments[channel]]

            for idx, item in enumerate(check_dup):

                for idx2 in range(idx+1, len(check_dup)):

                    if (item == check_dup[idx2]):
                        if channel not in idx_to_remove:
                            idx_to_remove[channel] = []
                        if (idx2 > idx):
                            idx_to_remove[channel].append(idx)

        # remove them
        #
        for chan in idx_to_remove:
            idx_to_remove[chan] = sorted(idx_to_remove[chan], reverse=True)

            for idx in idx_to_remove[chan]:
                del segments[chan][idx]

        # collapse the events in each channel to prevent overlaps of
        # the same event
        #
        tmp_segs = {}
        for channel in segments:
            seg_length = len(segments[channel])
            if channel not in tmp_segs:
                tmp_segs[channel] = []

            for seg_num in range(seg_length):
                current_seg = segments[channel][seg_num]

                if seg_num <= seg_length - 2:
                    next_seg = segments[channel][seg_num + 1]
                    if ((current_seg[1] >= next_seg[0])
                            and (current_seg[2] == next_seg[2])):
                        tmp_segs[channel].append([current_seg[0], next_seg[0],
                                                  current_seg[2]])
                    else:
                        tmp_segs[channel].append(current_seg)

                else:
                    tmp_segs[channel].append(current_seg)

        segments = tmp_segs

        # exit gracefully
        #
        return segments
    #
    # end of method

    # method: nedc_read_lbl
    #
    # arguments:
    #  fname_a: filename of the lbl file
    #
    # return:
    #  segments: the labels of the file in a general form
    #
    # This method takes an input lbl file and outputs the segments in a
    # general form.
    #
    def nedc_read_tse(self, fname_a):

        # declare local variables
        #
        segments = {}

        # calculate the frame rate ratio to get the segments in a general form
        #
        fr_ratio = HTK_FDUR / self.fdur

        # open the file for reading
        #
        try:
            lines = [line.strip() for line in open(fname_a)]
        except:
            print("LabelConverterUtils.nedc_read_tse(): %s %s."
                  % ("Error opening tse file", fname_a))
            return False

        # open file
        #
        with open(fname_a, 'r') as fp:

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

                a = 0
                if a == 0:
                    # loop over every part, starting after start/stop times
                    #
                    for i in range(2, len(parts), 2):

                        # create dict with label as key, prob as value
                        #
                        label = next(key for key, value
                                     in self.symbol_map_d[0].items()
                                     if value == parts[i])

                    chan = int(-1)
                    if chan not in segments:
                        segments[chan] = []

                    segments[chan].append(
                        [float(parts[0]), float(parts[1]), label])

                else:
                    print("Error: invalid annotation")
                    return False

        # if there is annotation in key __ALL__ in segments, append to all channels
        #
        if (-1 in segments):
            anns = segments[-1]

            for ch in range(self.num_ch):
                if ch not in segments:
                    segments[ch] = []
                for ann in anns:
                    segments[ch].append(ann)

            # delete key __ALL__ after
            #
            del segments[-1]

        # check if each channel has a signal
        #
        for ch in range(self.num_ch):

            # if the segments are missing a channel, add a null one
            #
            if ch not in segments:
                null_channel = {ch: [NULL_SEGMENT]}
                segments.update(null_channel)

        # sort the segments of each channel
        #
        for channel in segments:
            segments[channel] = sorted(segments[channel], key=itemgetter(0))

        # collapse the events in each channel to prevent overlaps of
        # the same event
        #
        tmp_segs = {}
        for channel in segments:
            seg_length = len(segments[channel])
            if channel not in tmp_segs:
                tmp_segs[channel] = []

            for seg_num in range(seg_length):
                current_seg = segments[channel][seg_num]

                if seg_num <= seg_length - 2:
                    next_seg = segments[channel][seg_num + 1]
                    if ((current_seg[1] >= next_seg[0])
                            and (current_seg[2] == next_seg[2])):
                        tmp_segs[channel].append([current_seg[0], next_seg[0],
                                                  current_seg[2]])
                    else:
                        tmp_segs[channel].append(current_seg)

                else:
                    tmp_segs[channel].append(current_seg)

        segments = tmp_segs

        # exit gracefully
        #
        return segments
    # end of method

    # method: nedc_read_lbl
    #
    # arguments:
    #  fname_a: filename of the lbl file
    #
    # return:
    #  segments: the labels of the file in a general form
    #
    # This method takes an input lbl file and outputs the segments in a
    # general form.
    #

    def nedc_read_lbl(self, fname_a):

        # declare local variables
        #
        segments = {}

        # calculate the frame rate ratio to get the segments in a general form
        #
        fr_ratio = HTK_FDUR / self.fdur

        # open the file for reading
        #
        try:
            lines = [line.strip() for line in open(fname_a)]
        except:
            print("LabelConverterUtils.nedc_read_lbl(): %s %s."
                  % ("Error opening lbl file", fname_a))
            return False

        # loop over lines in file
        #
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
                        self.ann.lbl_d.parse_montage(line)
                    self.chan_lbl_map_d[chan_num] = name
                    self.montage_lines_d.append(montage_line)
                except:
                    print("LabelConverterUtils.nedc_read_lbl(): %s %s."
                          % ("Error parsing montage", fname_a))
                    return False

            # parse the number of levels
            #
            elif line.startswith(DELIM_LBL_NUM_LEVELS):
                try:
                    self.num_levels_d = self.ann.lbl_d.parse_numlevels(line)
                except:
                    print("LabelConverterUtils.nedc_read_lbl(): %s %s."
                          % ("Error parsing number of levels", fname_a))
                    return False

            # parse the number of sublevels at a level
            #
            elif line.startswith(DELIM_LBL_LEVEL):
                try:
                    level, sublevels = self.ann.lbl_d.parse_numsublevels(line)
                    self.num_sublevels_d[level] = sublevels

                except:
                    print("LabelConverterUtils.nedc_read_lbl(): %s %s."
                          % ("Error parsing number of sublevels", fname_a))
                    return False

            # parse symbol definitions at a level
            #
            elif line.startswith(DELIM_LBL_SYMBOL):
                try:
                    level, mapping = self.ann.lbl_d.parse_symboldef(line)
                    self.symbol_map_d[level] = mapping
                except:
                    print("LabelConverterUtils.nedc_read_lbl(): %s %s."
                          % ("Error parsing symbols", fname_a))
                    return False

            # parse a single label
            #
            elif line.startswith(DELIM_LBL_LABEL):
                try:
                    lev, sub, start, stop, chan, symbols = \
                        self.ann.lbl_d.parse_label(line)

                    # append annotation to segments
                    #
                    if chan not in segments:
                        segments[chan] = []

                    label = next(key for key, value
                                 in self.symbol_map_d[0].items()
                                 if value == list(symbols.keys())[0])
                    segments[chan].append([float(start), float(stop), label])

                except:
                    print("LabelConverterUtils.nedc_read_lbl(): %s %s."
                          % ("Error parsing labels", fname_a))
                    return False

        # check if each channel has a signal
        #
        for ch in range(self.num_ch):

            # if the segments are missing a channel, add a null one
            #
            if ch not in segments:
                null_channel = {ch: [NULL_SEGMENT]}
                segments.update(null_channel)

        # sort the segments of each channel
        #
        for channel in segments:
            segments[channel] = sorted(segments[channel], key=itemgetter(0))

        # collapse the events in each channel to prevent overlaps of
        # the same event
        #
        tmp_segs = {}

        for channel in segments:
            seg_length = len(segments[channel])
            if channel not in tmp_segs:
                tmp_segs[channel] = []

            for seg_num in range(seg_length):
                current_seg = segments[channel][seg_num]

                if seg_num <= seg_length - 2:
                    next_seg = segments[channel][seg_num + 1]
                    if ((current_seg[1] >= next_seg[0])
                            and (current_seg[2] == next_seg[2])):
                        tmp_segs[channel].append([current_seg[0], next_seg[0],
                                                  current_seg[2]])
                    else:
                        tmp_segs[channel].append(current_seg)

                else:
                    tmp_segs[channel].append(current_seg)

        segments = tmp_segs

        # exit gracefully
        #
        return segments
    #
    # end of method

    # --------------------------------------------------------------------------
    #
    # methods for writing label files
    #
    # --------------------------------------------------------------------------

    # method: nedc_write_labels
    #
    # arguments:
    #  fname: name of the output file
    #  segments: the segments to be written
    #  ftype: output file type
    #  aggregate: a boolean indicating whether or not the output lab file
    #             should be aggregated
    #
    # return: boolean indicating success
    #
    # This method writes the segments to the specified output file type based
    # on the arguments
    #
    def nedc_write_labels(self, fname_a, segments_a, ftype_a, aggregate_a):

        # check if the segments have to be aggregated
        #
        if aggregate_a:
            segments_a = self.nedc_aggregate_segments(segments_a)

        # round the segments if the type is a lab file and the segments
        # are not being aggregated
        #
        if ftype_a == NEDC_LAB_EXT and not aggregate_a:
            segments_a = self.nedc_round_segments(segments_a)

        # check the output filetype
        #
        if ftype_a == NEDC_CSV_EXT:
            status = self.nedc_write_csv(fname_a, segments_a)

        elif ftype_a == NEDC_XML_EXT:
            status = self.nedc_write_xml(fname_a, segments_a)

        elif ftype_a == NEDC_TSE_EXT:
            status = self.nedc_write_tse(fname_a, segments_a)

        elif ftype_a == NEDC_LBL_EXT:
            status = self.nedc_write_lbl(fname_a, segments_a)

        # if the file type is not supported, print an error message
        #
        else:
            print("LabelConverterUtils.nedc_write_labels(): %s %s"
                  % ("the output file type is not supported", ftype_a))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: nedc_write_csv
    #
    # arguments:
    #  fname: filename of the output rec file
    #  segments: segments to be written
    #
    # return: boolean depicting method success
    #
    # This method takes in an output file name and segments. It then creates
    # the label file.
    #
    def nedc_write_csv(self, fname_a, segments_a):

        # load annotations to AnnEegGr
        #
        try:
            self.ann.csv_d.graph_d.graph_d = {}

            for channel in segments_a:
                for segment in segments_a[channel]:
                    
                    label = self.symbol_map_d[0][segment[2]]
                    sym = {label: 1.0}

                    self.ann.csv_d.graph_d.create(int(0), int(0),
                                                channel, segment[0], segment[1], sym)

            # make sure graph is sorted after loading
            #
            self.ann.csv_d.graph_d.sort()

            # write to file
            #
            self.ann.csv_d.write(fname_a, int(0), int(0))

        except:
            print("LabelConverterUtils.nedc_write_csv(): %s %s %s."
                  % ("unable to write to output file",
                     "make sure the correct parameter file is being used:", fname_a))
            return False

    #     # exit gracefully
    #     #
    #     return True
    # #
    # # end of function

    # method: nedc_write_xml
    #
    # arguments:
    #  fname: filename of the output rec file
    #  segments: segments to be written
    #
    # return: boolean depicting method success
    #
    # This method takes in an output file name and segments. It then creates
    # the label file.
    #
    def nedc_write_xml(self, fname_a, segments_a):

        # load annotations to AnnEegGr
        #
        try:
            self.ann.xml_d.graph_d.graph_d = {}

            for channel in segments_a:
                for segment in segments_a[channel]:
                    label = self.symbol_map_d[0][segment[2]]
                    sym = {label: 1.0}

                    self.ann.xml_d.graph_d.create(int(0), int(0),
                                                  channel, segment[0], segment[1], sym)

            # write to file
            #
            self.ann.xml_d.write(fname_a, int(0), int(0))
        except:
            print("LabelConverterUtils.nedc_write_xml(): %s %s %s."
                  % ("unable to write to output file",
                     "make sure the correct parameter file is being used:", fname_a))
            return False
        return True

    # method: nedc_write_tse
    #
    # arguments:
    #  fname: filename of the output rec file
    #  segments: segments to be written
    #
    # return: boolean depicting method success
    #
    # This method takes in an output file name and segments. It then creates
    # the label file.
    #
    def nedc_write_tse(self, fname_a, segments_a):

        # load annotations to AnnEegGr
        #
        try:
            self.ann.tse_d.graph_d.graph_d = {}

            for channel in segments_a:
                for segment in segments_a[channel]:
                    label = self.symbol_map_d[0][segment[2]]
                    sym = {label: 1.0}

                    self.ann.tse_d.graph_d.create(int(0), int(0),
                                                  channel, segment[0], segment[1], sym)

            # make sure graph is sorted after loading
            #
            self.ann.tse_d.graph_d.sort()

            # write to file
            #
            self.ann.tse_d.write(fname_a, int(0), int(0))

        except:
            print("LabelConverterUtils.nedc_write_csv(): %s %s %s."
                  % ("unable to write to output file",
                     "make sure the correct parameter file is being used:", fname_a))
            return False

        # exit gracefully
        #
        return True
    #
    # end of function

    # method: nedc_write_lbl
    #
    # arguments:
    #  fname: filename of the output lbl file
    #  segments: segments to be written
    #
    # return: boolean depicting method success
    #
    # This method takes in an output file name and segments. It then creates
    # the label file in lbl format.
    #
    def nedc_write_lbl(self, fname_a, segments_a):

        try:
            self.ann.lbl_d.graph_d.graph_d = {}

            for channel in segments_a:
                for segment in segments_a[channel]:
                    label = self.symbol_map_d[0][segment[2]]
                    sym = {label: 1.0}

                    self.ann.lbl_d.graph_d.create(int(0), int(0),
                                                  channel, segment[0], segment[1], sym)

            # make sure graph is sorted after loading
            #
            self.ann.lbl_d.graph_d.sort()

            # write to file
            #
            self.ann.lbl_d.write(fname_a, int(0), int(0))

        # if the output file could not be written to, print an error message
        # and return
        #
        except:
            print("LabelConverterUtils.nedc_write_lbl(): %s %s %s."
                  % ("unable to write to output file",
                     "make sure the correct parameter file is being used:",
                     fname_a))
            return False
    #
    # end of method

    # method: nedc_aggregate_segments
    #
    # arguments:
    #  segments: segments to be written
    #
    # return: boolean depicting method success
    #
    # This method takes in the segments and aggregates them in order to create
    # an overall lab file
    #
    def nedc_aggregate_segments(self, segments_a):

        # initialize local variables
        #
        channels = []
        unique_events = []
        agg_events = []
        aggregated_segments = {}

        # iterate over the dictionary of events indexed by channel
        #
        for channel in segments_a:
            channels.append(channel)
            event_list = segments_a[channel]

            # loop through the events in the list for this channel
            #
            for event in event_list:

                if event not in unique_events:
                    unique_events.append(event)
            #
            # end of for
        #
        # end of for

        # sort the list according to start times (this is important
        # as the aggregating process assumes the preceding event
        # start chronologically before the current event
        #
        unique_events = sorted(unique_events, key=itemgetter(0, 1))

        # loop through the events in the unique event list, compare
        # the start times, stop times, and event priorities to generate
        # the aggregated segments_a
        #
        for unique_event in unique_events:

            # if the aggregated list is empty, apend the unique event to it
            #
            if not agg_events:
                agg_events.append(unique_event)

            # if it is not empty, the times and priorities must be compared
            # to complete the aggregated list
            #
            else:
                agg_start_t = agg_events[-1][0]
                agg_end_t = agg_events[-1][1]
                agg_event_id = agg_events[-1][2]
                unq_start_t = unique_event[0]
                unq_end_t = unique_event[1]
                unq_event_id = unique_event[2]

                # if the unique and aggregated events overlap, check the
                # priorities
                #
                if int(unq_start_t * 100) < int(agg_end_t * 100):

                    # if the unique start time is less than the aggregated
                    # start time, other aggregated events will be affected
                    #
                    if int(unq_start_t * 100) < int(agg_start_t * 100):

                        # set variables before the loop
                        #  the next_agg_label varaible is used as an index for
                        #  each aggregated event
                        #
                        next_agg_label = -1
                        next_agg_start_t = agg_events[next_agg_label][0]
                        next_unq_end_t = unique_event[1]

                        # loop through the aggregaed events until the start
                        # time of the aggregated label us less than or equal to
                        # the start time of the unique label
                        #
                        while(int(unq_start_t * 100) <
                              int(next_agg_start_t * 100)):

                            # get the start and end time and event ID of the
                            # current label
                            #
                            next_agg_start_t = agg_events[next_agg_label][0]
                            next_agg_end_t = agg_events[next_agg_label][1]
                            next_agg_event_id = agg_events[next_agg_label][2]

                            # check if the end time of the unique event is less
                            # than the start time of the aggregated event
                            #
                            if (int(unq_end_t * 100) >
                                    int(next_agg_start_t * 100)):

                                # if the unique event ID is the same as the
                                # aggregated event ID, get the upper bound and
                                # edit the aggregated event
                                #
                                if unq_event_id == next_agg_event_id:
                                    upper_bound = max(next_agg_end_t,
                                                      next_unq_end_t)
                                    agg_events[next_agg_label] = (
                                        [next_agg_start_t, upper_bound,
                                         next_agg_event_id])

                                # if the unique event has higher priority,
                                # edit the next aggregated event accordingly
                                #
                                elif (self.priority_map[next_agg_event_id] <
                                      self.priority_map[unq_event_id]):

                                    seg_len = (next_unq_end_t -
                                               next_agg_start_t)

                                    # if the unique start time is greater than
                                    # the aggregated end time, make the
                                    # aggregated event the unique event
                                    #
                                    if (int(unq_start_t * 100) >
                                        int(next_agg_end_t * 100) and
                                            unq_start_t >= next_agg_start_t):
                                        agg_events[next_agg_label] = (
                                            [next_agg_start_t, next_agg_end_t,
                                             unq_event_id])
                                        break

                                    # if the segment length is less than the
                                    # window duration, extend windows
                                    #
                                    elif (seg_len < self.epoch_wdur):

                                        if ((next_agg_end_t -
                                            (next_agg_start_t +
                                             self.epoch_wdur)) >=
                                                self.epoch_wdur):

                                            agg_events[next_agg_label] = (
                                                [(next_agg_start_t +
                                                  self.epoch_wdur),
                                                 next_agg_end_t,
                                                 next_agg_event_id])

                                            # append the extended window for
                                            # the unique label
                                            #
                                            agg_events.append(
                                                [next_agg_start_t,
                                                 (next_agg_start_t +
                                                  self.epoch_wdur),
                                                 unq_event_id])

                                            next_agg_label -= 1

                                        else:

                                            agg_events[next_agg_label] = (
                                                [next_agg_start_t,
                                                 next_agg_end_t,
                                                 unq_event_id])

                                        # sort the aggregated events
                                        #
                                        agg_events = sorted(
                                            agg_events, key=itemgetter(0, 1))

                                    else:
                                        agg_events[next_agg_label] = (
                                            [next_agg_start_t, next_unq_end_t,
                                             unq_event_id])

                                        if (next_unq_end_t < next_agg_end_t):
                                            agg_events.append(
                                                [next_unq_end_t,
                                                 next_agg_end_t,
                                                 next_agg_event_id])
                                            next_agg_label -= 1

                            # sort the aggregated events
                            #
                            agg_events = sorted(agg_events,
                                                key=itemgetter(0, 1))

                            # decrement the loop counter for the next label
                            # and set the next unique end time equal to the
                            # start time of the current aggregated label
                            #
                            next_agg_label -= 1
                            next_unq_end_t = next_agg_start_t
                        #
                        # end of while

                    else:

                        # if the events have the same label, extend the time of
                        # the stop time of the aggregated event if needed
                        #
                        if agg_event_id == unq_event_id:
                            upper_bound = max(agg_end_t, unq_end_t)
                            agg_events[-1] = [agg_start_t, upper_bound,
                                              agg_event_id]

                        # if they are not the same, check if the unique
                        # event has higher priority than the aggregated
                        # event
                        #
                        elif (self.priority_map[agg_event_id] <
                              self.priority_map[unq_event_id]):

                            if (int(unq_start_t * 100) ==
                                    int(agg_start_t * 100)):
                                agg_events[-1] = [agg_start_t, unq_end_t,
                                                  unq_event_id]
                            else:
                                if (unq_start_t - agg_start_t <
                                        self.epoch_wdur):
                                    agg_events[-1] = (
                                        [agg_start_t, unq_end_t, unq_event_id])
                                    continue

                                if (unq_end_t - agg_end_t < self.epoch_wdur):
                                    agg_events[-1] = [agg_start_t, unq_start_t,
                                                      agg_event_id]

                                if int(unq_end_t * 100) < int(agg_end_t * 100):
                                    agg_events.append(unique_event)

                                # append the remainder of the last aggregated
                                # event if necessary
                                #
                                if (int(agg_end_t * 100) >
                                        int((unq_end_t) * 100)):
                                    agg_events.append([unq_end_t, agg_end_t,
                                                       agg_event_id])

                        # if the aggregated event has higher priority than the
                        # unique event, overwite the section  of the aggregated
                        # segments the unique event
                        #
                        else:
                            if (int(agg_end_t * 100) < int(unq_end_t * 100)):

                                if (unq_end_t - agg_end_t >= self.epoch_wdur):
                                    agg_events.append([agg_end_t, unq_end_t,
                                                       unq_event_id])
                                else:
                                    agg_events.append(
                                        [agg_end_t,
                                         (agg_end_t + self.epoch_wdur),
                                         unq_event_id])

                # if there is no overlap, append the unique event to the
                # aggregated segments
                #
                else:

                    if (unique_event[1] - unique_event[0] >= self.epoch_wdur):
                        agg_events.append(unique_event)
                    else:
                        agg_events.append([unique_event[0],
                                           (unique_event[0] +
                                           self.epoch_wdur),
                                           unique_event[2]])

                # sort the aggregated events
                #
                agg_events = sorted(agg_events, key=itemgetter(0, 1))

        #
        # end of for

        # find sequential events with the same ID to collapse them into
        # a single event with an extended duration
        #
        delete_list = []
        for event in range(1, len(agg_events)):
            if agg_events[event][2] == agg_events[event-1][2]:
                agg_events[event] = [agg_events[event-1][0],
                                     agg_events[event][1], agg_events[event][2]]
                delete_list.append(agg_events[event-1])

        # delete the events that have been made obsolete by the collapse
        #
        for item in delete_list:
            del agg_events[agg_events.index(item)]

        # collapse the unique events to intelligently make all segment
        # durations a multiple of the epoch window duration based on the
        # priority
        #
        deleted_signals = 0
        for event in range(len(agg_events) - 1):

            # values for the current label
            #
            event = event - deleted_signals
            current_start_t = agg_events[event][0]
            current_end_t = agg_events[event][1]
            current_event = agg_events[event][2]
            current_seg_len = current_end_t - current_start_t
            current_remainder = current_seg_len % self.epoch_wdur

            # values for the next label
            #
            next_start_t = agg_events[event + 1][0]
            next_end_t = agg_events[event + 1][1]
            next_event = agg_events[event + 1][2]

            # if the seg length of the current event is not evenly
            # divisible by the epoch window duration, extend the windows
            # accordingly
            #
            if current_remainder != 0:

                # if the priority of the current segment is greater than
                # that of the next segment, extend the current segment
                # into the next segment
                #
                if (self.priority_map[current_event] >
                        self.priority_map[next_event]):

                    # if the next event is long enough to be cut off,
                    # shorten the next event and expand the current event to
                    # make the current event a multiple of the epoch duration
                    #
                    if (next_end_t - (current_end_t +
                                      (self.epoch_wdur - current_remainder))
                            >= self.epoch_wdur):
                        agg_events[event] = (
                            [current_start_t,
                             (current_end_t +
                              (self.epoch_wdur - current_remainder)),
                             current_event])
                        agg_events[event+1] = (
                            [(current_end_t +
                             (self.epoch_wdur - current_remainder)),
                             next_end_t, next_event])

                    # if the next event cannot be shortened, make it a part of
                    # the unique event
                    #
                    else:
                        agg_events[event] = (
                            [current_start_t, next_end_t, current_event])
                        del agg_events[event + 1]
                        deleted_signals += 1

                # if the current event has a lower priority than the next event
                # then, depending on the current event's lenghth, shorten it or
                # make it a part of the next event
                #
                else:

                    # if the current event can be shortened, shorten it and
                    # extend the next event
                    #
                    if (((current_end_t - current_remainder) - current_start_t)
                            >= self.epoch_wdur):
                        agg_events[event] = (
                            [current_start_t,
                             (current_end_t - current_remainder),
                             current_event])
                        agg_events[event+1] = (
                            [current_end_t - current_remainder,
                             next_end_t,
                             next_event])

                    # if the current event cannot be shortened, make it a part
                    # of the next event
                    #
                    else:
                        agg_events[event] = (
                            [current_start_t, next_end_t, next_event])
                        del agg_events[event]
                        deleted_signals += 1
        #
        # end of for

        aggregated_segments = {0: agg_events}
        return aggregated_segments
    #
    # end of method

    # method: nedc_round_segments
    #
    # arguments:
    #  segments: segments to be rounded
    #
    # return: the rounded segments
    #
    # This method takes input segments and round the start and end times
    #
    def nedc_round_segments(self, segments_a):

        # initialize local variables
        #
        new_segments = {}

        # round the start and end times of the segments
        #
        for channel in segments_a:
            for segment in segments_a[channel]:
                if channel not in new_segments:
                    new_segments[channel] = []
                if (round(segment[1]) - round(segment[0]) >= 1):
                    new_segments[channel].append([round(segment[0]),
                                                  round(segment[1]),
                                                  segment[2]])

        # exit gracefully
        #
        return new_segments
    #
    # end of method

    # --------------------------------------------------------------------------
    #
    # informational methods
    #
    # --------------------------------------------------------------------------

    # method: nedc_is_csv
    #
    # arguments:
    #  fname: name of the input file
    #
    # return: boolean indicating whether or not the file is a csv
    #
    # This method takes in a filename and checks if it is a csv file.
    #
    def nedc_is_csv(self, fname_a):
        
        # open the file and check the first line
        #
        try:
            status = self.ann.csv_d.load(fname_a)

        # if the file could not be opened, print an error message and exit
        #
        except:
            return False

        # if all of the tests were passed, return True
        #
        return status
    #
    # end of method

    # method: nedc_is_xml
    #
    # arguments:
    #  fname: name of the input file
    #
    # return: boolean indicating whether or not the file is a xml
    #
    # This method takes in a filename and checks if it is a xml file.
    #
    def nedc_is_xml(self, fname_a):

        # open the file and check the first line
        #
        try:
            status = self.ann.xml_d.load(fname_a)
            
        # if the file could not be opened, print an error message and exit
        #
        except:
            return False

        # if all of the tests were passed, return True
        #
        return status
    #
    # end of method

    # method: nedc_is_tse
    #
    # arguments:
    #  fname: name of the input file
    #
    # return: boolean indicating whether or not the file is a lbl
    #
    # This method takes in a filename and checks if it is a lbl file.
    #
    def nedc_is_tse(self, fname_a):

        try:
            status = self.ann.tse_d.load(fname_a)

        # if the file lines could not be processed, return False
        #
        except:
            print("LabelConverterUtils.nedc_is_tse(): %s %s."
                  % ("the file could not be processed:", fname_a))
            return False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: nedc_is_lbl
    #
    # arguments:
    #  fname: name of the input file
    #
    # return: boolean indicating whether or not the file is a lbl
    #
    # This method takes in a filename and checks if it is a lbl file.
    #
    def nedc_is_lbl(self, fname_a):

        try:
            status = self.ann.lbl_d.load(fname_a)

        # if the file lines could not be processed, return False
        #
        except:
            print("LabelConverterUtils.nedc_is_lbl(): %s %s."
                  % ("the file could not be processed:", fname_a))
            return False

        # exit gracefully
        #
        return status
    #
    # end of method

    # --------------------------------------------------------------------------
    #
    # methods for reading parameter files
    #
    # --------------------------------------------------------------------------

    def set_demo_config(self,
                        dict_labels_priority_a,
                        dict_event_map_a,
                        dict_priority_map_a):

        # add the null event
        #
        null_event = {NULL_EVENT_LABEL: [NULL_EVENT_ID, NULL_EVENT_PRIORITY]}
        dict_labels_priority_a.update(null_event)
        dict_event_map_a.update({NULL_EVENT_ID: NULL_EVENT_LABEL})
        dict_priority_map_a.update({NULL_EVENT_ID: NULL_EVENT_PRIORITY})

        # find the event id for the BCKG event
        #
        for event_id in dict_event_map_a:
            if dict_event_map_a[event_id].upper() == NEDC_BCKG_EVENT_LABEL:
                self.bckg_id = event_id

        self.params = dict_labels_priority_a
        self.event_map = dict_event_map_a
        self.priority_map = dict_priority_map_a
        return dict_labels_priority_a, dict_event_map_a

    # method: nedc_read_param
    #
    # arguments:
    #  fname: the name to the parameters file.
    #
    # return: a dictionary including in the form
    #         {label_name: [label_id, label_order]}
    #
    # This method reads a file containing a list of labels and their
    # corresponding ID and proirity information
    #
    def nedc_read_param(self, fname_a):

        # initialize local variables
        #
        labels_priority = {}
        event_map = {}
        priority_map = {}

        # load the file into memory
        #
        lines = [line.strip() for line in open(fname_a)]

        # read the parameter file to find the labels section
        #
        for line in lines:
            if line.startswith("labels {"):
                ind = lines.index(line) + 1
                break

        #
        # end of for

        # loop through the labels section and create a dictionary
        # corresponding to the parameters
        #
        while lines[ind].find("}") == -1:
            parts = lines[ind].split(",")

            # create the label priority dictionary
            #
            if parts[0] not in labels_priority:
                labels_priority[parts[0]] = []
            labels_priority[parts[0]] = [int(parts[1]), int(parts[2])]
            ind += 1

            # create the label mapping and priotity mapping dictionary
            #
            event_map[int(parts[1])] = parts[0]
            priority_map[int(parts[1])] = int(parts[2])
        #
        # end of while

        # add the null event
        #
        null_event = {NULL_EVENT_LABEL: [NULL_EVENT_ID, NULL_EVENT_PRIORITY]}
        labels_priority.update(null_event)
        event_map.update({NULL_EVENT_ID: NULL_EVENT_LABEL})
        priority_map.update({NULL_EVENT_ID: NULL_EVENT_PRIORITY})

        # find the event id for the BCKG event
        #
        for event_id in event_map:
            if event_map[event_id].upper() == NEDC_BCKG_EVENT_LABEL:
                self.bckg_id = event_id

        # exit gracefully
        #
        self.params = labels_priority
        self.event_map = event_map
        self.priority_map = priority_map
        return labels_priority, event_map
    #
    # end of function

    # -------------------------------------------------------------------------
    #
    # methods for processing files
    #
    # -------------------------------------------------------------------------

    # method: nedc_process_ifiles
    #
    # arguments:
    #  ifiles: list of files to be processed. May contain files that are
    #          lists of files as well.
    #
    # return: a list of all the files to be processed
    #
    # This method takes a list of files to be processed and files that
    # contain a list of files to be processed and merges them into one list
    # to be returned.
    #
    def nedc_process_ifiles(self, ifiles_a):

        # initialize local variables
        #
        ofiles = []
        efiles = []

        # loop through the list of files
        #
        for ifile in ifiles_a:

            # if the file ends with the list extension, open it and process
            # its contents
            #
            if ifile.endswith(NEDC_LST_EXT):

                # open the file file to process it
                #
                with open(ifile, "r") as f:

                    # read the file line by line
                    #
                    for line in f.readlines():

                        # get the absolute path of the file
                        #
                        line = os.path.abspath(line.strip())
                        ofiles.append(line)
                    #
                    # end of for
                #
                # end of with

            # if the file is not a list, get the absolute path
            #
            else:
                ifile = os.path.abspath(ifile)

                # append the file to the output list
                #
                ofiles.append(ifile)

        #
        # end of for

        # exit gracefully
        #
        return ofiles
    #
    # end of method

    # method: nedc_create_filename
    #
    # arguments:
    #  fname: filename
    #  odir: the output directory
    #  rdir: the replace directory
    #  oext: the extension to replace the previous one
    #
    # return: a filename with new directory name and extension
    #
    # This method generates a filename based on the input with a new directory
    # name and extension.
    #
    def nedc_create_filename(self, fname_a, odir_a, rdir_a, oext_a):

        # get the absolute path of the filename and replace the extension
        #
        abs_fname = os.path.abspath(fname_a).replace(NEDC_DSK_RAID, NULL)
        ofname = (os.path.join(os.path.dirname(abs_fname),
                               os.path.basename(abs_fname).split(DOT)[0]
                               + DOT + oext_a))
        odir_a = os.path.abspath(odir_a).replace(NEDC_DSK_RAID, NULL)

        # generalize the output directory by removing the ending slash
        #
        odir_a = os.path.abspath(odir_a).replace(NEDC_DSK_RAID, NULL)
        while odir_a.endswith("/"):
            odir_a = odir_a[0:len(odir_a) - 1]

        # branch if the replace directory is passed
        #
        if rdir_a != NULL and rdir_a in ofname:

            # generalize the replace directory
            #
            while rdir_a.endswith("/"):
                rdir_a = rdir_a[0:len(rdir_a) - 1]

            # replace the replace directory portion of the path with the
            # output directory
            #
            ofname = ofname.replace(rdir_a, odir_a)

        # if the replace directory is not specified, append the basename
        # of the output file name to the output directory
        #
        else:
            ofname = os.path.join(odir_a, os.path.basename(ofname))

        # exit gracefully
        #
        return ofname
    #
    # end of method

    # method: nedc_create_filelist
    #
    # arguments:
    #  fname: input filename without channel modifier
    #
    # return: list of files with channel modifiers
    #
    # This method creates a list of filenames based on the number of channels
    # with the specified chanel modifier.
    #
    def nedc_create_filelist(self, fname_a):

        # initialize local variables and split the filename by the extension
        #
        ofiles = []
        fname_a = os.path.abspath(fname_a)
        base_fname = os.path.basename(fname_a)
        base, ext = base_fname.split(DOT)

        # loop through the number of channels and get the modifier
        #
        for ch in range(22):
            mod = (self.cfmt % ch)
            ofile = os.path.join(os.path.dirname(fname_a),
                                 base + mod + DOT + ext)
            ofiles.append(ofile)

        # exit gracefully
        #
        return ofiles
    #
    # end of method
#
# end of class
