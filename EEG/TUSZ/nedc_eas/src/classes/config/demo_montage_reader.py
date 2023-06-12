#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/config/demo_montage_reader.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
import re
import numpy as np

#----------------------------------------------------------------------------------
#
# file: DemoMontageModule
#
# this file separately parses the [Montage] section of preferences.cfg
# into standard montage definitions
#
class DemoMontageModule(object):

    # method: __init__
    #
    # arguments: montage_file_a
    #
    # returns: None
    #
    # initializes DemoMontageModule and calls parse_montage
    #
    def __init__(self,
                 montage_file_a=None):
        if montage_file_a is not None:
            self.montage_file = montage_file_a

    # method: parse_montage
    #
    # arguments: montage_file_a
    #
    # returns: None
    #
    # parses montage definitions into names, minuends, and subtrahends
    #
    def parse_montage(self,
                      montage_file_a):

        self.montage_names = {}
        self.montage_minuend = {}
        self.montage_subtrahend = {}

        lines = self._get_lines(montage_file_a)

        lines = [line.split("=")[1].strip() for line in lines]
        
        # reads montage.txt line by line.
        #
        sig_counter = 0
        for line in lines:
            list_separated_by_comma = line.split(",")
            name_and_montage_definition = list_separated_by_comma[1].split(":")
            montage_def = name_and_montage_definition[1].split(" -- ")

            self.montage_names[sig_counter] = name_and_montage_definition[0]
            self.montage_minuend[sig_counter] = montage_def[0].strip()

            test_for_subtrahend = (len(montage_def) > 1)
            if test_for_subtrahend:
                test_for_subtrahend = (montage_def[1].strip() != "None")
            if test_for_subtrahend:
                self.montage_subtrahend[sig_counter] = montage_def[1].strip()
            else:
                self.montage_subtrahend[sig_counter] = None
            sig_counter += 1

        self.number_signals = sig_counter

    def _get_lines(self,
                   montage_file_a):
        lines = []
        with open(montage_file_a) as montage_file:
            for line in montage_file:
                if line.startswith("montage ="):
                    lines.append(line)
        return lines
        
    def get_number_signals(self):
        return self.number_signals

    def get_montage_names(self):
        return self.montage_names

    def get_montage_lines_for_writing(self):
        return self._get_lines(self.montage_file)

    def map_montage_to_edf(self,
                           channels_read_from_edf_a):

        # Using this dictionary, every part of a signal (self.montage_minuend -
        # self.montage_subtrahend), will be assigned to a physical channel by
        # its index in edf file.
        #
        montage_dict = {}
        self.montage_minuend_index = {}
        self.montage_subtrahend_index = {}
        header_and_montage_match = True

        number_channels = len(channels_read_from_edf_a)

        for i in range(number_channels):
            for j in range(self.number_signals):
                m_str = str(self.montage_minuend[j])
                m_pat = re.compile(
                    r"([^_-]*)\s+" + m_str + r"\s*$", re.IGNORECASE)

                if m_pat.match(" " + channels_read_from_edf_a[i]):
                    montage_dict[self.montage_minuend[j]] = \
                        channels_read_from_edf_a.index(
                            channels_read_from_edf_a[i])

                s_str = str(self.montage_subtrahend[j])
                s_pat = re.compile(
                    r"([^_-]*)\s+" + s_str + r"\s*$", re.IGNORECASE)
                if s_pat.match(" " + channels_read_from_edf_a[i]):
                    montage_dict[self.montage_subtrahend[j]] = \
                        channels_read_from_edf_a.index(
                            channels_read_from_edf_a[i])

        for j in range(self.number_signals):

            minuend_chan_str = montage_dict.get(self.montage_minuend[j])

            if minuend_chan_str is not None:
                self.montage_minuend_index[j] = \
                    montage_dict[self.montage_minuend[j]]
            else:
                header_and_montage_match = False
            
            # this allows us to support either of these two versions
            # of a montage:
            #
            # 0:EKG,EKG-REF
            #   ~or~
            # 0:EKG,EKG-REF -- None
            #
            subtrahend_chan_str = montage_dict.get(self.montage_subtrahend[j])
            if (subtrahend_chan_str == "None"
                or subtrahend_chan_str is None):
                self.montage_subtrahend_index[j] = None
            else:
                self.montage_subtrahend_index[j] = \
                    montage_dict[self.montage_subtrahend[j]]

        return self.montage_minuend, self.montage_subtrahend, header_and_montage_match

    def montage_differentiation(self,
                                raw_channel_data_a):

        # signals y data which is y2 - y1. For example when we want to
        # calculate FP1 - F7, we deafine all instances which are related to
        # FP1 as minuend and for F7 as subtrahend. Then we calculate -(FP1
        # - F7) or - (y(minuend) - y(subtrahend)).
        #
        y_data = {}
        number_data_points = np.size(raw_channel_data_a[0])

        for i in range(self.number_signals):
            index_minuend = self.montage_minuend_index[i]
            index_subtrahend = self.montage_subtrahend_index[i]

            if index_minuend is None:
                y1 = np.zeros(number_data_points)
            else:
                y1 = raw_channel_data_a[self.montage_minuend_index[i]]
            if index_subtrahend is None:
                y2 = np.zeros(number_data_points)
            else:
                y2 = raw_channel_data_a[self.montage_subtrahend_index[i]]

            # inverse subtraction
            #
            y_data[i] = y1 - y2
        #
        # end of for
        return y_data
