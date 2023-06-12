#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/search/demo_eeg_object.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from demo_label_object import DemoLabelObject

class DemoEEGObject:

    def __init__(eeg_name_a,
                 session_rank_a,
                 file_rank_a,
                 file_labels_a=None):

        self.eeg_name = eeg_name_a
        self.session_rank = session_rank_a
        self.file_rank = file_rank_a

        if file_labels_a is not None:
            self.file_labels = file_labels_a

        if hasattr(self, 'file_labels'):
            self.init_label_object()


    def init_label_object():
        
        start_times = file_labels[0]
        stop_times = file_labels[1]
        label_list = file_labels[2]
        prob_list = file_labels[3]

        self.label_obj = DemoLabelObject(start_times,
                                         stop_times,
                                         label_list,
                                         prob_list)
        
