#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/search/demo_label_object.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
class DemoLabelObject:

    def __init__(start_a,
                 stop_a,
                 label_a,
                 probability_a):
        
        self.start_times = start_a
        self.stop_times = stop_a,
        self.labels_list = label_a,
        self.prob_list = probability_a
