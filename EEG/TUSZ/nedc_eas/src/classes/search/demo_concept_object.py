#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/search/demo_concept_object.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
class DemoConceptObject:

    def __init__(start_positions_a,
                 end_positions_a,
                 attr_types_a):

        self.start_positions_a = start_positions_a
        self.end_positions_a = end_positions_a
        self.attr_types = attr_types_a
        
