#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/search/demo_report_object.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from demo_concept_object import DemoConceptObject

class DemoReportObject:

    def __init__(report_name_a,
                 patient_gender_a,
                 patient_age_a,
                 report_concepts=None):

        self.report_name = report_name_a
        self.patient_gender = patient_gender_a
        self.patient_age = patient_age_a

        if report_concepts_a is not None:
            self.report_concepts = report_concepts_a

        if hasattr(self, 'report_concepts'):
            self.init_concept_object

    def init_concept_object():

        start_positions = report_concepts[0]
        end_positions = report_concepts[1]
        attr_types = report_concepts[2]

        self.concept_obj = DemoConceptObject(start_positions,
                                             end_positions,
                                             attr_types)
        
