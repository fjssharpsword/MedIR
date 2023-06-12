#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/search/demo_session_object.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from demo_eeg_object import DemoEEGObject
from demo_report_object import DemoReportObject

class DemoSessionObject:

    def __init__(eeg_item_a,
                 report_item_a):
        self.eeg_item = eeg_item
        self.report_item = report_item_a

        self.init_eeg_object()
        self.init_report_object()

    def init_eeg_object():

        eeg_name = self.eeg_item[0]
        session_rank = self.eeg_item[1]
        file_rank = self.eeg_item[2]
        file_labels = self.eeg_item[3]

        if file_labels is not None:
            self.eeg_obj = DemoEEGObject(eeg_name,
                                         session_rank,
                                         file_rank,
                                         file_labels)
        else:
            self.eeg_obj = DemoEEGObject(eeg_name,
                                         session_rank,
                                         file_rank)

    def init_report_object():

        report_name = self.report_item[0]
        patient_gender = self.report_item[1]
        patient_age = self.report_item[2]
        report_concepts = self.report_item[3]

        if report_concepts is not None:
            self.report_obj = DemoReportObject(report_name,
                                               patient_gender,
                                               patient_age,
                                               report_concepts)
        else:
            self.report_obj = DemoReportObject(report_name,
                                               patient_gender,
                                               patient_age)
