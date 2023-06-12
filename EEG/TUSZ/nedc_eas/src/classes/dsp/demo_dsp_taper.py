#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/dsp/demo_dsp_taper.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
import numpy as np

class DemoDSPTaper:
    def __init__(self,
                 string_window_type_a):
        self.tapering_fctn_options = {"bartlett"   : np.bartlett,
                                      "blackman"   : np.blackman,
                                      "hanning"    : np.hanning,
                                      "hamming"    : np.hamming,
                                      "kaiser"     : self.kaiser,
                                      "rectangular": np.ones}
        down_case_taper_a = string_window_type_a.lower()
        self.taper_fctn = self.tapering_fctn_options[down_case_taper_a]

    # for automating the beta parameter in the call to making a kaiser
    # window. unessential, but completes a general functionality of having
    # variable window envelope types. The Kaiser window is a special case,
    # the other windows do not have this 'beta' parameter.
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.kaiser.html
    # TODO: should beta be user configurable?
    # M is number of points in output window
    #
    def kaiser(self,
               M):
        beta = 14
        return np.kaiser(M, beta)

    def set_taper(self,
                  size_a,
                  taper_fctn_a=None):
        if taper_fctn_a is not None:
            down_case_taper_a = taper_fctn_a.lower()
            self.taper_fctn = self.tapering_fctn_options[down_case_taper_a]
        self.taper_array = self.taper_fctn(size_a)
