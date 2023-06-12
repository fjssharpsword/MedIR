# (1) designing# (1) designin# (1) designingg# The three base filters:
#  - "Low Cut"  (high pass)
#  - "High Cut" (low pass)
#  - "Notch"    (band reject)
#
# The frequency for all base filters is user configurable. There
# are a number of pre-programmed selections for each base filter type
# (i.e. for "Low Cut" there 5 Hz, 10 Hz, etc.),
# along with a option to select a custom frequency via a user dialog
# (dialog is in demo_user_interface.py, but prompt strings are in here
#
# "Rhythm" filters are from the "Low Cut" and "High Cut" filters.
#  - "Delta" (low pass)
#  - "Theta" (band pass)
#  - "Alpha" (band pass)
#  - "Beta"  (band pass)
#  - "Gamma" (high pass)
#
# The frequency ranges for the various rhythm filters can be set in
# src/_defaults/preferences/cfg.
# 
# Whenever the user selects a Rhythm (not Alpha, Beta, etc.) filter,
# the appropriate set of base filters is created. If there are
# previously stored user-selected filters, these are saved until the
# the user selects Rhythms -> Off, which calls self.rhythms_off(), at
# which point the previous filters are restored.
#
# The filter is made by
# designing each base filter using the scipy.signal library
#     function firwin(), which produces a polynomial numerator vector.
# (2) resetting self.filter_numerator to 1.
# (3) iterating over the base filters, and if that filter type is in
#     effect, convolving that filter's numerator vector with
#     self.numerator_vector.
#
# This module allows for filtering of a single channel. so to use this
# in another module to filter multiple channels, something like this
# must be done in the other module:
#
# filtered_y_data = {}
# for i in range(number_signals):
#     filtered_y_data[i] = this_filter_right_here.do_filter(y_data[i])
#

from scipy.signal import lfilter, group_delay
from numpy import convolve

from .demo_dsp_subfilters.demo_detrend_filter import DemoDetrendFilter
from .demo_dsp_subfilters.demo_high_cut_filter import DemoHighCutFilter
from .demo_dsp_subfilters.demo_low_cut_filter import DemoLowCutFilter
from .demo_dsp_subfilters.demo_notch_filter import DemoNotchFilter



class DemoDSPFilter(object):

    def __init__(self,
                 input_dialogue_function_a,
                 cfg_dict_rhythms_a,
                 cfg_dict_detrend_a,
                 cfg_dict_filters_default_a):

        self.custom_freq_dialogue = input_dialogue_function_a
        self.set_up_dialogue_strings()

        self.set_rhythm_freq_preferences(cfg_dict_rhythms_a)

        self.set_initial_state(cfg_dict_filters_default_a)

        self.detrend_filt = DemoDetrendFilter(
            cfg_dict_detrend_a,
            cfg_dict_filters_default_a['detrend_power'])

        self.low_cut_filt = DemoLowCutFilter(
            cfg_dict_filters_default_a['low_cut_power'])
        self.high_cut_filt = DemoHighCutFilter(
            cfg_dict_filters_default_a['high_cut_power'])
        self.notch_filt = DemoNotchFilter(
            cfg_dict_filters_default_a['notch_power'])

    def set_initial_state(self,
                          cfg_dict_filters_default):

        if cfg_dict_filters_default['detrend'] == "True":
            self.detrend_in_effect = True
        else:
            self.detrend_in_effect = False

        if cfg_dict_filters_default['notch'] == "True":
            self.notch_in_effect = True
        else:
            self.notch_in_effect = False

        self.low_cut_in_effect = False
        self.high_cut_in_effect = False
        
        self.non_trivial_filter_selected = self.notch_in_effect       \
                                           or self.low_cut_in_effect  \
                                           or self.high_cut_in_effect \
                                           or self.detrend_in_effect

        self.nyquist_limit = None
        self.filter_dict = {'high_cut': 1,
                            'low_cut': 1,
                            'notch': 1,
                            'detrend': 1}

        # declare some variables whose only use is to provide strings in
        # the custom frequency dialog if the dialog is made before any other
        # frequency has been declared
        #
        self.low_cut_cutoff_freq = 0
        self.high_cut_cutoff_freq = 100

        # default state - is changed to True when rhythm filter mode
        # is entered and false when rhythm filter mode is exited
        #
        self.state_values_saved = False

    def set_rhythm_freq_preferences(self,
                                    cfg_dict_rhythms_a):
        self.delta_low = float(cfg_dict_rhythms_a['delta'][0])
        self.delta_high = float(cfg_dict_rhythms_a['delta'][1])
        self.theta_low = float(cfg_dict_rhythms_a['theta'][0])
        self.theta_high = float(cfg_dict_rhythms_a['theta'][1])
        self.alpha_low = float(cfg_dict_rhythms_a['alpha'][0])
        self.alpha_high = float(cfg_dict_rhythms_a['alpha'][1])
        self.beta_low = float(cfg_dict_rhythms_a['beta'][0])
        self.beta_high = float(cfg_dict_rhythms_a['beta'][1])
        self.gamma_low = float(cfg_dict_rhythms_a['gamma'][0])
        self.gamma_high = float(cfg_dict_rhythms_a['gamma'][1])

    def set_detrender_range(self,
                            low_a,
                            high_a):
        self.detrend_filt.set_freqs(float(low_a),
                                    float(high_a))
    def enable_or_disable_detrender(self,
                                    desired_on=True):
        self.detrend_in_effect = desired_on
        self.update_flag_and_possibly_update_filter()
        

    def set_gnrl_power(self,
                       low_power_string_a,
                       high_power_string_a,
                       notch_power_string_a,
                       detrend_power_string_a):

        self.low_cut_filt.set_power(low_power_string_a)
        self.high_cut_filt.set_power(high_power_string_a)
        self.notch_filt.set_power(notch_power_string_a)
        self.detrend_filt.set_power(detrend_power_string_a)

    def make_filter(self):
        self.filter_numerator = 1
        if self.notch_in_effect:
            self.filter_numerator = convolve(self.filter_dict['notch'],
                                             self.filter_numerator)
        if self.low_cut_in_effect:
            self.filter_numerator = convolve(self.filter_dict['low_cut'],
                                             self.filter_numerator)
        if self.high_cut_in_effect:
            self.filter_numerator = convolve(self.filter_dict['high_cut'],
                                             self.filter_numerator)
        if self.detrend_in_effect:
            self.filter_numerator = convolve(self.filter_dict['detrend'],
                                             self.filter_numerator)
        self.phase_delay = group_delay((self.filter_numerator, 1), 1)[1]

    def do_filter(self,
                  y_data_one_channel):
        return lfilter(self.filter_numerator,
                       1,
                       y_data_one_channel)

    def update_flag_and_possibly_update_filter(self):
        self.non_trivial_filter_selected = self.notch_in_effect       \
                                           or self.low_cut_in_effect  \
                                           or self.high_cut_in_effect \
                                           or self.detrend_in_effect

        if self.non_trivial_filter_selected:
            self.make_filter()

    def update_sample_rate(self,
                           sample_rate_a):
        new_nyquist_limit = float(sample_rate_a) / 2
        if self.nyquist_limit is not new_nyquist_limit:
            self.nyquist_limit = new_nyquist_limit
            self.low_cut_filt.set_nyquist(new_nyquist_limit)
            self.high_cut_filt.set_nyquist(new_nyquist_limit)
            self.detrend_filt.set_nyquist(new_nyquist_limit)
            self.notch_filt.set_nyquist(new_nyquist_limit)

            self.filter_dict['detrend'] = self.detrend_filt.make_filt()
            self.filter_dict['notch'] = self.notch_filt.make_filt()
            self.update_flag_and_possibly_update_filter()


    def low_cut_off(self):
        self.low_cut_in_effect = False
        self.update_flag_and_possibly_update_filter()

    def low_cut_5hz(self):
        self.low_cut_in_effect = True
        self.filter_dict['low_cut'] = self.low_cut_filt.make_filt(5.0)
        self.update_flag_and_possibly_update_filter()

    def low_cut_10hz(self):
        self.low_cut_in_effect = True
        self.filter_dict['low_cut'] = self.low_cut_filt.make_filt(10.0)
        self.update_flag_and_possibly_update_filter()

    def low_cut_15hz(self):
        self.low_cut_in_effect = True
        self.filter_dict['low_cut'] = self.low_cut_filt.make_filt(15.0)
        self.update_flag_and_possibly_update_filter()

    def low_cut_20hz(self):
        self.low_cut_in_effect = True
        self.filter_dict['low_cut'] = self.low_cut_filt.make_filt(20.0)
        self.update_flag_and_possibly_update_filter()

    def low_cut_25hz(self):
        self.low_cut_in_effect = True
        self.filter_dict['low_cut'] = self.low_cut_filt.make_filt(25.0)
        self.update_flag_and_possibly_update_filter()

    def low_cut_30hz(self):
        self.low_cut_in_effect = True
        self.filter_dict['low_cut'] = self.low_cut_filt.make_filt(30.0)
        self.update_flag_and_possibly_update_filter()

    def low_cut_custom_frequency(self):
        self.low_cut_in_effect = True
        cutoff_frequency, ok = self.custom_freq_dialogue(
            self.low_cut_dialogue_title,
            self.low_cut_dialogue_prompt.format(self.nyquist_limit),
            self.low_cut_filt.cutoff_freq,
            self.nyquist_limit) # upper bound accepted (lower bound is 0)

        if ok:
            self.filter_dict['low_cut'] = \
                self.low_cut_filt.make_filt(float(cutoff_frequency))
            self.update_flag_and_possibly_update_filter()
        return ok


    def high_cut_off(self):
        self.high_cut_in_effect = False
        self.update_flag_and_possibly_update_filter()

    def high_cut_100hz(self):
        self.high_cut_in_effect = True
        self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(100.0)
        self.update_flag_and_possibly_update_filter()

    def high_cut_75hz(self):
        self.high_cut_in_effect = True
        self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(75.0)
        self.update_flag_and_possibly_update_filter()

    def high_cut_50hz(self):
        self.high_cut_in_effect = True
        self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(50.0)
        self.update_flag_and_possibly_update_filter()

    def high_cut_40hz(self):
        self.high_cut_in_effect = True
        self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(40.0)
        self.update_flag_and_possibly_update_filter()

    def high_cut_30hz(self):
        self.high_cut_in_effect = True
        self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(30.0)
        self.update_flag_and_possibly_update_filter()

    def high_cut_20hz(self):
        self.high_cut_in_effect = True
        self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(20.0)
        self.update_flag_and_possibly_update_filter()

    def high_cut_10hz(self):
        self.high_cut_in_effect = True
        self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(10.0)
        self.update_flag_and_possibly_update_filter()

    def high_cut_custom_frequency(self):
        self.high_cut_in_effect = True
        lowest_accepted_value = 0
        cutoff_frequency, ok = self.custom_freq_dialogue(
            self.high_cut_dialogue_title,
            self.high_cut_dialogue_prompt.format(
                lowest_accepted_value,
                self.nyquist_limit),
            self.high_cut_filt.cutoff_freq,
            self.nyquist_limit)

        if ok:
            self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(
                float(cutoff_frequency))
            self.update_flag_and_possibly_update_filter()
        return ok

    def notch_off(self):
        self.notch_in_effect = False
        self.update_flag_and_possibly_update_filter()

    def notch_60hz(self):
        self.notch_in_effect = True
        self.notch_filt.set_center_freq(60.0)
        self.filter_dict['notch'] = self.notch_filt.make_filt()
        self.update_flag_and_possibly_update_filter()

    def notch_50hz(self):
        self.notch_in_effect = True
        self.notch_filt.set_center_freq(50.0)
        self.filter_dict['notch'] = self.notch_filt.make_filt()
        self.update_flag_and_possibly_update_filter()

    def notch_custom_frequency(self):
        self.notch_in_effect = True
        lowest_accepted_value = 0 # declare variable for sake of readability
        cutoff_frequency, ok = self.custom_freq_dialogue(
            self.notch_dialogue_title,
            self.notch_dialogue_prompt.format(
                lowest_accepted_value,
                self.nyquist_limit),
            self.notch_filt.center_freq,
            self.nyquist_limit)
        if ok:
            self.notch_filt.set_center_freq(float(cutoff_frequency))
            self.filter_dict['notch'] = self.notch_filt.make_filt()
            self.update_flag_and_possibly_update_filter()
        return ok

    def rhythms_off(self):

        # we have left rhythm selection 'mode', so if (eventually) we
        # reenter that mode we should be prepared to save the new values
        #
        self.state_values_saved = False

        self.low_cut_in_effect = self.low_cut_used_to_be_in_effect
        self.high_cut_in_effect = self.high_cut_used_to_be_in_effect
        self.low_cut_filt.restore_freq()
        self.high_cut_filt.restore_freq()

        if self.low_cut_in_effect:
            self.filter_dict['low_cut'] = self.low_cut_filt.make_filt()
        if self.high_cut_in_effect:
            self.filter_dict['high_cut'] = self.high_cut_filt.make_filt()

        self.update_flag_and_possibly_update_filter()

    def _save_non_rhythmic_filter_state(rhythm_select_func):

        # if we have just set a rhythm filter, then we should save the
        # current low_cut, high_cut, and notch filter values return
        # when done with rhythm filters (as selected via Rhythms->Off)
        #
        def wrapper_remember_filter(self):

            if not self.state_values_saved:
                self.low_cut_used_to_be_in_effect = self.low_cut_in_effect
                self.high_cut_used_to_be_in_effect = self.high_cut_in_effect
                self.low_cut_filt.save_freq()
                self.high_cut_filt.save_freq()
                self.state_values_saved = True

            rhythm_select_func(self)

        return wrapper_remember_filter

    @_save_non_rhythmic_filter_state
    def rhythms_delta_select(self):

        self.low_cut_in_effect = False
        self.high_cut_in_effect = True
        self.notch_in_effect = True
        self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(self.delta_high)
        self.make_filter()

    @_save_non_rhythmic_filter_state
    def rhythms_theta_select(self):
        self.low_cut_in_effect = True
        self.high_cut_in_effect = True
        self.notch_in_effect = True
        self.filter_dict['high_cut'] = \
            self.high_cut_filt.make_filt(self.theta_high, extra_strength=True)
        self.filter_dict['low_cut'] = \
            self.low_cut_filt.make_filt(self.theta_low, extra_strength=True)
        self.make_filter()

    @_save_non_rhythmic_filter_state
    def rhythms_alpha_select(self):
        self.low_cut_in_effect = True
        self.high_cut_in_effect = True
        self.notch_in_effect = True
        self.filter_dict['high_cut'] = self.high_cut_filt.make_filt(self.alpha_high)
        self.filter_dict['low_cut'] = self.low_cut_filt.make_filt(self.alpha_low)
        self.make_filter()

    @_save_non_rhythmic_filter_state
    def rhythms_beta_select(self):
        self.low_cut_in_effect = True
        self.high_cut_in_effect = True
        self.notch_in_effect = True
        self.filter_dict['high_cut'] = \
            self.high_cut_filt.make_filt(self.beta_high, extra_strength=True)
        self.filter_dict['low_cut'] = \
            self.low_cut_filt.make_filt(self.beta_low, extra_strength=True)
        self.make_filter()

    @_save_non_rhythmic_filter_state
    def rhythms_gamma_select(self):
        self.low_cut_in_effect = True
        self.high_cut_in_effect = False
        self.notch_in_effect = True
        self.filter_dict['low_cut'] = self.low_cut_filt.make_filt(self.gamma_low)
        self.make_filter()

    def set_up_dialogue_strings(self):

        self.low_cut_dialogue_title = \
            "Low cut filter cut off frequency selection"
        self.low_cut_dialogue_prompt = \
            'Please enter a value in Hz between 0 and {0}.'

        self.high_cut_dialogue_title = \
            "High cut filter cut off frequency selection"
        self.high_cut_dialogue_prompt = \
            'Please enter a value in Hz in the range {0} and {1}.'

        self.notch_dialogue_title = \
            "Notch filter frequency selection"
        self.notch_dialogue_prompt = \
            'Please enter a value in Hz in the range {0} and {1}.'
