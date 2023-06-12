from scipy.signal import firwin

class DemoLowCutFilter:
    def __init__(self,
                 power_string_a='medium',
                 cutoff_a=0,
                 extra_strength_ntaps_increase_a=60):

        self.cutoff_freq = cutoff_a
        self.extra_strength_ntaps_increase = extra_strength_ntaps_increase_a

        self.power_dict = {'low': 21,
                           'medium': 41,
                           'high': 61}
        self.set_power(power_string_a)

    def set_cutoff_freq(self,
                        cutoff_freq_a):
        self.cutoff_freq = cutoff_freq_a

    def set_power(self,
                  power_string_a):
        self.num_taps = self.power_dict[power_string_a]

    def set_nyquist(self,
                    nyquist_limit_a):
        self.nyquist_limit = nyquist_limit_a

    def set_extra_strength_ntaps_increase(self,
                                          extra_strength_ntaps_increase):
        self.extra_strength_ntaps_increase = extra_strength_ntaps_increase

    def make_filt(self,
                  freq=None,
                  extra_strength=False):
        if extra_strength is True:
            number_taps = self.num_taps + self.extra_strength_ntaps_increase
        else:
            number_taps = self.num_taps

        if freq is not None:
            self.cutoff_freq = freq

        Wn = self.cutoff_freq / self.nyquist_limit
        return firwin(number_taps,
                      Wn,
                      pass_zero=False)

    def save_freq(self):
        self.old_freq = self.cutoff_freq

    def restore_freq(self):
        self.cutoff_freq = self.old_freq

