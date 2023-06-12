from scipy.signal import firwin

class DemoDetrendFilter:
    def __init__(self,
                 cfg_dict_a,
                 power_level_string_a):
        self.low_cut_freq = float(cfg_dict_a['freqs'][0])
        self.high_cut_freq = float(cfg_dict_a['freqs'][1])

        self.power_dict = {'low': 19,
                           'medium': 39,
                           'high': 59}

        self.set_power(power_level_string_a)

    def set_freqs(self,
                   high_cut_a,
                   low_cut_a):
        self.high_cut_freq = high_cut_a
        self.low_cut_freq = low_cut_a

    def set_power(self,
                  power_string_a):
        self.num_taps = self.power_dict[power_string_a]

    def set_nyquist(self,
                    nyquist_limit_a):
        self.nyquist_limit = nyquist_limit_a

    def make_filt(self):
        bpass_low_Wn = self.low_cut_freq / self.nyquist_limit
        bpass_high_Wn = self.high_cut_freq / self.nyquist_limit

        pass_band = [bpass_low_Wn, bpass_high_Wn]

        # pass_zero argument creates a band-pass filter
        #
        return firwin(self.num_taps,
                      pass_band,
                      pass_zero=False)
