from scipy.signal import firwin

class DemoNotchFilter:
    def __init__(self,
                 power_string_a='medium',
                 center_freq_a=60,
                 stopband_width_a=3):
        self.center_freq = center_freq_a
        self.stopband_width = stopband_width_a

        self.power_dict = {'low': 21,
                           'medium': 41,
                           'high': 61}
        self.set_power(power_string_a)

    def set_center_freq(self,
                        center_freq_a):
        self.center_freq = center_freq_a

    def set_power(self,
                  power_string_a):
        self.num_taps = self.power_dict[power_string_a]

    def set_nyquist(self,
                    nyquist_limit_a):
        self.nyquist_limit = nyquist_limit_a

    def make_filt(self):
                    
        low_Wn = (self.center_freq - self.stopband_width) \
                    / self.nyquist_limit
        high_Wn = (self.center_freq + self.stopband_width) \
                        / self.nyquist_limit

        cutoff_band = [low_Wn, high_Wn]

        return firwin(self.num_taps,
                      cutoff_band)

