#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/dsp/demo_psd_calculator.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
import numpy as np

class DemoPSDCalculator:
    def __init__(self,
                 nfft_a):
        self.nfft = nfft_a

    def set_nfft(self,
                 nfft_a):
        self.nfft = nfft_a

    def compute_one_spectrum(self,
                               window_a):
        # spec is an array whose length is half of nfft
        # (it does not return frequency information above Nyquist limit)
        #
        spec = np.fft.rfft(window_a)

        # get magnitude
        #
        power_spectral_density = abs(spec)

        # convert to dB scale
        #
        power_spectral_density = 20 * np.log10(power_spectral_density)

        return power_spectral_density


    # this method computes the spectrogram by looping over the
    # windows, on each:
    #     -performing the fft algorithm
    #     -converting the result to magnitude
    #     -scaling via the dB rule
    #     -possible auto scaling (probably not?)
    #
    def calc_spectra_array(self,
                           window_list_a):

        num_spectra = len(window_list_a) + 1

        # allocate a new array for the new image.
        # note second element in the tuple argument to np.zeros sizes
        # the array for the maximum possible frequency range given nfft
        #
        full_freq_img_array = np.zeros((num_spectra,
                                        self.nfft // 2 + 1))

        i = 0
        for window in window_list_a:

            full_freq_img_array[i] = self.compute_one_spectrum(window)
            i += 1
        #
        # end of for

        return full_freq_img_array
