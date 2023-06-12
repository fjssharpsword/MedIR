#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/sigplots/demo_sigplots_widget.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph.Qt import QtGui, QtCore

from .demo_sigplots_page_only_waveform import DemoSigplotsPageOnlyWaveform
from .demo_sigplots_page_mixed_view import DemoSigplotsPageMixedView


# TODO: make a set method for the following handles both pages requirements:
#         self.montage_names = montage_names_a
#         self.number_signals = number_channels_a
#

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: DemoSigplotsWidget
#
# THIS INHERITS FROM STACKED WIDGET CLASS
# should be passed 'frame' widget at init
#
class DemoSigplotsWidget(QtGui.QStackedWidget):

    def __init__(self,
                 parent=None,
                 y_max_a=None,
                 time_scale_a=None,
                 number_signals_a=None,
                 montage_names_a=None,
                 cfg_dict_sigplot_misc_a=None,
                 cfg_dict_waveform_a=None,
                 cfg_dict_spectrogram_a=None,
                 cfg_dict_energy_a=None,
                 dict_order_a=None):

        # initialize as child of parent widget (presumably Main.frame
        #
        QtGui.QStackedWidget.__init__(self, parent=parent)

        # set some internal data based on args to init function
        #
        self.time_scale = time_scale_a
        self.y_max = y_max_a

        # setup some configuration dictionaries
        #
        self.cfg_dict_sigplot_misc = cfg_dict_sigplot_misc_a
        self.cfg_dict_waveform = cfg_dict_waveform_a

        # initialize classic montage view page and add the waveform-only
        # montage view to the stacked widget from which this class inherits
        #
        self.page_only_waveform = DemoSigplotsPageOnlyWaveform(
            y_max_a=self.y_max,
            cfg_dict_a=self.cfg_dict_waveform,
            initial_time_scale_a=self.time_scale)

        self.addWidget(self.page_only_waveform)

        # initialize the page which supports mixed views and add the
        # montage view to the stacked widget from which this class
        # inherits
        #
        self.page_mixed_view = DemoSigplotsPageMixedView(
            montage_names_a,
            number_signals_a,
            time_scale_a,
            cfg_dict_waveform_a,
            cfg_dict_spectrogram_a,
            cfg_dict_energy_a,
            dict_order_a)
        self.addWidget(self.page_mixed_view)

        # initially we will be in the waveform-only view, where
        # current index is 0
        #
        self.setCurrentIndex(0)

        self.view_status_dict = {}
        self.view_status_dict['waveform'] = True
        self.view_status_dict['spectrogram'] = False
        self.view_status_dict['energy'] = False
        self.special_case_flag = True

        self.set_montage(number_signals_a,
                         montage_names_a)

    def set_montage(self,
                    number_sigs_a,
                    montage_names_a):

        self.montage_names = montage_names_a
        self.number_signals = number_sigs_a

        self.page_only_waveform.do_label(number_sigs_a,
                                         montage_names_a)
        
    # method: switch_views
    #
    # arguments: view_status_dict_a: dict containing bools for each view type
    #
    # return: None
    #
    # This method is called through the menu options, and it is used to switch
    # between combinations of the three available views: spectrogram, waveform,
    # and energy.
    #
    def switch_views(self, view_status_dict_a=None):

        # this is later used to limit computations (ie unnecessary fft)
        # and limit unnecessary plotting
        #
        self.view_status_dict = view_status_dict_a

        # check if we are dealing with waveform-only view, which is a special
        # case in which the waveforms are allowed to share screen real estate
        #
        self.special_case_flag = (self.view_status_dict['waveform']
                                  and not self.view_status_dict['spectrogram']
                                  and not self.view_status_dict['energy'])

        # waveform-only view (special case)
        #
        if self.special_case_flag is True:

            # update stacked (parent) widget
            #
            self.setCurrentIndex(0)
            self.page_only_waveform.do_plot()

        # mixed view (standard case)
        #
        else:

            # update stacked (parent) widget
            #
            self.setCurrentIndex(1)

            # connect argument dictionary with combinatorial logic of
            # show / hide cases
            #
            self.page_mixed_view.show_or_hide_waveform_plots(
                show=view_status_dict_a['waveform'])
            self.page_mixed_view.show_or_hide_spectrogram(
                show=view_status_dict_a['spectrogram'])
            self.page_mixed_view.show_or_hide_energy_plots(
                show=view_status_dict_a['energy'])
    #
    # end of function

    # method: gnrl_set_windowing_info
    #
    # arguments:
    #  -slider_current_pos_a: where the slider on the bottom of the
    #    main window is
    #  -time_scale_a: the "zoom" level set at the main window level
    #
    # returns: None
    #
    # this method updates the time range of each signal plot.
    #
    # the plots that are of class PlotItem use that classes
    # setXRange() methods, and are listed here:
    #  -page_only_waveform_signal_plot
    #  -the plots listed in dict_mixed_view_sigplots_waveform
    #
    def gnrl_set_windowing_info(self,
                                slider_current_position_a,
                                time_scale_a):

        self.slider_current_position = slider_current_position_a
        self.time_scale = time_scale_a

        # update the plot on page_only_waveform_view
        # according to the slider position and timescale
        #
        self.page_only_waveform.signal_plot.setXRange(
            self.slider_current_position,
            self.slider_current_position + self.time_scale,
            padding=0)

        # update waveform plots on mixed view page
        # according to the slider position and timescale
        #
        for plot in self.page_mixed_view.dict_waveform_plots.values():
            plot.setXRange(self.slider_current_position,
                           self.slider_current_position + self.time_scale,
                           padding=0)
        #
        # end of for

        # get the left index of the numpy array t_data which corresponds to the
        # slider_current_position (this value is true across all sigplots). 
        # here, we are just getting one, because that is all that we need
        #
        example_plot = next(iter(self.page_mixed_view.dict_waveform_plots.values()))
        left_index = example_plot.t_data.tolist(). \
                             index(self.slider_current_position)

        #left_index = self.slider_current_position
        # update spectrograms according to the slider position and timescale
        #
        for plot in self.page_mixed_view.dict_spectrograms.values():

            # update plot
            #
            plot.set_x_range(left_index,
                             self.time_scale)
        #
        # end of for

        # update energy plots according to the slider position and timescale
        #
        for plot in self.page_mixed_view.dict_energy_plots.values():

            # update plot
            #
            plot.set_x_range(left_index,
                             self.time_scale)

            plot.setXRange(self.slider_current_position,
                           self.slider_current_position + self.time_scale,
                           padding=0)
        #
        # end of for
    #
    # end of function

    # TODO: finish and comment this function
    # perhaps TODO: clean up sensitivity logic
    #
    def gnrl_set_signal_data(self,
                             t_data_a,
                             y_data_a):

        comp_y_data = [self.sensitivity_scale[i] * y_data_a[i] \
                       for i in range(self.number_signals)]

        self.page_only_waveform.set_signal_data(t_data_a,
                                                comp_y_data)

        i = 0
        for channel in self.page_mixed_view.dict_channels.values():
            channel.set_signal_data(t_data_a,
                                    comp_y_data[i])
            i += 1
    #
    # end of function

    # TODO: comment this function
    #
    def gnrl_set_sensitivity(self, sensitivity_scale_a):

        self.sensitivity_scale = sensitivity_scale_a
    #
    # end of function

    # method: update_all_plots
    #
    # arguments: None
    #
    # returns: None
    #
    # this method calls various functions to keep the plots up to date
    #
    def update_all_plots(self):

        # if only waveform view (ie page 1)
        #
        if self.special_case_flag:
            self.page_only_waveform.do_plot()

        # iterate over dictionary of waveform plots set up at the
        # creation of the channel widgets
        #

        if self.view_status_dict['waveform'] and not self.special_case_flag:
            for plot in self.page_mixed_view.dict_waveform_plots.values():
                plot.do_plot()

        if self.view_status_dict['spectrogram']:
            for plot in self.page_mixed_view.dict_spectrograms.values():
                plot.do_plot()

        if self.view_status_dict['energy']:
            for plot in self.page_mixed_view.dict_energy_plots.values():
                plot.do_plot()

        if not self.special_case_flag:
            for splitter in self.page_mixed_view. \
                dict_channel_splitters.values():
                splitter.refresh()
    #
    # end of function

    def update_spectrogram_preferences(self,
                                       nfft_a,
                                       window_size_a,
                                       ratio_a,
                                       window_a):
        for spectrogram in self.page_mixed_view.dict_spectrograms.values():
            spectrogram.update_preferences(nfft_a,
                                           window_size_a,
                                           ratio_a,
                                           window_a)

    def update_waveform_preferences(self,
                                    signal_color_tuple):
        for waveform in self.page_mixed_view.dict_waveform_plots.values():
            waveform.update_preferences(signal_color_tuple)
            
        self.page_only_waveform.update_preferences(signal_color_tuple)

    def update_energy_preferences(self,
                                  ratio_a,
                                  window_length_a,
                                  color_pen_a,
                                  plot_scheme_a,
                                  max_value_a):

        for energy_plot in self.page_mixed_view.dict_energy_plots.values():
            energy_plot.update_preferences(ratio_a,
                                           window_length_a,
                                           color_pen_a,
                                           plot_scheme_a,
                                           max_value_a)
