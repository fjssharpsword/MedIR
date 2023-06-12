#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/config/demo_preferences.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from .demo_config_parser import DemoConfigParser
from classes.ui.preferences.demo_preferences_widget import DemoPreferencesWidget

import os

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: DemoPreferences
#
# This class parses preferences.cfg, as well as serves as a connector from the
# DemoPreferencesWidget to DemoEventLoop. Contains methods that handle the
# logic of the preferences functionality
#
class DemoPreferences(object):

    # method: __init__
    #
    # arguments:
    # - config_file_a: file path to preferences.cfg, file to be parsed
    #
    # returns: none
    #
    # this method initializes the DemoPreferences class, including the
    # DemoPreferencesWidget class. Also connects DemoPreferencesWidget
    # buttons to methods in this class
    #
    def __init__(self,
                 config_file_a):

        self.config_file = config_file_a
        self.parser = DemoConfigParser(config_file_a)

        self.parse_file()

        self.widget = DemoPreferencesWidget(self.dict_lut_widget,
                                            self.dict_main_window,
                                            self.dict_spectrogram,
                                            self.dict_filters_default,
                                            self.dict_detrender,
                                            self.dict_rhythms,
                                            self.dict_roi,
                                            self.dict_waveform,
                                            self.dict_energy,
                                            self.dict_order)

        self.widget.apply_button.clicked.connect(
            self.deal_with_apply_button_clicked)

        self.widget.ok_button.clicked.connect(
            self.deal_with_ok_button_clicked)

        self.widget.cancel_button.clicked.connect(
            self.deal_with_cancel_button_clicked)

        self.widget.save_button.clicked.connect(
            self.deal_with_save_button_clicked)
    #
    # end of method

    # method: setup_updating_functions
    #
    # arguments: event_loop_a
    #
    # returns: none
    #
    # this method allows DemoPreferences to access DemoEventLoop once, in
    # order to connect the DemoPreferences values to DemoEventLoop and beyond.
    #
    def setup_updating_functions(self,
                                 event_loop_a):

        self.update_rhythm_freq_preferences = \
            event_loop_a.freq_filter.set_rhythm_freq_preferences

        self.update_gnrl_filter_power = \
            event_loop_a.freq_filter.set_gnrl_power

        self.update_detrender_range = \
            event_loop_a.freq_filter.set_detrender_range

        self.enable_or_disable_detrend = \
            event_loop_a.freq_filter.enable_or_disable_detrender

        self.eeg_plotter = \
            event_loop_a.eeg_plotter

        self.update_energy = \
            event_loop_a.sigplots_widget.update_energy_preferences

        self.update_spectrogram = \
            event_loop_a.sigplots_widget.update_spectrogram_preferences

        self.update_general = \
            event_loop_a.update_general_preferences

        self.update_annotations = \
            event_loop_a.annotator.update_annotation_preferences

        self.update_waveform = \
            event_loop_a.sigplots_widget.update_waveform_preferences

        self.process_events = event_loop_a.processEvents
    #
    # end of method

    # method: parse_file
    #
    # arguments: None
    #
    # returns: None
    #
    # this method parses preferences.cfg into dicts separated by sections
    #
    def parse_file(self):

        # create some dictionaries by parsing the configuration files
        #
        # to add new dictionaries (assuming that you have already
        # added to the cfg file), simply follow this form:
        #
        # self.<dict name> = self.parser.get_sect_dict(
        #     '<section name>')
        #
        # if you need the dictionary to be a list of tuples, pass the
        # (optional) (boolean) second argument do_tuple as True, as
        # can be seen in the call to create dict_anno_colors_old_style.
        # rgb values in particular will probably want to use this tuple option
        #
        self.dict_main_window     = self.parser.get_sect_dict('MainWindow')
        self.dict_sigplot_misc    = self.parser.get_sect_dict('SignalPlotMisc')
        self.dict_spectrogram     = self.parser.get_sect_dict("Spectrogram")
        self.dict_lut_widget      = self.parser.get_sect_dict("LUTWidget")
        self.dict_filters_default = self.parser.get_sect_dict("FiltersDefault")
        
        self.dict_ann_map        = self.parser.get_sect_dict("AnnEegMap")
        
        self.xml_schema            = self.parser.get_sect_dict("XmlSchema")
        
        self.dict_waveform       = self.parser.get_sect_dict("WaveformPlot",
                                                             do_tuple=True)
        self.dict_energy         = self.parser.get_sect_dict("EnergyPlot",
                                                             do_tuple=True)
        self.dict_rhythms        = self.parser.get_sect_dict('Rhythms',
                                                             do_tuple=True)
        self.dict_detrender      = self.parser.get_sect_dict('Detrender',
                                                             do_tuple=True)
        self.dict_roi            = self.parser.get_sect_dict('ROI',
                                                             do_tuple=True)
      
        self.dict_order          = self.parser.get_sect_dict('OrderOfViews')

        self.dict_montage        = self.parser.get_sect_dict('Montage')
    #
    # end of method

    # give montage file name to DemoPreferences, write to config file only if
    # the user has an overriden config file
    #
    def set_montage(self,
                    montage_file_a):
        self.montage_file = montage_file_a

        if os.path.basename(self.config_file) == ".eas.cfg":
            self.parser.write_montage_to_file(self.montage_file)

    # method: deal_with_apply_button_clicked
    #
    # arguments: None
    #
    # returns: None
    #
    # this method applies changes to DemoEventLoop only if a value
    # in the tab has changed.
    #
    def deal_with_apply_button_clicked(self):

        if self.widget.tab_general.is_changed is True:
            self.change_general_preferences()
            self.widget.tab_general.set_unchanged()

        if self.widget.tab_spectrogram.is_changed is True:
            self.change_spectrogram_preferences()
            self.widget.tab_general.set_unchanged()

        if self.widget.tab_filters.is_changed is True:
            self.change_filter_preferences()
            self.widget.tab_filters.set_unchanged()

        if self.widget.tab_annotations.is_changed is True:
            self.change_annotations_preferences()
            self.widget.tab_annotations.set_unchanged()

        if self.widget.tab_energy.is_changed is True:
            self.change_energy_preferences()
            self.widget.tab_energy.set_unchanged()

        # call these methods to accurately update plots
        #
        self.process_events()
        self.eeg_plotter()
    #
    # end of method

    # method: deal_with_ok_button_clicked
    #
    # arguments: None
    #
    # returns: None
    #
    # this method applies changes to DemoEventLoop only if a value
    # in the tab has changed. Also closes DemoPreferencesWidget
    #
    def deal_with_ok_button_clicked(self):

        if self.widget.tab_general.is_changed is True:
            self.change_general_preferences()
            self.widget.tab_general.set_unchanged()

        if self.widget.tab_spectrogram.is_changed is True:
            self.change_spectrogram_preferences()
            self.widget.tab_general.set_unchanged()

        if self.widget.tab_filters.is_changed is True:
            self.change_filter_preferences()
            self.widget.tab_filters.set_unchanged()

        if self.widget.tab_annotations.is_changed is True:
            self.change_annotations_preferences()
            self.widget.tab_annotations.set_unchanged()

        if self.widget.tab_energy.is_changed is True:
            self.change_energy_preferences()
            self.widget.tab_energy.set_unchanged()

        self.widget.done(0)
        
        # call these methods to accurately update plots
        #
        self.process_events()
        self.eeg_plotter()
    #
    # end of method

    # method: deal_with_cancel_button_clicked
    #
    # arguments: None
    #
    # returns: None
    #
    # this method changes the preferences values back to the
    # preferences.cfg values and closes DemoPreferencesWidget
    #
    def deal_with_cancel_button_clicked(self):

        revert_changes = True

        if self.widget.tab_general.is_changed is True:
            self.change_general_preferences(revert_changes)
            self.widget.tab_general.set_unchanged()

        if self.widget.tab_spectrogram.is_changed is True:
            self.change_spectrogram_preferences(revert_changes)
            self.widget.tab_general.set_unchanged()

        if self.widget.tab_filters.is_changed is True:
            self.change_filter_preferences(revert_changes)
            self.widget.tab_filters.set_unchanged()

        if self.widget.tab_annotations.is_changed is True:
            self.change_annotations_preferences(revert_changes)
            self.widget.tab_annotations.set_unchanged()

        if self.widget.tab_energy.is_changed is True:
            self.change_energy_preferences(revert_changes)
            self.widget.tab_energy.set_unchanged()

        self.widget.done(0)

        # call these methods to accurately update plots
        #
        self.process_events()
        self.eeg_plotter()
    #
    # end of method

    # method: deal_with_save_button_clicked
    #
    # arguments: None
    #
    # returns: None
    #
    # this method applies changes to DemoEventLoop, also writes these
    # changes to preferences.cfg
    #
    def deal_with_save_button_clicked(self):

        self.change_general_preferences()
        self.widget.tab_general.set_unchanged()
        self.parser.set_config_file_from_dict('MainWindow',
                                              self.general_preferences_dict)
        self.parser.set_config_file_from_dict('WaveformPlot',
                                              self.waveform_preferences_dict)
        self.parser.set_config_file_from_dict('OrderOfViews',
                                              self.order_of_views_dict)

        self.change_spectrogram_preferences()
        self.widget.tab_general.set_unchanged()
        self.parser.set_config_file_from_dict('Spectrogram',
                                              self.spectrogram_preferences_dict)

        self.change_filter_preferences()
        self.widget.tab_filters.set_unchanged()
        self.parser.set_config_file_from_dict('FiltersDefault',
                                              self.detrend_enabled_dict)

        self.parser.set_config_file_from_dict('Rhythms',
                                              self.rhythms_dictionary)

        self.parser.set_config_file_from_dict('FiltersDefault',
                                              self.powers_dict)

        self.parser.set_config_file_from_dict('Detrender',
                                                  self.detrender_dict)

        self.change_annotations_preferences()
        self.widget.tab_annotations.set_unchanged()
        self.parser.set_config_file_from_dict('ROI',
                                              self.annotations_preferences_dict)



        self.change_energy_preferences()
        self.widget.tab_energy.set_unchanged()
        self.parser.set_config_file_from_dict('EnergyPlot',
                                              self.energy_preferences_dict)


        
        self.parser.write_config_file()
        self.parser.write_montage_to_file(self.montage_file)

        # call these methods to accurately update plots
        #
        self.process_events()
        self.eeg_plotter()
    #
    # end of method

    # method: change_general_preferences
    #
    # arguments:
    #  - revert_changes_a: optional argument that, when passed, reverts
    #                        changes back to preferences.cfg defaults
    #
    # returns: None
    #
    # this method gets values from classes in directory preference_tabs/ and
    # plugs them into functions connected in setup_updating_functions
    #
    def change_general_preferences(self,
                                   revert_changes_a=False):

        # checks to see if revert_changes_a is passed. if so, revert
        # values back to orignal values
        #
        if revert_changes_a is True:
            time_scale = self.dict_main_window['initial_time_scale']

            sensitivity = self.dict_main_window['initial_sensitivity']

            window_width = int(self.dict_main_window['x_pixels_initial_number'])

            window_height = int(self.dict_main_window['y_pixels_initial_number'])

            signal_color_tuple = self.dict_waveform['signal_color_pen']
            
            view_items_string = self.dict_order['view_items']
        # else get values from line edits in DemoPreferencesTabGeneral
        #
        else:
            time_scale, sensitivity, window_width, window_height, signal_color_tuple, view_items_string = self.widget.tab_general.get_settings()

        # if value is negative, show error dialog and end method without
        # updating DemoEventLoop
        #
        if int(time_scale)<= 0:
            self.widget.error_dialog.show_negative_error_value("Time Scale")
            return

        if int(sensitivity) <= 0:
            self.widget.error_dialog.show_negative_error_value("Sensitivity")
            return

        if window_width <= 0:
            self.widget.error_dialog.show_negative_error_value("Initial Window Width")
            return

        if window_height <= 0:
            self.widget.error_dialog.show_negative_error_value("Initial Window Height")
            return

        # setup dictionary for saving purposes
        #
        self.general_preferences_dict = {'initial_time_scale': time_scale,
                                         'initial_sensitivity': sensitivity,
                                         'x_pixels_initial_number': window_width,
                                         'y_pixels_initial_number': window_height}

        
        self.order_of_views_dict = {'view_items': view_items_string}

        self.waveform_preferences_dict = {'signal_color_pen': signal_color_tuple}

        self.update_waveform(signal_color_tuple)
        
        # call function connected to DemoEventLoop.update_general_preferences
        #
        self.update_general(time_scale,
                            sensitivity,
                            window_width,
                            window_height)
        
        self.widget.tab_general.show_dialog()
    #
    # end of method

    # method: change_spectrogram_preferences
    #
    # arguments:
    #  - revert_changes_a: optional argument that, when passed, reverts
    #                        changes back to preferences.cfg defaults
    #
    # returns: None
    #
    # this method gets values from classes in directory preference_tabs/ and
    # plugs them into functions connected in setup_updating_functions
    #
    def change_spectrogram_preferences(self,
                                       revert_changes_a=False):

        if revert_changes_a is True:
            nfft = int(self.dict_spectrogram['nfft'])

            window_size = float(self.dict_spectrogram['window_size'])

            decimation_factor = float(
                self.dict_spectrogram['decimation_factor'])

            window_type = str(self.dict_spectrogram['window_type']).lower()

        else:
            nfft, window_size, decimation_factor, window_type \
                = self.widget.tab_spectrogram.get_settings()

        if nfft - window_size * self.sampling_rate < 0:
            self.widget.error_dialog.show_nfft_window_size_error_message()
            return

        if window_size < 0:
            self.widget.error_dialog.show_negative_error_value("Window Size")
            return

        if decimation_factor < 0:
            self.widget.error_dialog.show_negative_error_value(
                "Decimation Factor")
            return

        self.spectrogram_preferences_dict = {
            'nfft': nfft,
            'window_size': window_size,
            'decimation_factor': decimation_factor,
            'window_type':  window_type}

        self.update_spectrogram(nfft,
                                window_size,
                                decimation_factor,
                                window_type)
    #
    # end of method

    # TODO: check for error values
    #
    # method: change_filter_preferences
    #
    # arguments:
    #  - revert_changes_a: optional argument that, when passed, reverts
    #                        changes back to preferences.cfg defaults
    #
    # returns: None
    #
    # this method gets values from classes in directory preference_tabs/ and
    # plugs them into functions connected in setup_updating_functions
    #
    def change_filter_preferences(self,
                                  revert_changes_a=False):

        if revert_changes_a is True:

            delta_range_tuple = self.dict_rhythms['delta']

            theta_range_tuple = self.dict_rhythms['theta']

            alpha_range_tuple = self.dict_rhythms['alpha']

            beta_range_tuple = self.dict_rhythms['beta']

            gamma_range_tuple = self.dict_rhythms['gamma']
        else:
            delta_range_tuple, theta_range_tuple, alpha_range_tuple, \
            beta_range_tuple, gamma_range_tuple                      \
                = self.widget.tab_filters.get_rhythm_settings()

        self.rhythms_dictionary = {'delta': delta_range_tuple,
                                   'theta': theta_range_tuple,
                                   'alpha': alpha_range_tuple,
                                   'beta': beta_range_tuple,
                                   'gamma': gamma_range_tuple}

        self.update_rhythm_freq_preferences(self.rhythms_dictionary)

        if revert_changes_a is True:
            low_cut_power = self.dict_filters_default['low_cut_power']

            high_cut_power = self.dict_filters_default['high_cut_power']

            notch_power = self.dict_filters_default['notch_power']

            detrend_power = self.dict_filters_default['detrend_power']

        else:

            low_cut_power, high_cut_power, notch_power, detrend_power \
                = self.widget.tab_filters.get_power_settings()

        self.powers_dict = {'low_cut_power': low_cut_power,
                            'high_cut_power': high_cut_power,
                            'notch_power': notch_power,
                            'detrend_power': detrend_power}

        self.update_gnrl_filter_power(low_cut_power,
                                      high_cut_power,
                                      notch_power,
                                      detrend_power)

        if revert_changes_a is True:
            detrend_lower_bound = self.dict_detrender['freqs'][0]
            detrend_upper_bound = self.dict_detrender['freqs'][1]
            detrend_enabled = self.dict_filters_default['detrend']

        else:

            detrend_enabled, detrend_range_list =  \
                        self.widget.tab_filters.get_detrend_settings()

            detrend_range_list = list(detrend_range_list)
            
            detrend_lower_bound = detrend_range_list[0]
            detrend_upper_bound = detrend_range_list[1]

        detrend_tuple = (detrend_lower_bound, detrend_upper_bound)

        self.detrender_dict = {'freqs': detrend_tuple}
        self.detrend_enabled_dict = {'detrend': detrend_enabled}

        self.enable_or_disable_detrend(detrend_enabled)

        self.update_detrender_range(detrend_lower_bound,
                                    detrend_upper_bound)

        self.eeg_plotter()
    #
    # end of method

    # TODO check for error values
    #
    # method: change_annotations_preferences
    #
    # arguments:
    #  - revert_changes_a: optional argument that, when passed, reverts
    #                        changes back to preferences.cfg defaults
    #
    # returns: None
    #
    # this method gets values from classes in directory preference_tabs/ and
    # plugs them into functions connected in setup_updating_functions
    #
    def change_annotations_preferences(self,
                                       revert_changes_a=False):

        if revert_changes_a is True:
            handle_color_tuple = self.dict_roi['pen_handle']

            handle_size = self.dict_roi['handle_size']

            default_border_width = self.dict_roi['border_width_default']

            selected_border_width = self.dict_roi['border_width_selected']

            label_color = self.dict_roi['lbl_color']

            label_font_size = self.dict_roi['lbl_font_size']

        else:
            handle_color_tuple, handle_size, default_border_width,  \
                selected_border_width, label_color, label_font_size \
                = self.widget.tab_annotations.get_settings()

        self.annotations_preferences_dict = {'pen_handle': handle_color_tuple,
                                             'handle_size': handle_size,
                                             'border_width_default': default_border_width,
                                             'border_width_selected': selected_border_width,
                                             'lbl_color': label_color,
                                             'lbl_font_size': label_font_size}

        self.update_annotations(handle_color_tuple,
                                handle_size,
                                default_border_width,
                                selected_border_width,
                                label_color,
                                label_font_size)
    #
    # end of method

    # TODO: does signal color work? check for error values
    #
    # method: change_energy_preferences
    #
    # arguments:
    #  - revert_changes_a: optional argument that, when passed, reverts
    #                        changes back to preferences.cfg defaults
    #
    # returns: None
    #
    # this method gets values from classes in directory preference_tabs/ and
    # plugs them into functions connected in setup_updating_functions
    #
    def change_energy_preferences(self,
                                  revert_changes_a=False):

        if revert_changes_a is True:
            decimation_factor = self.dict_energy['decimation_factor']
            signal_color_tuple = self.dict_energy['signal_color_pen']
            window_duration = self.dict_energy['window_duration']
            plot_scheme = self.dict_energy['plot_scheme']
            max_value = self.dict_energy['max_value']

        else:
            decimation_factor, signal_color_tuple, window_duration, plot_scheme, \
                max_value = self.widget.tab_energy.get_energy_settings()

        self.energy_preferences_dict = {
            'decimation_factor': decimation_factor,
            'signal_color_pen': signal_color_tuple,
            'window_duration': window_duration,
            'plot_scheme': plot_scheme,
            'max_value': max_value}

        self.update_energy(decimation_factor,
                           window_duration,
                           signal_color_tuple,
                           plot_scheme,
                           max_value)
    #
    # end of method

    # preferences needs to know the sampling rate (read on an initial
    # edf load), so as to give proper error messages when the user
    # provides impossible parameters for spectrogram and energy plots
    #
    def set_sampling_rate(self,
                          sampling_rate_a):
        self.sampling_rate = sampling_rate_a
