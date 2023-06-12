# -*- coding: utf-8 -*-

from pyqtgraph.Qt import QtGui

from .menus.demo_menu_file import DemoMenuFile
from .menus.demo_menu_eas import DemoMenuEAS
from .menus.demo_menu_view import DemoMenuView
from .menus.demo_menu_filter import DemoMenuFilter
from .menus.demo_menu_help import DemoMenuHelp
from .menus.demo_menu_fft import DemoMenuFFT
from .menus.demo_menu_montage import DemoMenuMontage
from .menus.demo_menu_map import DemoMenuMap


class DemoMenuBar(QtGui.QMenuBar):
    def __init__(self,
                 main_window_parent_a):
        QtGui.QMenuBar.__init__(self,
                                main_window_parent_a)

        # create and name the menues
        #
        self.setNativeMenuBar(False)

        self.menu_eas = DemoMenuEAS(self)
        self.addAction(self.menu_eas.menuAction())
        
        self.menu_file = DemoMenuFile(self)
        self.addAction(self.menu_file.menuAction())

        self.menu_view = DemoMenuView(self)
        self.addAction(self.menu_view.menuAction())

        self.menu_filter = DemoMenuFilter(self)
        self.addAction(self.menu_filter.menuAction())

        self.menu_help = DemoMenuHelp(self)
        self.addAction(self.menu_help.menuAction())

        self.menu_fft = DemoMenuFFT(self)
        self.addAction(self.menu_fft.menuAction())

        self.menu_montage = DemoMenuMontage(self)
        self.addAction(self.menu_montage.menuAction())

        self.menu_map = DemoMenuMap(self)
        self.addAction(self.menu_map.menuAction())

        # disable menu_view / menu_filter until an EDF
        # file has been loaded by the user.
        #
        self.menu_view.setEnabled(False)
        self.menu_filter.setEnabled(False)
        self.menu_fft.setEnabled(False)

    def connect_actions_to_event_loop(self,
                                      event_loop_a):
        # EAS menu
        #
        self.menu_eas.action_about_eas.triggered.connect(
            event_loop_a.ui.about_eas)
        self.menu_eas.action_quit_eas.triggered.connect(
            event_loop_a.ui.quit_eas)
        self.menu_eas.action_preferences.triggered.connect(
            event_loop_a.preferences.widget.show_window)

        # File/Open..Search..Print
        #
        self.menu_file.action_open.triggered.connect(
            event_loop_a.prompt_user_for_open_edf_file)
        self.menu_file.action_search_for_file.triggered.connect(
            event_loop_a.make_demo_search_window)
        self.menu_file.action_print.triggered.connect(
            event_loop_a.print_edf_file)

        # save annotations
        #
        # Save Option
        #
        self.menu_file.action_file_save.triggered.connect(
            lambda: event_loop_a.annotator.write_annotations_to_file("csv"))
        # Save As Option
        #
        self.menu_file.action_file_save_as.triggered.connect(
            lambda: event_loop_a.annotator.save_as_annotations_to_file("csv"))

        # Views/Spectrogram
        #
        self.menu_view.action_spectrogram.triggered.connect(
            event_loop_a.switch_views)

        # Views/Waveform
        #
        self.menu_view.action_waveform.triggered.connect(
            event_loop_a.switch_views)

        # Views/Energy
        #
        self.menu_view.action_energy.triggered.connect(
            event_loop_a.switch_views)

        # Format/Low_Cut/Off..Custom_Frequency
        #
        self.menu_filter.action_low_cut_off.triggered.connect(
            event_loop_a.freq_filter.low_cut_off)
        self.menu_filter.action_low_cut_5hz.triggered.connect(
            event_loop_a.freq_filter.low_cut_5hz)
        self.menu_filter.action_low_cut_10hz.triggered.connect(
            event_loop_a.freq_filter.low_cut_10hz)
        self.menu_filter.action_low_cut_15hz.triggered.connect(
            event_loop_a.freq_filter.low_cut_15hz)
        self.menu_filter.action_low_cut_20hz.triggered.connect(
            event_loop_a.freq_filter.low_cut_20hz)
        self.menu_filter.action_low_cut_25hz.triggered.connect(
            event_loop_a.freq_filter.low_cut_25hz)
        self.menu_filter.action_low_cut_30hz.triggered.connect(
            event_loop_a.freq_filter.low_cut_30hz)
        self.menu_filter.action_low_cut_custom_frequency.triggered.connect(
            event_loop_a.freq_filter.low_cut_custom_frequency)

        # Format/High_Cut/Off..Custom_Frequency
        #
        self.menu_filter.action_high_cut_off.triggered.connect(
            event_loop_a.freq_filter.high_cut_off)
        self.menu_filter.action_high_cut_100hz.triggered.connect(
            event_loop_a.freq_filter.high_cut_100hz)
        self.menu_filter.action_high_cut_75hz.triggered.connect(
            event_loop_a.freq_filter.high_cut_75hz)
        self.menu_filter.action_high_cut_50hz.triggered.connect(
            event_loop_a.freq_filter.high_cut_50hz)
        self.menu_filter.action_high_cut_40hz.triggered.connect(
            event_loop_a.freq_filter.high_cut_40hz)
        self.menu_filter.action_high_cut_30hz.triggered.connect(
            event_loop_a.freq_filter.high_cut_30hz)
        self.menu_filter.action_high_cut_20hz.triggered.connect(
            event_loop_a.freq_filter.high_cut_20hz)
        self.menu_filter.action_high_cut_10hz.triggered.connect(
            event_loop_a.freq_filter.high_cut_10hz)
        self.menu_filter.action_high_cut_custom_frequency.triggered.connect(
            event_loop_a.freq_filter.high_cut_custom_frequency)

        # Format/Notch/Off..Custom_Frequency
        #
        self.menu_filter.action_notch_off.triggered.connect(
            event_loop_a.freq_filter.notch_off)
        self.menu_filter.action_notch_60hz.triggered.connect(
            event_loop_a.freq_filter.notch_60hz)
        self.menu_filter.action_notch_50hz.triggered.connect(
            event_loop_a.freq_filter.notch_50hz)
        self.menu_filter.action_notch_custom_frequency.triggered.connect(
            event_loop_a.freq_filter.notch_custom_frequency)

        # Format/Rhythms/Alpha..Theta..etc.
        #
        self.menu_filter.action_rhythms_off.triggered.connect(
            event_loop_a.freq_filter.rhythms_off)
        self.menu_filter.action_rhythms_delta_select.triggered.connect(
            event_loop_a.freq_filter.rhythms_delta_select)
        self.menu_filter.action_rhythms_theta_select.triggered.connect(
            event_loop_a.freq_filter.rhythms_theta_select)
        self.menu_filter.action_rhythms_alpha_select.triggered.connect(
            event_loop_a.freq_filter.rhythms_alpha_select)
        self.menu_filter.action_rhythms_beta_select.triggered.connect(
            event_loop_a.freq_filter.rhythms_beta_select)
        self.menu_filter.action_rhythms_gamma_select.triggered.connect(
            event_loop_a.freq_filter.rhythms_gamma_select)

        # connect all filter menu actions to eeg_plotter
        #
        for action in self.menu_filter.list_action_filters:
            action.triggered.connect(event_loop_a.eeg_plotter)

        self.menu_fft.action_show.triggered.connect(event_loop_a.show_fft_plot)

        self.menu_montage.action_load.triggered.connect(
            event_loop_a.montage_definer.load_button_pressed)
        self.menu_montage.action_define.triggered.connect(
            event_loop_a.setup_montage_definer)

        # TODO: define the map menu function here  
        #
        self.menu_map.action_load.triggered.connect(event_loop_a.map_definer.load_button_pressed)
