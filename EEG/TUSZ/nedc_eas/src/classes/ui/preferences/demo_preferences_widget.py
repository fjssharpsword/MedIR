from pyqtgraph.Qt import QtGui

from .pref_tabs.demo_pref_tab_annotations import DemoPrefTabAnnotations
from .pref_tabs.demo_pref_tab_filters import DemoPrefTabFilters
from .pref_tabs.demo_pref_tab_general import DemoPrefTabGeneral
from .pref_tabs.demo_pref_tab_spectrogram import DemoPrefTabSpectrogram
from .pref_tabs.demo_pref_tab_energy import DemoPrefTabEnergy

from .demo_error_dialog import DemoErrorMessageDialog

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: DemoPreferencesWidget
#
# This class serves as the main QDialog for preferences functionality.
# It holds all the prefernces_tabs/ as well as the preferences buttons.
#
class DemoPreferencesWidget(QtGui.QDialog):

    # method: __init__
    #
    # arguments:
    #  - dict_*: these dictionaries are passed in from DemoPreferences and
    #             will be immediately passed to their respective tabs
    #
    # returns: None
    #
    # this method initializes DemoPreferencesWidget, as well as calls
    # other init methods
    #
    def __init__(self,
                 dict_lut_widget_a,
                 dict_main_window_a,
                 dict_spectrogram_a,
                 dict_filters_default_a,
                 dict_detrender_a,
                 dict_rhythms_a,
                 dict_roi_a,
                 dict_waveform_a,
                 dict_energy_a,
                 dict_order_a):

        super(DemoPreferencesWidget, self).__init__()

        self.dict_lut_widget = dict_lut_widget_a

        self.dict_main_window = dict_main_window_a

        self.dict_spectrogram = dict_spectrogram_a

        self.dict_filters_default = dict_filters_default_a

        self.dict_detrender = dict_detrender_a

        self.dict_rhythms = dict_rhythms_a

        self.dict_roi = dict_roi_a

        self.dict_waveform = dict_waveform_a

        self.dict_energy = dict_energy_a

        self.dict_order = dict_order_a

        self.init_top_level()

        self.init_tabs()

        self.tab_widget.setCurrentIndex(0)

        self.error_dialog = DemoErrorMessageDialog()
    #
    # end of method

    # method: init_top_level
    #
    # arguments: None
    #
    # return: None
    #
    # this method creates a QGridLayout used to store the tabs and buttons,
    # also creates buttons
    #
    def init_top_level(self):
        self.setWindowTitle("Preferences")

        self.top_level_layout = QtGui.QGridLayout(self)
        self.tab_widget = QtGui.QTabWidget(self)
        self.tab_widget.setTabShape(QtGui.QTabWidget.Rounded)
        self.top_level_layout.addWidget(self.tab_widget, 1, 1, 1, 5)

        self.ok_button = QtGui.QPushButton("OK", self)
        self.top_level_layout.addWidget(self.ok_button, 2, 5, 1, 1)

        self.apply_button = QtGui.QPushButton("Apply", self)
        self.top_level_layout.addWidget(self.apply_button, 2, 1, 1, 1)

        self.cancel_button = QtGui.QPushButton("Cancel", self)
        self.top_level_layout.addWidget(self.cancel_button, 2, 4, 1, 1)

        self.save_button = QtGui.QPushButton("Save", self)
        self.top_level_layout.addWidget(self.save_button, 2, 2, 1, 1)
    #
    # end of method

    # method: init_tabs
    #
    # arguments: None
    #
    # return: None
    #
    # this method initializes all tabs and adds them to the QGridLayout
    #
    def init_tabs(self):

        self.tab_general = DemoPrefTabGeneral(self.dict_main_window,
                                              self.dict_order,
                                              self.dict_waveform)
        self.tab_widget.addTab(self.tab_general, "General")

        self.tab_filters = DemoPrefTabFilters(self.dict_filters_default,
                                              self.dict_detrender,
                                              self.dict_rhythms)

        self.tab_widget.addTab(self.tab_filters, "Filters")

        self.tab_annotations = DemoPrefTabAnnotations(self.dict_roi)
        self.tab_widget.addTab(self.tab_annotations, "Annotations")

        self.tab_spectrogram = DemoPrefTabSpectrogram(self.dict_lut_widget,
                                                      self.dict_spectrogram)
        self.tab_widget.addTab(self.tab_spectrogram, "Spectrogram")

        self.tab_energy = DemoPrefTabEnergy(self.dict_energy)
        self.tab_widget.addTab(self.tab_energy, "Energy")

        #self.tab_montage = DemoPrefTabMontage()
        #self.tab_widget.addTab(self.tab_montage, "Montage")
    #
    # end of method

    def show_window(self):
        self.show()
        self.activateWindow()
        self.raise_()
        
