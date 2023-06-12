from pyqtgraph.Qt import QtGui

#---------------------------------------------------------------------
#
# file: DemoPrefTabFilters
#
# this file holds spectrogram preferences widgets to be placed
# into DemoPrefencesWidget's spectrogram tab.
#
# ***This class is very similar to DemoPreferencesTabGeneral,
#        a more comprehensive documentation can be found there.***
#
class DemoPrefTabFilters(QtGui.QWidget):
    def __init__(self,
                 dict_filters_default_a,
                 dict_detrender_a,
                 dict_rhythms_a):
        super(DemoPrefTabFilters, self).__init__()
        self.layout_grid = QtGui.QGridLayout(self)

        self.dict_filters_default = dict_filters_default_a

        self.dict_detrender = dict_detrender_a

        self.dict_rhythms = dict_rhythms_a

        self.init_detrending_checkbox()

        self.init_detrending_bounds()

        # add a dividing line between first and second pref columns
        #
        dividing_line_1 = QtGui.QFrame(self)
        dividing_line_1.setFrameShape(QtGui.QFrame.VLine)
        dividing_line_1.setFrameShadow(QtGui.QFrame.Sunken)
        self.layout_grid.addWidget(dividing_line_1, 1, 2, 5, 1)

        self.init_filter_powers()

        # add a dividing line between second and third pref columns
        #
        dividing_line_2 = QtGui.QFrame(self)
        dividing_line_2.setFrameShape(QtGui.QFrame.VLine)
        dividing_line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.layout_grid.addWidget(dividing_line_2, 1, 5, 5, 1)

        self.init_delta_rhythm_bounds()

        self.init_theta_rhythm_bounds()

        self.init_alpha_rhythm_bounds()

        self.init_beta_rhythm_bounds()

        self.init_gamma_rhythm_bounds()

        self.is_changed = False

    def init_detrending_checkbox(self):
        self.detrending_checkbox = QtGui.QCheckBox()
        self.detrending_checkbox.setText("Enable Detrending")
        self.layout_grid.addWidget(self.detrending_checkbox, 1, 1, 1, 1)
        self.detrending_checkbox.setChecked(True)
        self.detrending_checkbox.stateChanged.connect(self.set_changed)

    def init_detrending_bounds(self):
        self.label_detrending_bounds = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_detrending_bounds, 2, 1, 1, 1)
        self.line_edit_detrending_bounds = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_detrending_bounds, 3, 1, 1, 1)
        self.label_detrending_bounds.setText("Detrender Range")
        self.line_edit_detrending_bounds.setText(str(self.dict_detrender['freqs']))
        self.line_edit_detrending_bounds.textChanged.connect(self.set_changed)

    def init_filter_powers(self):
        self.filter_powers = ["low", "medium", "high"]

        # low cut combo box
        #
        self.label_low_cut_power = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_low_cut_power, 1, 3, 1, 1)
        self.label_low_cut_power.setText("Low Cut Power")
        self.combo_box_low_cut_power = QtGui.QComboBox(self)
        self.combo_box_low_cut_power.setEditable(False)
        self.layout_grid.addWidget(self.combo_box_low_cut_power, 1, 4, 1, 1)
        self.combo_box_low_cut_power.currentIndexChanged.connect(self.set_changed)

        # add values to low cut power combo box, save index of current
        # default if found
        #
        for index, filter_string in enumerate(self.filter_powers):
            self.combo_box_low_cut_power.addItem(filter_string)
            if filter_string == self.dict_filters_default['low_cut_power']:
                default_index = index

        # set current default if available
        #
        try:
            self.combo_box_low_cut_power.setCurrentIndex(default_index)
        except UnboundLocalError:
            print ("default filter power read from preferences does not ", end = ''),
            print ("match any in combobox")

        # high cut combo box
        #
        self.label_high_cut_power = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_high_cut_power, 2, 3, 1, 1)
        self.label_high_cut_power.setText("High Cut Power")
        self.combo_box_high_cut_power = QtGui.QComboBox(self)
        self.combo_box_high_cut_power.setEditable(False)
        self.layout_grid.addWidget(self.combo_box_high_cut_power, 2, 4, 1, 1)
        self.combo_box_high_cut_power.currentIndexChanged.connect(self.set_changed)

        for index, filter_string in enumerate(self.filter_powers):
            self.combo_box_high_cut_power.addItem(filter_string)
            if filter_string == self.dict_filters_default['high_cut_power']:
                default_index = index

        try:
            self.combo_box_high_cut_power.setCurrentIndex(default_index)
        except UnboundLocalError:
            print ("default filter power read from preferences does not ",end = '')
            print ("match any in combobox")

        # notch combo_box
        #
        self.label_notch_power = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_notch_power, 3, 3, 1, 1)
        self.label_notch_power.setText("Notch Power")
        self.combo_box_notch_power = QtGui.QComboBox(self)
        self.combo_box_notch_power.setEditable(False)
        self.layout_grid.addWidget(self.combo_box_notch_power, 3, 4, 1, 1)
        self.combo_box_notch_power.currentIndexChanged.connect(self.set_changed)

        for index, filter_string in enumerate(self.filter_powers):
            self.combo_box_notch_power.addItem(filter_string)
            if filter_string == self.dict_filters_default['notch_power']:
                default_index = index

        try:
            self.combo_box_notch_power.setCurrentIndex(default_index)
        except UnboundLocalError:
            print ("default filter power read from preferences does not ", end = '')
            print ("match any in combobox")

        # detrend combo box
        #
        self.label_detrend_power = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_detrend_power, 4, 3, 1, 1)
        self.label_detrend_power.setText("Detrend Power")
        self.combo_box_detrend_power = QtGui.QComboBox(self)
        self.combo_box_detrend_power.setEditable(False)
        self.layout_grid.addWidget(self.combo_box_detrend_power, 4, 4, 1, 1)
        self.combo_box_detrend_power.currentIndexChanged.connect(self.set_changed)

        for index, filter_string in enumerate(self.filter_powers):
            self.combo_box_detrend_power.addItem(filter_string)
            if filter_string == self.dict_filters_default['detrend_power']:
                default_index = index

        try:
            self.combo_box_detrend_power.setCurrentIndex(default_index)
        except UnboundLocalError:
            print ("default filter power read from preferences does not ", end = '')
            print ("match any in combobox")

    def init_delta_rhythm_bounds(self):

        self.label_delta_rhythm = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_delta_rhythm, 1, 6, 1, 1)
        self.label_delta_rhythm.setText("Delta Range")
        self.line_edit_delta_rhythm = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_delta_rhythm, 1, 7, 1, 1)
        self.line_edit_delta_rhythm.setText(str(self.dict_rhythms['delta']))
        self.line_edit_delta_rhythm.textChanged.connect(self.set_changed)

    def init_theta_rhythm_bounds(self):

        self.label_theta_rhythm = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_theta_rhythm, 2, 6, 1, 1)
        self.label_theta_rhythm.setText("Theta Range")
        self.line_edit_theta_rhythm = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_theta_rhythm, 2, 7, 1, 1)
        self.line_edit_theta_rhythm.setText(str(self.dict_rhythms['theta']))
        self.line_edit_theta_rhythm.textChanged.connect(self.set_changed)

    def init_alpha_rhythm_bounds(self):

        self.label_alpha_rhythm = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_alpha_rhythm, 3, 6, 1, 1)
        self.label_alpha_rhythm.setText("Alpha Range")
        self.line_edit_alpha_rhythm = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_alpha_rhythm, 3, 7, 1, 1)
        self.line_edit_alpha_rhythm.setText(str(self.dict_rhythms['alpha']))
        self.line_edit_alpha_rhythm.textChanged.connect(self.set_changed)

    def init_beta_rhythm_bounds(self):

        self.label_beta_rhythm = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_beta_rhythm, 4, 6, 1, 1)
        self.label_beta_rhythm.setText("Beta Range")
        self.line_edit_beta_rhythm = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_beta_rhythm, 4, 7, 1, 1)
        self.line_edit_beta_rhythm.setText(str(self.dict_rhythms['beta']))
        self.line_edit_beta_rhythm.textChanged.connect(self.set_changed)

    def init_gamma_rhythm_bounds(self):

        self.label_gamma_rhythm = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_gamma_rhythm, 5, 6, 1, 1)
        self.label_gamma_rhythm.setText("Gamma Range")
        self.line_edit_gamma_rhythm = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_gamma_rhythm, 5, 7, 1, 1)
        self.line_edit_gamma_rhythm.setText(str(self.dict_rhythms['gamma']))
        self.line_edit_gamma_rhythm.textChanged.connect(self.set_changed)

    def get_detrend_settings(self):

        detrend_enabled = self.detrending_checkbox.isChecked()

        detrend_range_string = str(self.line_edit_detrending_bounds.text())

        detrend_range_list =  map(int, detrend_range_string[1:-1].
                                  replace(" ", "").split(','))

        return (detrend_enabled, detrend_range_list)

    def get_rhythm_settings(self):

        delta_range_string = str(self.line_edit_delta_rhythm.text())
        theta_range_string = str(self.line_edit_theta_rhythm.text())
        alpha_range_string = str(self.line_edit_alpha_rhythm.text())
        beta_range_string = str(self.line_edit_beta_rhythm.text())
        gamma_range_string = str(self.line_edit_gamma_rhythm.text())

        # converts passed in values from strings to tuple.
        # removes '(', ')', ' ', and splits each value in string into tuple
        # format by ','
        #
        delta_range_tuple = tuple(map(int, delta_range_string[1:-1]
                                      .replace(" ", "").split(',')))

        theta_range_tuple = tuple(map(int, theta_range_string[1:-1]
                                      .replace(" ", "").split(',')))

        alpha_range_tuple = tuple(map(int, alpha_range_string[1:-1]
                                      .replace(" ", "").split(',')))

        beta_range_tuple = tuple(map(int, beta_range_string[1:-1]
                                     .replace(" ", "").split(',')))

        gamma_range_tuple = tuple(map(int, gamma_range_string[1:-1]
                                      .replace(" ", "").split(',')))

        return (delta_range_tuple,
                theta_range_tuple,
                alpha_range_tuple,
                beta_range_tuple,
                gamma_range_tuple)

    def get_power_settings(self):

        low_cut_power = str(self.combo_box_low_cut_power.currentText())
        high_cut_power = str(self.combo_box_high_cut_power.currentText())
        notch_power = str(self.combo_box_notch_power.currentText())
        detrend_power = str(self.combo_box_detrend_power.currentText())

        return (low_cut_power,
                high_cut_power,
                notch_power,
                detrend_power)

    def set_changed(self):
        self.is_changed = True

    def set_unchanged(self):
        self.is_changed = False
