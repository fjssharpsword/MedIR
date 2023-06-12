from pyqtgraph.Qt import QtGui

#---------------------------------------------------------------------
#
# file: DemoPrefTabEnergy
#
# this file holds spectrogram preferences widgets to be placed
# into DemoPrefencesWidget's spectrogram tab.
#
# ***This class is very similar to DemoPreferencesTabGeneral,
#        a more comprehensive documentation can be found there.***
#
class DemoPrefTabEnergy(QtGui.QWidget):
    def __init__(self,
                 dict_energy_a):

        super(DemoPrefTabEnergy, self).__init__()
        self.layout_grid = QtGui.QGridLayout(self)

        self.dict_energy = dict_energy_a

        self.init_decimation_factor_line_edit()

        self.init_signal_color_line_edit()

        self.init_window_duration_line_edit()

        self.init_plot_scheme_dropdown()

        self.init_max_value_line_edit()

        self.is_changed = False

    def init_decimation_factor_line_edit(self):

        self.label_decimation_factor = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_decimation_factor, 0, 0, 1, 1)
        self.line_edit_decimation_factor = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_decimation_factor, 0, 1, 1, 1)
        self.label_decimation_factor.setText("Decimation Factor")
        self.line_edit_decimation_factor.setText(str(self.dict_energy
                                                   ['decimation_factor']))
        self.line_edit_decimation_factor.textChanged.connect(self.set_changed)

    def init_signal_color_line_edit(self):

        self.label_signal_color = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_signal_color, 1, 0, 1, 1)
        self.line_edit_signal_color = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_signal_color, 1, 1, 1, 1)
        self.label_signal_color.setText('Signal Color')
        self.line_edit_signal_color.setText(str(self.dict_energy['signal_color_pen']))
        self.line_edit_signal_color.textChanged.connect(self.set_changed)

    def init_window_duration_line_edit(self):

        self.label_window_duration = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_window_duration, 2, 0, 1, 1)
        self.line_edit_window_duration = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_window_duration, 2, 1, 1, 1)
        self.label_window_duration.setText('Window Duration (sec)')
        self.line_edit_window_duration.setText(str(self.dict_energy['window_duration']))
        self.line_edit_window_duration.textChanged.connect(self.set_changed)

    def init_plot_scheme_dropdown(self):

        self.label_plot_scheme = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_plot_scheme, 3, 0 ,1, 1)
        self.dropdown_plot_scheme = QtGui.QComboBox(self)
        self.layout_grid.addWidget(self.dropdown_plot_scheme, 3, 1, 1, 1)
        self.label_plot_scheme.setText("Plotting Scheme")
        self.dropdown_plot_scheme.addItem("RMS")
        self.dropdown_plot_scheme.addItem("Logarithmic")
        
        if self.dict_energy['plot_scheme'] == "RMS":
            self.dropdown_plot_scheme.setCurrentIndex(0)
        else:
            self.dropdown_plot_scheme.setCurrentIndex(1)

        self.dropdown_plot_scheme.currentIndexChanged.connect(self.set_changed)

    def init_max_value_line_edit(self):

        self.label_max_value = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_max_value, 4, 0, 1, 1)
        self.line_edit_max_value = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_max_value, 4, 1, 1, 1)
        self.label_max_value.setText("Max Amp. Scale")
        self.line_edit_max_value.setText(str(self.dict_energy['max_value']))
        self.line_edit_max_value.textChanged.connect(self.set_changed)
        
    def get_energy_settings(self):

        decimation_factor = float(self.line_edit_decimation_factor.text())
        signal_color_string = str(self.line_edit_signal_color.text())
        window_duration = float(self.line_edit_window_duration.text())
        max_value = float(self.line_edit_max_value.text())

        # converts passed in values from strings to tuple.
        # removes '(', ')', ' ', and splits each value in string into tuple
        # format by ','
        #
        signal_color_tuple = tuple(map(int, signal_color_string[1:-1]
                                           .replace(" ", "").split(",")))

        plot_scheme = self.dropdown_plot_scheme.currentText()

        return (decimation_factor,
                signal_color_tuple,
                window_duration,
                plot_scheme,
                max_value)

    def set_changed(self):
        self.is_changed = True

    def set_unchanged(self):
        self.is_changed = False
