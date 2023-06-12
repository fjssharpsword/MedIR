from pyqtgraph.Qt import QtGui
from .extras.demo_lut_widget import DemoLUTWidget

#---------------------------------------------------------------------
#
# file: DemoPrefTabSpectrogram
#
# this file holds spectrogram preferences widgets to be placed
# into DemoPrefencesWidget's spectrogram tab.
#
# ***This class is very similar to DemoPreferencesTabGeneral,
#        a more comprehensive documentation can be found there.***
#
class DemoPrefTabSpectrogram(QtGui.QWidget):
    def __init__(self,
                 dict_lut_widget_a,
                 dict_spectrogram_a):
        super(DemoPrefTabSpectrogram, self).__init__()

        self.dict_lut_widget = dict_lut_widget_a

        self.dict_spectrogram = dict_spectrogram_a

        self.layout_grid = QtGui.QGridLayout(self)

        self.init_nfft_box()

        self.init_window_type_box()

        self.init_window_size_edit()

        self.init_decimation_factor_line_edit()

        self.lut_widget = DemoLUTWidget(self.dict_lut_widget)

        # the '-1' in the 4th argument allows this widget to span multiple "rows"
        #
        self.layout_grid.addWidget(self.lut_widget, 0, 2, -1, 1)

        self.is_changed = False

    def init_nfft_box(self):
        self.label_nfft = QtGui.QLabel(self)
        self.label_nfft.setText("NFFT")
        self.layout_grid.addWidget(self.label_nfft, 0, 0, 1, 1)

        self.combo_box_nfft = QtGui.QComboBox(self)
        self.combo_box_nfft.setEditable(False)
        self.layout_grid.addWidget(self.combo_box_nfft, 0, 1, 1, 1)
        self.nfft_values = [
            "64",
            "128",
            "256",
            "512",
            "1024",
            "2048",
            "4096"]

        # add values to window type combo box, save index of current
        # default if found
        #
        for index, nfft_value in enumerate(self.nfft_values):
            self.combo_box_nfft.addItem(nfft_value)
            if int(nfft_value) == int((self.dict_spectrogram['nfft'])):
                default_index = index

        # set current default if available
        #
        try:
            self.combo_box_nfft.setCurrentIndex(default_index)
        except UnboundLocalError:
            print ("default nfft value read from preferences does not ", end = ''),
            print ("match any in combobox")
            self.combo_box_nfft.setCurrentIndex(0)

        self.combo_box_nfft.currentIndexChanged.connect(self.set_changed)

    def init_window_type_box(self):
        self.label_window_type = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_window_type, 1, 0, 1, 1)
        self.label_window_type.setText("Window type")

        self.combo_box_window_type = QtGui.QComboBox(self)
        self.combo_box_window_type.setEditable(False)
        self.layout_grid.addWidget(self.combo_box_window_type, 1, 1, 1, 1)
        self.window_types = [
            "Bartlett",
            "Blackman",
            "Hanning",
            "Hamming",
            "Kaiser",
            "Rectangular"]

        # add values to window type combo box, save index of current
        # default if found
        #
        for index, window_string in enumerate(self.window_types):
            self.combo_box_window_type.addItem(window_string)
            if window_string.lower() == self.dict_spectrogram['window_type']:
                default_index = index

        # set current default if available
        #
        try:
            self.combo_box_window_type.setCurrentIndex(default_index)
        except UnboundLocalError:
            print ("default window type read from preferences does not ", end = '')
            print ("match any in combobox")
            self.combo_box_window_type.setCurrentIndex(0)

        self.combo_box_window_type.currentIndexChanged.connect(self.set_changed)

    def init_window_size_edit(self):
        self.label_window_size = QtGui.QLabel(self)
        self.label_window_size.setText("Window Size (sec)")
        self.layout_grid.addWidget(self.label_window_size, 2, 0, 1, 1)

        self.window_size_line_edit = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.window_size_line_edit, 2, 1, 1, 1)
        self.window_size_line_edit.setText(self.dict_spectrogram
                                            ['window_size'])
        self.window_size_line_edit.textChanged.connect(self.set_changed)

    def init_decimation_factor_line_edit(self):
        self.label_decimation_factor = QtGui.QLabel(self)
        self.label_decimation_factor.setText("Decimation Factor")
        self.layout_grid.addWidget(self.label_decimation_factor, 3, 0, 1, 1)

        self.decimation_factor_line_edit = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.decimation_factor_line_edit, 3, 1, 1, 1)
        self.decimation_factor_line_edit.setText(self.dict_spectrogram
                                                  ['decimation_factor'])
        self.decimation_factor_line_edit.textChanged.connect(self.set_changed)

    def get_settings(self):

        nfft = int(self.combo_box_nfft.currentText())

        window_size = float(self.window_size_line_edit.text())

        decimation_factor = float(self.decimation_factor_line_edit.text())

        window_type = str(self.combo_box_window_type.currentText()).lower()

        return (nfft,
                window_size,
                decimation_factor,
                window_type)

    def set_changed(self):
        self.is_changed = True

    def set_unchanged(self):
        self.is_changed = False

