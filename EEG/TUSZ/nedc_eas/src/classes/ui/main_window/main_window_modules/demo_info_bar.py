from pyqtgraph.Qt import QtGui

import os

from .dropdowns.dropdown_spectro_low import DemoSpectroLowComboBox
from .dropdowns.dropdown_spectro_high import DemoSpectroHighComboBox

class DemoInfoBar(QtGui.QHBoxLayout):
    def __init__(self,
                 font_a):
        QtGui.QHBoxLayout.__init__(self)

        # get font from DemoMainWindow
        #
        self.font = font_a

        # create, name and the label to display patient name
        #
        self.label_patient_name = QtGui.QLabel()
        self.addWidget(self.label_patient_name)

        # set the initial text for the patient name label
        #
        self.label_patient_name.setFont(self.font)
        self.label_patient_name.setText("Patient:")

        # create, name and add the label to display session date
        #
        self.label_date = QtGui.QLabel()
        self.addWidget(self.label_date)

        # set the initial text for the session date label
        #
        self.label_date.setFont(self.font)
        self.label_date.setText("Date:")

        # create, name, and add label to display session start time
        #
        self.label_start_time = QtGui.QLabel()
        self.addWidget(self.label_start_time)

        # set the initial text for the session start time label
        #
        self.label_start_time.setFont(self.font)
        self.label_start_time.setText("Start Time:")

        # create a spacer and add it between the label_date and
        # the label_start_time items
        #
        # 1st argument - width in pixels
        # 2nd argument - height
        # 3rd argument - spacer expands horizontally with window
        # 4th argument - spacer does not expand vertically with window
        #
        self.spacer = QtGui.QSpacerItem(40,
                                        20,
                                        QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Minimum)
        self.addItem(self.spacer)

        # create, name, and add the freq dropdown label
        #
        self.label_spectro_freq_dropdown = QtGui.QLabel()
        self.label_spectro_freq_dropdown.setText(
            "Spectrogram freq. range (Hz):")
        self.label_spectro_freq_dropdown.setFont(self.font)
        self.addWidget(self.label_spectro_freq_dropdown)

        self.dropdown_spectro_low = DemoSpectroLowComboBox()
        self.addWidget(self.dropdown_spectro_low)

        self.dropdown_spectro_high = DemoSpectroHighComboBox()
        self.addWidget(self.dropdown_spectro_high)

        # initially set frequency selection dropdowns to not be visible
        # (they will be set to visible upon user selection of spectrogram view)
        #
        self.label_spectro_freq_dropdown.setVisible(False)
        self.dropdown_spectro_low.setVisible(False)
        self.dropdown_spectro_high.setVisible(False)

        self.label_montage_used = QtGui.QLabel()
        self.addWidget(self.label_montage_used)

        self.label_montage_used.setText("Montage Being Used: None")
    #
    # end of function

    def update_montage_used_label(self,
                                  montage_name_a):
        if montage_name_a is not None:
            montage_name = os.path.abspath(montage_name_a)
            self.label_montage_used.setText("Montage Being Used: " + montage_name)
