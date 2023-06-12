from pyqtgraph.Qt import QtGui

class DemoMenuView(QtGui.QMenu):
    def __init__(self,
                 menu_bar_parent_a):
        QtGui.QMenu.__init__(self,
                             menu_bar_parent_a)

        # create and name sub_menu of menu_view, menu_confidence
        #
        self.sub_menu_confidence = QtGui.QMenu(self)
        self.sub_menu_confidence.setEnabled(False)

        self.actions_setup()
        self.actions_connect_and_name()


    def actions_setup(self):
        self.action_labels = QtGui.QAction(self)
        self.action_labels.setCheckable(True)
        self.action_labels.setChecked(True)

        self.action_events = QtGui.QAction(self)
        self.action_events.setCheckable(True)

        self.action_session_info = QtGui.QAction(self)
        self.action_session_info.setEnabled(False)

        self.action_montage = QtGui.QAction(self)
        self.action_montage.setEnabled(False)

        self.action_waveform = QtGui.QAction(self)
        self.action_waveform.setCheckable(True)
        self.action_waveform.setChecked(True)

        self.action_spectrogram = QtGui.QAction(self)
        self.action_spectrogram.setCheckable(True)
        self.action_spectrogram.setEnabled(True)

        self.action_energy = QtGui.QAction(self)
        self.action_energy.setCheckable(True)
        self.action_energy.setEnabled(True)

        self.action_trends = QtGui.QAction(self)
        self.action_trends.setEnabled(False)

        self.action_confidence_low = QtGui.QAction(self)

        self.action_confidence_medium = QtGui.QAction(self)

        self.action_confidence_high = QtGui.QAction(self)
        self.action_confidence_high.setEnabled(True)

    def actions_connect_and_name(self):
        self.sub_menu_confidence.addAction(self.action_confidence_low)
        self.sub_menu_confidence.addAction(self.action_confidence_medium)
        self.sub_menu_confidence.addAction(self.action_confidence_high)
        self.addAction(self.action_labels)
        self.addAction(self.action_events)
        self.addAction(self.action_session_info)
        self.addSeparator()
        self.addAction(self.sub_menu_confidence.menuAction())
        self.addSeparator()
        self.addAction(self.action_montage)
        self.addSeparator()
        self.addAction(self.action_waveform)
        self.addAction(self.action_spectrogram)
        self.addAction(self.action_energy)
        self.addAction(self.action_trends)

        self.setTitle("View")
        self.sub_menu_confidence.setTitle("Confidence")
        self.action_confidence_low.setText("Low")
        self.action_confidence_medium.setText("Medium")
        self.action_confidence_high.setText("High")
        self.action_labels.setText("Labels")
        self.action_events.setText("Events")
        self.action_session_info.setText("Session Info")
        self.action_montage.setText("Montage")
        self.action_waveform.setText("Waveform")
        self.action_spectrogram.setText("Spectrogram")
        self.action_energy.setText("Energy")
        self.action_trends.setText("Trends")

    def get_view_status_dict(self):
        view_status_dict = {}
        view_status_dict['waveform'] = self.action_waveform.isChecked()
        view_status_dict['spectrogram'] = self.action_spectrogram.isChecked()
        view_status_dict['energy'] = self.action_energy.isChecked()

        return view_status_dict
