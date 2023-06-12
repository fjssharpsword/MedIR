#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/anno/demo_channels_selector.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph.Qt import QtGui, QtCore

class DemoChannelsSelector(QtGui.QWidget):
    signal_return_ok=QtCore.Signal(dict)
    signal_update_channels=QtCore.Signal(dict)
    signal_closed=QtCore.Signal()

    def __init__(self, parent=None):
        super(DemoChannelsSelector, self).__init__()

        # create the layout to govern how all buttons are placed in widget
        #
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)

        self.ok_button = QtGui.QPushButton("OK", self)
        self.ok_button.clicked[bool].connect(self.ok_pressed)
        self.layout.addWidget(self.ok_button,
                              24,
                              0)

        self.waiting_for_user_selection = True

    def ok_pressed(self):
        selected_channels = self.get_check_box_dict()
        self.signal_return_ok.emit(selected_channels)
        self.hide()

    def closeEvent(self,
                   event):
        self.signal_closed.emit()
        self.close()
        event.accept()

    def update_annotations(self):
        if self.waiting_for_user_selection:
            selected_channels = self.get_check_box_dict()
            self.signal_update_channels.emit(selected_channels)
        else:
            pass

    def get_check_box_dict(self):
        selected_channels = {}
        for channel_id in self.dict_check_boxes:
            is_checked = self.dict_check_boxes[channel_id].isChecked()
            selected_channels[channel_id] = is_checked
        return selected_channels

    def set_montage(self,
                    montage_name_a):
        self.montage_names = montage_name_a
        self.dict_check_boxes = {}
        count  = 0
        for name in self.montage_names.values():
            check_box = DemoChannelCheckBox(name, self.update_annotations)
            self.layout.addWidget(check_box, count, 0)
            self.dict_check_boxes[count] = check_box
            count += 1

    def set_channels_selected(self,
                              channel_low_a,
                              channel_high_a):

        self.waiting_for_user_selection = False

        self.uncheck_all_channels()

        for channel_number in range(channel_low_a, channel_high_a):
            checkbox = self.dict_check_boxes[channel_number]
            checkbox.setChecked(True)

        self.waiting_for_user_selection = True

    def uncheck_all_channels(self):
        for checkbox in self.dict_check_boxes.values():
            checkbox.setChecked(False)

class DemoChannelCheckBox(QtGui.QCheckBox):
    def __init__(self,
                 name_a=None,
                 function_signal_changed=None):
        super(DemoChannelCheckBox, self).__init__()
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setText(name_a)
        self.stateChanged.connect(function_signal_changed)
