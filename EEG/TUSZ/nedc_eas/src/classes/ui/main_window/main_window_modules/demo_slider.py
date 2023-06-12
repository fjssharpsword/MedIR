from pyqtgraph.Qt import QtGui, QtCore

class DemoSlider(QtGui.QSlider):
    def __init__(self,
                 config_dict_a):
        QtGui.QSlider.__init__(self)

        self.setOrientation(QtCore.Qt.Horizontal)

        # TODO: does not seem to have any effect?
        #
        self.setPageStep(int(config_dict_a['initial_time_scale']))

        # sets the frequency of the tickmarks
        #
        self.setTickInterval(int(config_dict_a['initial_time_scale']))

        # sets the beginning and end of where the slider can point to
        # in time when an edf is read in main the slider's maximum
        # value is set via self.update_slider to points to the end of
        # the signal
        #
        self.setMinimum(int(config_dict_a['slider_initial_position']))
        self.setMaximum(int(config_dict_a['total_time_recording']))

        # sets the slider to point to the beginning any signal to be read
        #
        self.setValue(int(config_dict_a['slider_initial_position']))

    def set(self, time_scale_a, total_time_recording_a):
        self.setTickPosition(1)
        self.setTickInterval(time_scale_a)
        self.setMinimumWidth(0)
        self.setMaximum(total_time_recording_a-time_scale_a+1)
        self.setSingleStep(1)
        self.setPageStep(time_scale_a)
