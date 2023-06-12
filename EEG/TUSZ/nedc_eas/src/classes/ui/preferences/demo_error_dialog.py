from pyqtgraph.Qt import QtGui

# strings that will be shown on the QMessageBox
#
# TODO: update for window size param change
message_spectrogram_nfft_zero_padding_error = \
    "Window Size is too large. \nNot enough samples in NFFT."

message_non_int_error = ''' must be an integer.'''

message_negative_value = ''' must be larger than zero.'''
#------------------------------------------------------------------
#
# file: DemoErrorMessageDialog
#
# This file serves as the QMessageBox that will show when a
# user inputtes a bad value in the preferences window.

class DemoErrorMessageDialog(QtGui.QMessageBox):
    def __init__(self):
        super(DemoErrorMessageDialog, self).__init__(
            icon=QtGui.QMessageBox.Warning)
                                                     
        self.setWindowTitle("Error")

    # method: show_nfft_window_size_error_message
    #
    # arguments: None
    #
    # returns: None
    #
    # this message will show when a user inputs a zero padding larger than nfft
    #
    def show_nfft_window_size_error_message(self):
        self.setText(message_spectrogram_nfft_zero_padding_error)
        self.show()

    # method: show_type_error_integer
    #
    # arguments: None
    #
    # returns: None
    #
    # this message will show when a user inputs a value that is not an integer
    #
    def show_type_error_int(self,
                            variable_name_a):
        self.setText(variable_name_a + message_non_int_error)
        self.show()

    # method: show_negative_error_value
    #
    # arguments: None
    #
    # returns: None
    #
    # this message will show when a user inputs a negative value
    #
    def show_negative_error_value(self,
                                  variable_name_a):
        self.setText(variable_name_a + message_negative_value)
        self.show()
