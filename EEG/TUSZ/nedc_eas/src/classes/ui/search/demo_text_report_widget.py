from pyqtgraph.Qt import QtGui, QtCore

# window_manager is a global object (essentially a glorified list)
# that keeps references to various kinds of windows
# (DemoSearchUserInterface, DemoMainWindow, DemoTextReportWidget)
# until these windows are closed. See the documentation in this class
# for the reasoning, if you care.
# 
from classes.ui.demo_window_manager import window_manager

class DemoTextReportWidget(QtGui.QDialog):
    sig_closed = QtCore.Signal(object)
    def __init__(self):
        QtGui.QDialog.__init__(self)

        self.layout_top_level = QtGui.QGridLayout(self)
        self.results_area = QtGui.QScrollArea()
        self.results_area.setFrameShape(QtGui.QFrame.NoFrame)
        self.results_area.setWidgetResizable(True)

        self.scroll_area_contents = QtGui.QWidget()
        self.layout_scroll_area = QtGui.QVBoxLayout(self.scroll_area_contents)

        self.results_area.setWidget(self.scroll_area_contents)
        self.layout_top_level.addWidget(self.results_area, 1, 1, 1, 1)

        self.text_area = QtGui.QTextEdit(self.scroll_area_contents)
        self.layout_scroll_area.addWidget(self.text_area)
        self.text_area.setFrameShape(QtGui.QFrame.NoFrame)
        self.text_area.setReadOnly(True)

        self.resize(600, 500)
        self.show()

        # inform the window manager about this new window.
        # see comments in that class for reasoning for this, if you care.
        #
        window_manager.manage(self)

        
    # method: closeEvent
    #
    # args:
    #  -event: Close events are sent to widgets that the user wants to close, 
    #          usually by choosing "Close" from the window menu, or by
    #          clicking the X title bar button.  will go here
    #
    # returns: none
    #
    # reimplementation of closeEvent so that the window_manager can
    # now that this is closed, and remove its reference to this
    # widget.
    #
    def closeEvent(self,
                   event):
        self.sig_closed.emit(self)
        QtGui.QDialog.closeEvent(self, event)
