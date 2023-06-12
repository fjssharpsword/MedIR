from pyqtgraph.Qt import QtGui, QtCore

from .entry_modules.demo_search_entry_text_preview import DemoSearchEntryTextPreview

# get some html string constants
#
from .entry_modules.demo_search_entry_html import *

# icons for buttons
#
from resources import resource_rc
REPORT_ICON_PATH_NAME = ":icons/text_report.png";
SIGNAL_ICON_PATH_NAME = ":icons/signal_report.png"

SPACER_WIDTH  = 390
SPACER_HEIGHT = 20

class DemoSearchEntry(QtGui.QFrame):
    def __init__(self,
                 parent,
                 extract_text,
                 edf_name,
                 func_launch_main_window=None,
                 func_launch_text_report=None):

        QtGui.QFrame.__init__(self, parent)

        # the layout used to organize everything for a particular entry
        #
        self.layout_grid = QtGui.QGridLayout(self)

        # use html so our text looks pretty
        #
        self.html_area = DemoSearchEntryTextPreview(self)
        self.layout_grid.addWidget(self.html_area, 0, 0, 1, 3)
        self.html_area.setHtml(
            "<html>" + HTML_HEAD +
            "<span " + EDF_NAME_FONT + edf_name + "</span>"
            "<p "    + EXTRACT_TEXT_STYLE + extract_text + "</p>" + 
            "</body></html>")

        # button to display signal
        #
        self.push_button_disp_signal = DemoPushButtonDispSignal()
        self.layout_grid.addWidget(self.push_button_disp_signal, 1, 1, 1, 1)
        self.push_button_disp_signal.clicked.connect(
            func_launch_main_window)

        # button to display text report
        #
        self.push_button_disp_report = DemoPushButtonDispReport()
        self.layout_grid.addWidget(self.push_button_disp_report, 1, 0, 1, 1)
        self.push_button_disp_report.clicked.connect(
            func_launch_text_report)

        # a spacer to keep the buttons to one side
        #
        self.button_spacer = QtGui.QSpacerItem(SPACER_WIDTH,
                                               SPACER_HEIGHT,
                                               QtGui.QSizePolicy.Expanding,
                                               QtGui.QSizePolicy.Minimum)
        self.layout_grid.addItem(self.button_spacer, 1, 2, 1, 1)

        # trying to prevent the user from being able to scroll throught the report
        # by clicking. Not entirely successful. :(
        #
        self.html_area.verticalScrollBar().triggerAction(
            QtGui.QScrollBar.SliderToMinimum)
        self.html_area.moveCursor(QtGui.QTextCursor.Start)
        

class DemoPushButtonDispReport(QtGui.QPushButton):
    def __init__(self):
        QtGui.QPushButton.__init__(self)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(REPORT_ICON_PATH_NAME))
        self.setIcon(icon)

class DemoPushButtonDispSignal(QtGui.QPushButton):
    def __init__(self):
        QtGui.QPushButton.__init__(self)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(SIGNAL_ICON_PATH_NAME))
        self.setIcon(icon)
