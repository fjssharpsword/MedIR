# general PyQt imports
#
from pyqtgraph.Qt import QtCore, QtGui

# the dynamically generated entries for each result from search
#
from .search_ui_modules.demo_search_entry import DemoSearchEntry

# the various static pieces of the ui
#
from .search_ui_modules.area_results import DemoSearchAreaResults
from .search_ui_modules.area_params import DemoSearchAreaParams
from .search_ui_modules.area_search import DemoSearchAreaSearch

# window_manager is a global object (essentially a glorified list)
# that keeps references to various kinds of windows
# (display_none_found, DemoMainWindow, DemoTextReportWidget)
# until these windows are closed. See the documentation in this class
# for the reasoning, if you care.
# 
from classes.ui.demo_window_manager import window_manager

# class: DemoSearchUserInterface
#
# this class is responsible for:
# - taking the user input regarding a search (text string, checkboxes)
# - capturing the user's intention to search (hitting search or the return key)
# - dynamically displaying the entries of for the search. These entries
#   include buttons to launch text report dialogs, as well as new
#   DemoEventLoops / DemoMainWindows
#
class DemoSearchUserInterface(QtGui.QDialog):

    # emitted when the user presses return or clicks on search button
    # (used to indicate that a search should take place)
    #
    sig_do_search = QtCore.Signal(object)

    # emitted when window is closed
    # (used by window_manager)
    #
    sig_closed = QtCore.Signal(object)

    def __init__(self,
                 function_launch_new_demo_a,
                 function_launch_new_text_window_a):
        super(DemoSearchUserInterface, self).__init__()

        # save the functions necessary to launch new text and signal windows
        #
        self.launch_main_window_generic = function_launch_new_demo_a
        self.launch_text_window_generic = function_launch_new_text_window_a

        self.layout_grid = QtGui.QGridLayout(self)

        # param area - checkboxes
        #
        self.param_area = DemoSearchAreaParams(self)
        self.layout_grid.addWidget(self.param_area, 2, 0, 2, 1)

        # results area - where search entries are displayed,
        #
        self.results_area = DemoSearchAreaResults(self)
        self.layout_grid.addWidget(self.results_area, 3, 2, 1, 1)

        # search area - line edit and search button
        #
        self.search_area = DemoSearchAreaSearch(self)
        self.layout_grid.addWidget(self.search_area, 0, 0, 1, 3)
        self.search_area.button.clicked.connect(self.sig_do_search.emit)

        # show when initialized
        #
        self.show()

        # set initial window size and title
        #
        self.resize(946, 1000)
        self.setWindowTitle("Find EEG Cohorts")

        # inform the window manager about this new window.
        # see comments in that class for reasoning for this, if you care.
        #
        window_manager.manage(self)

    # method: clear_search_results
    #
    # args: none
    #
    # returns: none
    #
    # when a new search is conducted, we need to remove old search queries
    #
    def clear_search_results(self):

        for i in reversed(range(self.results_area.layout_box.count())):

            widget_to_remove = self.results_area.layout_box.itemAt(i).widget()

            # remove it from the layout list
            #
            self.results_area.layout_box.removeWidget(widget_to_remove)

            # remove it from the gui
            #
            widget_to_remove.setParent(None)

    # method: add_entry
    #
    # args:
    #  -edf_path:     path to pertinent edf file
    #  -txt_path:     path to pertinent txt file
    #  -extract_text: text from text file
    #  -edf_name:     name of edf to be displayed in blue
    #
    # return: none, although one could make it return the new entry
    #
    # used to dynamically create search entries, with associated buttons,
    # text, and titles
    #
    def add_entry(self,
                  edf_path,
                  txt_path,
                  extract_text,
                  edf_name):

        # create the new functions for this entries buttons. The
        # button.clicked.connect() methods in cannot take functions with
        # arguments, so we have to make these functions here to feed to them.
        # 
        func_launch_main_window = self.make_launch_loop(edf_path)
        func_launch_text_report = self.make_text_window_function(txt_path)

        # create the entry
        #
        entry = DemoSearchEntry(self,
                                extract_text,
                                edf_name,
                                func_launch_main_window=func_launch_main_window,
                                func_launch_text_report=func_launch_text_report)

        # add the entry to the results area
        #
        self.results_area.layout_box.addWidget(entry)

    # method: make_launch_loop
    #
    # args:
    #  -edf_path_a: a path that to an edf file
    #
    # returns:
    #  -func_launch_specific_main_window: a function that will open the
    #                                     edf_path_a in a new event loop
    #
    # factory method to dynamically generate a function necessary to
    # open a particular edf in the a DemoMainWindow / DemoEventLoop.
    # The button.clicked.connect() method in the buttons in
    # DemoSearchEntry cannot take functions with arguments, so we have
    # to make this function here to feed to them.
    #
    def make_launch_loop(self,
                         edf_path_a):
        def func_launch_specific_main_window():
            self.launch_main_window_generic(file_name=edf_path_a)
        return func_launch_specific_main_window

    # method: make_text_window_function
    #
    # args:
    #  -txt_path_a: a path that to a text report file
    #
    # returns:
    #  -func_launch_specific_text_window: a function that will open the
    #                                     txt_path_a in a report window
    #
    # factory method to dynamically generate a function necessary to
    # open a particular text file in a report window. The
    # button.clicked.connect() method in the buttons in
    # DemoSearchEntry cannot take functions with arguments, so we have
    # to make this function here to feed to them.
    #
    def make_text_window_function(self,
                                  txt_path_a):
        def func_launch_specific_text_window():
            self.launch_text_window_generic(txt_path_a)
        return func_launch_specific_text_window        

    # method: keyPressEvent
    #
    # args:
    #  -event: any key pressed on the keyboard while search window has focus
    #   will go here
    #
    # returns: none
    #
    # reimplementation of keyPressEvent to capture <return> presses
    # allows user to press return rather than hitting the search button
    # emits self.sig_do_search, which will cause DemoSearch to search.
    #
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.sig_do_search.emit(self)

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
