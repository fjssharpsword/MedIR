from pyqtgraph.Qt import QtGui, QtCore

# message to be displayed when help button is pressed
#
HELP_MESSAGE = '''Any unchecked annotation types will not be visible on screen.'''

# class: DemoAnnoTypeSelector
#
# this class inherits from QWidget, it is used by the user to determine
# which annotation types should be plotted to the screen.
#
class DemoAnnoTypeSelector(QtGui.QWidget):

    # signal to be connected to DemoAnnotator, passes dictionary
    # of bools, determining if each checkbox is checked or not
    #
    signal_update_selections=QtCore.Signal(dict)

    # method: __init__:
    #
    # arguments:
    #  -dict_anno_a: dictionary from config file, contains anno type names
    #                 and their corresponding mapping
    #
    # returns: none
    #
    # this method is the constructor, it creates the layouts, buttons,
    # and checkboxes to be added to it's parent widget
    #
    def __init__(self,
                 dict_anno_a):
        super(DemoAnnoTypeSelector, self).__init__()
        self.dict_anno = dict_anno_a

        # layout of parent widget, allows for more complex
        # sublayout configurations
        #
        self.top_level_layout = QtGui.QGridLayout()
        self.setLayout(self.top_level_layout)

        self.checkbox_layout = QtGui.QGridLayout()
        self.top_level_layout.addLayout(self.checkbox_layout, 0, 0)

        self.button_layout = QtGui.QGridLayout()
        self.top_level_layout.addLayout(self.button_layout, 1, 0)

        self.close_button = QtGui.QPushButton("Close", self)
        self.close_button.clicked.connect(self.cancel_pressed)
        self.button_layout.addWidget(self.close_button, 0, 2)

        self.select_button = QtGui.QPushButton("Select All", self)
        self.select_button.clicked.connect(self.select_pressed)
        self.select_button.setCheckable(True)
        self.select_button.setChecked(True)
        self.button_layout.addWidget(self.select_button, 0, 0)
        
        self.help_button = QtGui.QPushButton("Help", self)
        self.help_button.clicked.connect(self.help_pressed)
        self.button_layout.addWidget(self.help_button, 0, 1)

        self.init_type_check_boxes()

    # method: init_type_check_boxes
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is responsible for constructing all check boxes
    #
    def init_type_check_boxes(self):
        
        self.dict_check_boxes = {}
        row = 0
        column = 0

        # iterate over 4 letter code for each anno type
        #
        for type in self.dict_anno:

            # check if type is not null
            #
            if type != 'null':

                check_box = QtGui.QCheckBox(type)
                self.checkbox_layout.addWidget(check_box, row, column)

                # only call method when check box is clicked,
                # not when user selects all
                #
                check_box.clicked.connect(self.update_selections)
                check_box.setChecked(True)

                # store all checkboxes by 4 letter code in dictionary
                #
                self.dict_check_boxes[type] = check_box

                # this logic allows us to format the checkboxes going
                # down a column, until there are 5 checkboxes in it.
                row += 1
                if row == 5:
                    column+=1
                    row = 0

    # method: get_check_box_dict
    #
    # arguments: none
    #
    # returns: a dictionary of the following format
    #          key: 4 letter type code     value: bool
    #
    # this method returns a dict that contains whether or not
    # each checkbox is checked
    #
    def get_check_box_dict(self):
        selected_types = {}
        for type in self.dict_check_boxes:
            is_checked = self.dict_check_boxes[type].isChecked()
            selected_types[type] = is_checked
        return selected_types

    # method: update_selections
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is called when a user clicks a checkbox, it is responible
    # for passing selected_types to DemoAnnotator
    #
    def update_selections(self):
        selected_types = self.get_check_box_dict()
        self.signal_update_selections.emit(selected_types)

        # in the case that a user unchecks a box after selecting all checkboxes,
        # we want to allow the user to select all checkboxes
        #
        if self.select_button.isChecked():
            self.select_button.setChecked(False)

        # this for loop is designed to end the function if any check box is
        # not checked
        #
        for i in selected_types.values():
            if i is False:
                return

        # in the case that a user checks a box, and all other boxes are checked,
        # we want to allow the user to deselect all checkboxes
        #
        self.select_button.setChecked(True)

    def cancel_pressed(self):
        self.hide()

    # method: select_pressed
    #
    # arguments: none
    #
    # returns: none
    #
    # this method deselects, or selects all annotations, depending on
    # if self.select_all is False or True
    #
    def select_pressed(self):

        # when True, we select all boxes
        #
        if self.select_button.isChecked() is True:
            for check_box in self.dict_check_boxes.values():
                check_box.setChecked(True)

        # when False, we deselect all boxes
        #
        else:
            for check_box in self.dict_check_boxes.values():
                check_box.setChecked(False)

        # bypass self.update_selection, as this is a different operation than
        # simply clicking a checkbox
        #
        self.signal_update_selections.emit(self.get_check_box_dict())

    # method: help_pressed
    #
    # arguments: none
    #
    # returns: none
    #
    # this method displays a description of the anno type menu when
    # help is clicked.
    #
    def help_pressed(self):
        help_box = QtGui.QMessageBox(self)
        help_box.setWindowTitle("Help")
        help_box.setText(HELP_MESSAGE)
        help_box.show()    
