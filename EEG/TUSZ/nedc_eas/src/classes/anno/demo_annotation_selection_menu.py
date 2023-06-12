#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/anno/demo_annotation_selection_menu.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph.Qt import QtGui, QtCore
from .demo_channels_selector import DemoChannelsSelector

class DemoAnnotationSelectionMenu(QtGui.QWidget):
    signal_return_ok=QtCore.Signal(int)
    signal_return_no_ok=QtCore.Signal()
    signal_return_remove=QtCore.Signal()
    signal_channel_selector_opened=QtCore.Signal()
    signal_adjust_pressed=QtCore.Signal()

    def __init__(self,
                 parent=None, # currently unused
                 cfg_dict_event_map_a=None):
        super(DemoAnnotationSelectionMenu, self).__init__()

        # create the layout to govern how all buttons are placed in widget
        #
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.dict_event_map = cfg_dict_event_map_a

        # initialize an empty dictionary for annotation buttons
        #
        self.annotation_buttons = {}

        # some numbers to aid in placing buttons on the layout
        #
        row_length = 3
        num_buttons_placed_so_far = row_number = column_number = 0

        # make one colored button for each annotation / color pair
        #
        for annotation_name in self.dict_event_map:

            # get the color from the dictionary
            #
            ann_color = str(self.dict_event_map[annotation_name][2])
            annotation_number = int(self.dict_event_map[annotation_name][0])
            # create the button
            #
            button = DemoAnnotationSelectButton(self,
                                                annotation_name,
                                                annotation_number,
                                                ann_color)

            # connect button to selector function
            #
            button.clicked[bool].connect(self.select_class)

            # add button to the layout
            #
            self.layout.addWidget(button, row_number, column_number)

            # accumulate button in button dictionary
            #
            self.annotation_buttons[annotation_name] = button

            # update the column_number and row_number coordinates
            #
            num_buttons_placed_so_far += 1
            column_number = num_buttons_placed_so_far % row_length
            row_number = num_buttons_placed_so_far / row_length

        # create a frame to make the window look more put together.
        # this is actually unnecessary, and kind of looks ugly
        #
        self.frame = QtGui.QFrame()
        self.frame.setFrameShape(QtGui.QFrame.HLine)
        self.frame.setFrameShadow(QtGui.QFrame.Sunken)
        self.layout.addWidget(self.frame,
                              row_number + 1,
                              0,
                              1,
                              row_length)

        # a number to indicate how far below the annotation buttons to
        # place their ok, remove, and remove buttons
        #
        bottom_button_row_number = row_number + 2

        # create, connect, and add to the layout the ok button
        #
        self.ok_button = QtGui.QPushButton("OK", self)
        self.ok_button.clicked[bool].connect(self.ok_button_pressed)
        self.layout.addWidget(self.ok_button,
                              bottom_button_row_number,
                              0)

        # create, connect, and add to the layout the remove button
        #
        self.remove_button= QtGui.QPushButton("Remove", self)
        self.remove_button.clicked[bool].connect(self.remove_pressed)
        self.layout.addWidget(self.remove_button,
                              bottom_button_row_number,
                              1)
        self.select_channels_button= QtGui.QPushButton("Select Channels", self)
        self.select_channels_button.clicked[bool].connect(
            self.select_channels_pressed)
        self.layout.addWidget(self.select_channels_button,
                              bottom_button_row_number,
                              2)

        self.channel_selector = DemoChannelsSelector()

        self.adjust_button = QtGui.QPushButton("Adjust", self)
        self.adjust_button.clicked[bool].connect(
            self.adjust_annotations_pressed)
        self.layout.addWidget(self.adjust_button, bottom_button_row_number, 2)

        self.deselect_channels = True

    # override the built-in closeEvent to allow for emitting a signal
    #
    def closeEvent(self,
                   event):
        if self.deselect_channels is True:
            self.signal_return_no_ok.emit()
        self.deselect_channels = True
        self.channel_selector.close()
        event.accept()

    # what happens when an annotation button is selected
    # records the annotation type, and then updates the buttons
    #
    def select_class(self):

        # this is kind of mysterious, but it works
        #
        pressed_button = self.sender()

        # record the annotation type to possibly later emit to main
        #
        self.selected_id = pressed_button.annotation_number

        # set all other buttons to unchecked, and then set pressed
        # button to checked
        #
        for button in self.annotation_buttons.values():
            button.setChecked(False)
        pressed_button.setChecked(True)

    # if ok is pressed, then we want to emit a signal to main
    # signaling what type of annotation is selected
    #
    def ok_button_pressed(self):
        try:
            self.signal_return_ok.emit(self.selected_id)
            self.deselect_channels = False
        except:
            pass
        self.close()

    # if remove is pressed, then we want to close the widget without
    # doing anything else
    #
    def remove_pressed(self):
        try:
            self.signal_return_remove.emit()
        except:
            pass
        self.close()

    def select_channels_pressed(self):
        self.channel_selector.show()
        self.signal_channel_selector_opened.emit()

    def adjust_annotations_pressed(self):
        self.deselect_channels = False
        self.signal_adjust_pressed.emit()
        self.close()

# class: DemoAnnotationSelectButton
#
# this class is a very basic inheritance from QtGui.QPushButton, built
# only for the sake of encapsulation
#
class DemoAnnotationSelectButton(QtGui.QPushButton):
    def __init__(self,
                 parent=None,
                 item_name_a=None,
                 item_number_a=None,
                 color_a=None):
        super(QtGui.QPushButton, self).__init__(item_name_a,
                                                parent)
        # set the button name to use later
        #
        self.annotation_name = item_name_a

        self.annotation_number = item_number_a

        # make checkable (so you can click on it)
        #
        self.setCheckable(True)

        # set the color
        #
        self.setStyleSheet("background-color: rgba" + color_a + ";")
