from pyqtgraph.Qt import QtGui, QtCore

# the icons used by the navigation buttons
#
from resources import resource_rc

# the various dropdown menus
#
from .dropdowns.dropdown_channels import DemoChannelsComboBox
from .dropdowns.dropdown_scale import DemoScaleComboBox
from .dropdowns.dropdown_sensitivity import DemoSensitivityComboBox

# the order in which the tools, their respective lavels, and the
# spacers iin between them are added seems to be the determining
# factor in how they are laid out, although ostensibly you should
# be able to set the layout when each of these child widgets are
# added using the overloaded addWidget and addLayout functions
#
class DemoToolBar(QtGui.QHBoxLayout):
    def __init__(self,
                 font_a):
        QtGui.QHBoxLayout.__init__(self)

        # get font from DemoMainWindow
        #
        self.font = font_a

        # set up the label which shows the current position of the
        # cursor when the cursor is on the signal plotting widget
        #
        self.label_cursor = QtGui.QLabel()
        self.addWidget(self.label_cursor)

        # set the text that is displayed in the label
        #
        self.label_cursor.setFont(self.font)
        self.label_cursor.setText("Cursor: 00:00:00")

        # add a spacer between the cursor label and the channel dropdown item
        # 1st argument - width in pixels
        # 2nd argument - height
        # 3rd argument - spacer expands horizontally with window
        # 4th argument - spacer does not expand vertically with window
        #
        self.spacer_1 = QtGui.QSpacerItem(18,
                                          21,
                                          QtGui.QSizePolicy.Expanding,
                                          QtGui.QSizePolicy.Minimum)
        self.addItem(self.spacer_1)

        # create button for channel sensitivities
        #
        self.channels_button = QtGui.QPushButton("Channel Sensitivity")
        self.addWidget(self.channels_button)

        # create, name, and add to the tool bar the sensitivity tool label
        #
        self.label_sensitivity = QtGui.QLabel()
        self.addWidget(self.label_sensitivity)

        # set the text that is displayed in the label
        #
        self.label_sensitivity.setFont(self.font)
        self.label_sensitivity.setText("All Channel Sensitivity")

        # create dropdown for all channel sensitivity
        #
        self.dropdown_sensitivity = DemoSensitivityComboBox()
        self.addWidget(self.dropdown_sensitivity)

        # create, name, and add to the tool_bar the sensitivity units label
        #
        self.label_sensitivity_units = QtGui.QLabel()
        self.label_sensitivity_units.setText("uV/mm")
        self.addWidget(self.label_sensitivity_units)
        
        # create, name, and add to the tool bar the time_scale tool label
        #
        self.label_time_scale = QtGui.QLabel()
        self.addWidget(self.label_time_scale)

        # set the text that is displayed in the label
        #
        self.label_time_scale.setText("        Time Scale")
        self.label_time_scale.setFont(self.font)

        # Initialize dropdown menu from which the user
        # can select time_scale levels. 
        #
        self.dropdown_time_scale = DemoScaleComboBox()
        self.addWidget(self.dropdown_time_scale)

        # create, name, and add to the tool_bar the time_range units lable
        #
        self.label_time_scale_units = QtGui.QLabel()
        self.label_time_scale_units.setText("sec/page")
        self.addWidget(self.label_time_scale_units)

        # add a spacer between the time_scale dropdown item and the
        # annotation navigation buttons
        #
        # 1st argument - width in pixels
        # 2nd argument - height
        # 3rd argument - spacer expands horizontally with window
        # 4th argument - spacer does not expand vertically with window
        #
        self.toolbar_spacer_1 = QtGui.QSpacerItem(18,
                                                  21,
                                                  QtGui.QSizePolicy.Expanding,
                                                  QtGui.QSizePolicy.Minimum)
        self.addItem(self.toolbar_spacer_1)

        anno_button_label = "Annotation Selection"
        self.annotations_button = QtGui.QPushButton(anno_button_label)
        self.annotations_button.setFont(self.font)
        self.addWidget(self.annotations_button)
        # add the push buttons that allow for navigation by annotation
        #
        self.navigation_annotation_init()

    # method: navigation_annotation_init
    #
    # arguments: None
    #
    # return: None
    #
    # this method initializes the four buttons for navigating via annotation.
    #
    # loads each of the annotation navigation icons as pixmaps to make
    # them available to the method navigation_annotation_init
    #
    # here are some arguments to the addPixmap function that I deleted
    # from the old QtDesigner code. I leave them here in case they are
    # necessary for compatibility with other systems. On my system (os x)
    # these arguments to addPixmap seem to do nothing. -EK
    #
    # -> QtGui.QIcon.Normal
    # -> QtGui.QIcon.Off
    #
    def navigation_annotation_init(self):

        # get the icons which will be over-layed on the buttons
        #
        self.icon_goto_first_annotation = QtGui.QIcon()
        self.icon_goto_first_annotation.addPixmap(
            QtGui.QPixmap(":/icons/GoFirstAnnotation.png"))

        self.icon_goto_previous_annotation = QtGui.QIcon()
        self.icon_goto_previous_annotation.addPixmap(
            QtGui.QPixmap(":/icons/GoPreviousAnnotation.png"))

        self.icon_goto_next_annotation = QtGui.QIcon()
        self.icon_goto_next_annotation.addPixmap(
            QtGui.QPixmap(":/icons/GoNextAnnotation.png"))

        self.icon_goto_last_annotation = QtGui.QIcon()
        self.icon_goto_last_annotation.addPixmap(
            QtGui.QPixmap(":/icons/GoLastAnnotation.png"))

        # create all four buttons as QtGui.QPushButtons
        #
        self.push_button_first_annotation = \
            QtGui.QPushButton()
        self.push_button_previous_annotation = \
            QtGui.QPushButton()
        self.push_button_next_annotation = \
            QtGui.QPushButton()
        self.push_button_last_annotation = \
            QtGui.QPushButton()

        # set the icon of each
        #
        self.push_button_first_annotation.setIcon(
            self.icon_goto_first_annotation)
        self.push_button_previous_annotation.setIcon(
            self.icon_goto_previous_annotation)
        self.push_button_next_annotation.setIcon(
            self.icon_goto_next_annotation)
        self.push_button_last_annotation.setIcon(
            self.icon_goto_last_annotation)

        # add each to the layout
        #
        self.addWidget(self.push_button_first_annotation)
        self.addWidget(self.push_button_previous_annotation)
        self.addWidget(self.push_button_next_annotation)
        self.addWidget(self.push_button_last_annotation)
    #
    # end of function
