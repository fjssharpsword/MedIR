from pyqtgraph.Qt import QtGui

#---------------------------------------------------------------------
#
# file: DemoPrefTabAnnotations
#
# this file holds spectrogram preferences widgets to be placed
# into DemoPrefencesWidget's spectrogram tab.
#
# ***This class is very similar to DemoPreferencesTabGeneral,
#        a more comprehensive documentation can be found there.***
#
class DemoPrefTabAnnotations(QtGui.QWidget):
    def __init__(self,
                 dict_roi_a):
        super(DemoPrefTabAnnotations, self).__init__()
        self.layout_grid = QtGui.QGridLayout(self)

        self.dict_roi = dict_roi_a

        self.init_handles_menus()

        self.init_borders_menus()

        self.init_lbls_menus()

        self.is_changed = False

    def init_handles_menus(self):

        self.label_handle_color = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_handle_color, 1, 1, 1, 1)
        self.line_edit_handle_color = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_handle_color, 1, 2, 1, 1)
        self.label_handle_color.setText("Handle Color")
        self.line_edit_handle_color.setText(str(self.dict_roi['pen_handle']))
        self.line_edit_handle_color.textChanged.connect(self.set_changed)

        self.label_handle_size = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_handle_size, 1, 3, 1, 1)
        self.line_edit_handle_size = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_handle_size, 1, 4, 1, 1)
        self.label_handle_size.setText("Handle Size")
        self.line_edit_handle_size.setText(str(self.dict_roi['handle_size']))
        self.line_edit_handle_size.textChanged.connect(self.set_changed)

    def init_borders_menus(self):

        self.label_border_width_default = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_border_width_default, 2, 1, 1, 1)
        self.line_edit_border_width_default = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_border_width_default, 2, 2, 1, 1)
        self.label_border_width_default.setText("Default Border Width")
        self.line_edit_border_width_default.setText(str(
            self.dict_roi['border_width_default']))
        self.line_edit_border_width_default.textChanged.connect(self.set_changed)

        self.label_border_width_selected = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_border_width_selected, 2, 3, 1, 1)
        self.line_edit_border_width_selected = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_border_width_selected, 2, 4, 1, 1)
        self.label_border_width_selected.setText("Selected Border Width")
        self.line_edit_border_width_selected.setText(str(
            self.dict_roi['border_width_selected']))
        self.line_edit_border_width_selected.textChanged.connect(self.set_changed)

    def init_lbls_menus(self):

        self.label_lbl_color = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_lbl_color, 3, 1, 1, 1)
        self.line_edit_lbl_color = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_lbl_color, 3, 2, 1, 1)
        self.label_lbl_color.setText("Label Color")
        self.line_edit_lbl_color.setText(str(self.dict_roi['lbl_color']))
        self.line_edit_lbl_color.textChanged.connect(self.set_changed)

        self.label_lbl_font_size = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_lbl_font_size, 3, 3, 1, 1)
        self.line_edit_lbl_font_size = QtGui.QLineEdit(self)
        self.layout_grid.addWidget(self.line_edit_lbl_font_size, 3, 4, 1, 1)
        self.label_lbl_font_size.setText("Label Font Size")
        self.line_edit_lbl_font_size.setText(str(self.dict_roi['lbl_font_size']))
        self.line_edit_lbl_font_size.textChanged.connect(self.set_changed)

    # method: get_settings
    #
    # arguments: none
    #
    # returns: 
    #  -handle_color_tuple: a tuple converted from a string containing tuple formatting.
    #  -all other returns: text from QLineEdits converted to integers.
    #
    # this method returns values to be connected to DemoEventLoop and further.
    #
    def get_settings(self):
        handle_color_string = str(self.line_edit_handle_color.text())

        # converts passed in values from strings to tuple.
        # removes '(', ')', ' ', and splits each value in string into tuple
        # format by ','
        #
        handle_color_tuple = tuple(map(int, handle_color_string[1:-1].
                                       replace(" ", "").split(",")))

        handle_size = int(self.line_edit_handle_size.text())

        default_border_width = int(self.line_edit_border_width_default.text())

        selected_border_width = int(self.line_edit_border_width_selected.text())

        label_color = int(self.line_edit_lbl_color.text())

        label_font_size = int(self.line_edit_lbl_font_size.text())

        return (handle_color_tuple,
                handle_size,
                default_border_width,
                selected_border_width,
                label_color,
                label_font_size)

    def set_changed(self):
        self.is_changed = True

    def set_unchanged(self):
        self.is_changed = False
