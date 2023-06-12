from pyqtgraph.Qt import QtGui

import os

class DemoMapInfoBar(QtGui.QHBoxLayout):
    def __init__(self, font_a):

        QtGui.QHBoxLayout.__init__(self)

        # get front from DemoMainWindow
        #
        self.font = font_a

        self.spacer = QtGui.QSpacerItem(40,
                                        20,
                                        QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Minimum)
        self.addItem(self.spacer)

        # create, name of map file currently being used
        #
        self.label_map_used = QtGui.QLabel()
        self.addWidget(self.label_map_used)

        self.label_map_used.setText("Map Being Used: None")

    #
    # end of a functionw

    def update_map_used_label(self, map_name_a):
        if map_name_a is not None:
            map_name = os.path.abspath(map_name_a)
            self.label_map_used.setText("Map Being Used: " + map_name)
