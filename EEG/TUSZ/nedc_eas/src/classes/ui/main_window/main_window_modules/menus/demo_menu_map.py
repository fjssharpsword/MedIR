from pyqtgraph.Qt import QtGui

class DemoMenuMap(QtGui.QMenu):
    def __init__(self, menu_bar_parent_a):
        QtGui.QMenu.__init__(self, menu_bar_parent_a)

        self.action_load = QtGui.QAction(self)
        self.action_load.setText("Load Map")
        self.addAction(self.action_load)

        self.setTitle("Map")