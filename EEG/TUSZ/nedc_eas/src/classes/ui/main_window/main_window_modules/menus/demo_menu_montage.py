from pyqtgraph.Qt import QtGui

class DemoMenuMontage(QtGui.QMenu):
    def __init__(self,
                 menu_bar_parent_a):
        QtGui.QMenu.__init__(self,
                             menu_bar_parent_a)

        self.action_load = QtGui.QAction(self)
        self.action_load.setText("Load Montage")
        self.addAction(self.action_load)

        self.action_define = QtGui.QAction(self)
        self.action_define.setText("Define New Montage")
        self.addAction(self.action_define)

        self.setTitle("Montage")
