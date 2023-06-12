from pyqtgraph.Qt import QtGui

class DemoMenuFFT(QtGui.QMenu):
    def __init__(self,
                 menu_bar_parent_a):
        QtGui.QMenu.__init__(self,
                             menu_bar_parent_a)

        self.action_show = QtGui.QAction(self)
        self.action_show.setText("Show")

        self.addAction(self.action_show)

        self.setTitle("FFT")
        
