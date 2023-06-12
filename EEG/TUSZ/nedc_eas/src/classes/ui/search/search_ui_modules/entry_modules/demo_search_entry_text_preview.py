from pyqtgraph.Qt import QtGui, QtCore

class DemoSearchEntryTextPreview(QtGui.QTextEdit):
    def __init__(self,
                 parent=None):
       QtGui.QTextEdit.__init__(self, parent=parent)
       self.setFrameShape(QtGui.QFrame.NoFrame)
       self.setReadOnly(True)
       self.setMaximumHeight(72)
       self.setVerticalScrollBarPolicy(
           QtCore.Qt.ScrollBarAlwaysOff)

    def wheelEvent(self, e):
        e.ignore();
