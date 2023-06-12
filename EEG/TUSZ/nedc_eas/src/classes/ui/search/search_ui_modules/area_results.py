from pyqtgraph.Qt import QtGui, QtCore

class DemoSearchAreaResults(QtGui.QScrollArea):
    def __init__(self,
                 parent):
        QtGui.QScrollArea.__init__(self, parent)

        self.setFrameShape(QtGui.QFrame.NoFrame)
        self.setWidgetResizable(True)

        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        self.setPalette(palette)

        self.scroll_area_contents = QtGui.QWidget()
        self.scroll_area_contents.setGeometry(QtCore.QRect(0, 0, 735, 546))
        self.layout_box = QtGui.QVBoxLayout(
            self.scroll_area_contents)

        self.setWidget(self.scroll_area_contents)
        

    def display_none_found(self):
        frame = QtGui.QFrame(self.scroll_area_contents)
        self.layout_box.addWidget(frame)
        layout_box = QtGui.QGridLayout(frame)

        text_area = QtGui.QTextEdit(self.results_area.layout_box)
        layout_box.addWidget(text_area, 0, 0, 1, 3)
        text_area.setFrameShape(QtGui.QFrame.NoFrame)
        text_area.setReadOnly(True)
        text_area.setText("No matches on search criteria")
