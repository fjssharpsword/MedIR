from pyqtgraph.Qt import QtGui, QtCore

class DemoSearchAreaSearch(QtGui.QFrame):
    def __init__(self,
                 parent):
        QtGui.QFrame.__init__(self, parent)

        self.layout_box = QtGui.QHBoxLayout(self)

        # line edit
        #
        self.search_line_edit = QtGui.QLineEdit(self)
        self.search_line_edit.setFocus()
        self.layout_box.addWidget(self.search_line_edit)

        # button
        #
        self.button = QtGui.QPushButton(self)
        self.button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.button.setText("Search")
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.button.sizePolicy().hasHeightForWidth())
        self.button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.button.setFont(font)
        self.layout_box.addWidget(self.button)

        # style
        #
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(139, 138, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
