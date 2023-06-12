from pyqtgraph.Qt import QtCore

from pyqtgraph import PlotWidget

class DemoUnusedPlotWidget(PlotWidget):
    def __init__(self):
        PlotWidget.__init__(self)

        # set the size of the unused plot widget so that it holds the
        # slider in its proper place
        #
        self.setMaximumSize(QtCore.QSize(77, 12))

        # hide axes
        #
        self.plotItem.hideAxis('left')
        self.plotItem.hideAxis('bottom')

        self.setBackground((0, 0, 0, 0))
