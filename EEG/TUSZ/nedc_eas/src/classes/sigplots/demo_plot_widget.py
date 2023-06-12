#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/sigplots/demo_plot_widget.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
# this file was created to fix a bug fixed in a newer version of pyqtgraph
# https://github.com/pyqtgraph/pyqtgraph/commit/9df4df55c49ace05020e6b7c52128fdce09e2a8d
# search for 'extra-axis' to find location of change
#
# in the interests of only using one version of pyqtgraph this fix was
# ported into our codebase via this file
#
# this file also affords us more control over the initialization of
# the PlotWidgets / PlotItems
#
# for example, the 'letter A' bug could be fixed from higher up in the
# hierarchy, but here we can address it in the constructor of
# DemoPlotItem itself
#
# EK: September 1, 2016: this could use a critical eye, comments, and cleanup
#                        it works, but its sloppy
#                        much code was copy-pasted, reformatted
#
# TODO: delete addItem method, allowing it to use the built in method
#
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import weakref
import numpy as np


from pyqtgraph.graphicsItems.PlotItem import plotConfigTemplate_pyqt5 \
    as ConfigTemplate


class DemoPlotWidget(pg.PlotWidget):
    def __init__(self,
                 parent=None,
                 background=None,
                 cfg_dict_a=None,
                 stylesheet_a=None,
                 time_scale_a=None,
                 **kwargs):
        """
        When initializing PlotWidget, *parent* and *background* are passed
        to :func:`GraphicsWidget.__init__()
        <pyqtgraph.GraphicsWidget.__init__>` and all others are passed
        to :func:`PlotItem.__init__()
        <pyqtgraph.PlotItem.__init__>`.
        """
        pg.GraphicsView.__init__(self,
                                 parent,
                                 background=background)
        self.setSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.plotItem = DemoPlotItem(**kwargs)
        self.setCentralItem(self.plotItem)


        # Explicitly wrap methods from plotItem
        #
        # NOTE: If you change this list, update the documentation
        # above as well.
        #
        for m in ['addItem',
                  'removeItem',
                  'autoRange',
                  'clear',
                  'setXRange',
                  'setYRange',
                  'setRange',
                  'setAspectLocked',
                  'setMouseEnabled',
                  'setXLink',
                  'setYLink',
                  'enableAutoRange',
                  'disableAutoRange',
                  'setLimits',
                  'register',
                  'unregister',
                  'viewRect']:
            setattr(self, m, getattr(self.plotItem, m))

        self.plotItem.sigRangeChanged.connect(self.viewRangeChanged)
        self.sig_mouse_moved = self.scene().sigMouseMoved
        self.sigMouseClicked = self.scene().sigMouseClicked

        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

    def get_mouse_secs_if_in_plot(self,
                                  event_a):
        if self.plotItem.sceneBoundingRect().contains(event_a):
            return self.plotItem.vb.mapSceneToView(event_a).x()

class DemoPlotItem(pg.PlotItem):
    """
    **Bases:** :class:`GraphicsWidget <pyqtgraph.GraphicsWidget>`

    Plot graphics item that can be added to any graphics
    scene. Implements axes, titles, and interactive viewbox.
    PlotItem also provides some basic analysis functionality that
    may be accessed from the context menu.  Use :func:`plot()
    <pyqtgraph.PlotItem.plot>` to create a new PlotDataItem and
    add it to the view.  Use :func:`addItem()
    <pyqtgraph.PlotItem.addItem>` to add any QGraphicsItem to the
    view.

    This class wraps several methods from its internal ViewBox:
        :func:`setXRange <pyqtgraph.ViewBox.setXRange>`,
        :func:`setYRange <pyqtgraph.ViewBox.setYRange>`,
        :func:`setRange <pyqtgraph.ViewBox.setRange>`,
        :func:`autoRange <pyqtgraph.ViewBox.autoRange>`,
        :func:`setXLink <pyqtgraph.ViewBox.setXLink>`,
        :func:`setYLink <pyqtgraph.ViewBox.setYLink>`,
        :func:`setAutoPan <pyqtgraph.ViewBox.setAutoPan>`,
        :func:`setAutoVisible <pyqtgraph.ViewBox.setAutoVisible>`,
        :func:`setLimits <pyqtgraph.ViewBox.setLimits>`,
        :func:`viewRect <pyqtgraph.ViewBox.viewRect>`,
        :func:`viewRange <pyqtgraph.ViewBox.viewRange>`,
        :func:`setMouseEnabled <pyqtgraph.ViewBox.setMouseEnabled>`,
        :func:`enableAutoRange <pyqtgraph.ViewBox.enableAutoRange>`,
        :func:`disableAutoRange <pyqtgraph.ViewBox.disableAutoRange>`,
        :func:`setAspectLocked <pyqtgraph.ViewBox.setAspectLocked>`,
        :func:`invertY <pyqtgraph.ViewBox.invertY>`,
        :func:`invertX <pyqtgraph.ViewBox.invertX>`,
        :func:`register <pyqtgraph.ViewBox.register>`,
        :func:`unregister <pyqtgraph.ViewBox.unregister>`

    The ViewBox itself can be accessed by calling
    :func:`getViewBox() <pyqtgraph.PlotItem.getViewBox>`

    **Signals:**
    sigYRangeChanged     wrapped from :class:`ViewBox <pyqtgraph.ViewBox>`
    sigXRangeChanged     wrapped from :class:`ViewBox <pyqtgraph.ViewBox>`
    sigRangeChanged      wrapped from :class:`ViewBox <pyqtgraph.ViewBox>`
        """

    # Emitted when the ViewBox range has changed
    #
    sigRangeChanged = QtCore.Signal(object, object)

    # Emitted when the ViewBox Y range has changed
    #
    sigYRangeChanged = QtCore.Signal(object, object)

    # Emitted when the ViewBox X range has changed
    #
    sigXRangeChanged = QtCore.Signal(object, object)

    lastFileDir = None

    def __init__(self,
                 cfg_dict_a=None,
                 stylesheet_a=None,
                 time_scale_a=None,
                 parent=None,
                 viewBox=None,
                 name=None,
                 labels=None,
                 title=None,
                 axisItems=None,
                 enableMenu=True,
                 *args,
                 **kwargs):

        """
        Create a new PlotItem. All arguments are optional.
        Any extra keyword arguments are passed to PlotItem.plot().

        **Arguments:**
        *title*      Title to display at the top of the item. Html is allowed.
        *labels*     A dictionary specifying the axis labels to display:
                        {'left': (args), 'bottom': (args), ...}
                     The name of each axis and the corresponding arguments are
                     passed to
                      :func:`PlotItem.setLabel() <pyqtgraph.PlotItem.setLabel>`
                     Optionally, PlotItem my also be initialized with the
                     keyword arguments left,right, top, or bottom to achieve
                     the same effect.

        *name*       Registers a name for this view so that others
                     may link to it
        *viewBox*    If specified, the PlotItem will be constructed with this
                     as its ViewBox.
        *axisItems*  Optional dictionary instructing the PlotItem to use
                     pre-constructed items for its axes.
                     The dict keys must be axis names:
                        ('left', 'bottom', 'right', 'top')
                     and the values must be instances of AxisItem
                     (or at least compatible with AxisItem)
        """

        pg.GraphicsWidget.__init__(self, parent)

        self.setSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        
        self.autoBtn = pg.ButtonItem(pg.icons.getGraphPixmap('auto'), 14, self)
        self.autoBtn.mode = 'auto'
        self.autoBtn.clicked.connect(self.autoBtnClicked)

        # this prevents a meaningless "letter A" button from coming up
        #
        self.buttonsHidden = True
        self.mouseHovering = False

        self.layout = QtGui.QGraphicsGridLayout()
        self.layout.setContentsMargins(1,
                                       1,
                                       1,
                                       1)
        self.setLayout(self.layout)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)

        if viewBox is None:
            viewBox = DemoWaveformViewBox(parent=self)
        self.vb = viewBox
        self.vb.sigStateChanged.connect(self.viewStateChanged)

        # en/disable plotitem and viewbox menus
        #
        self.setMenuEnabled(enableMenu, enableMenu)

        if name is not None:
            self.vb.register(name)
        self.vb.sigRangeChanged.connect(self.sigRangeChanged)
        self.vb.sigXRangeChanged.connect(self.sigXRangeChanged)
        self.vb.sigYRangeChanged.connect(self.sigYRangeChanged)

        self.layout.addItem(self.vb,
                            2,
                            1)
        self.alpha = 1.0
        self.autoAlpha = True
        self.spectrumMode = False

        self.legend = None

        # Create and place axis items
        #
        if axisItems is None:
            axisItems = {}
        self.axes = {}
        for k, pos in (('top', (1, 1)),
                       ('bottom', (3, 1)),
                       ('left', (2, 0)),
                       ('right', (2, 2))):

            # this is the change that was made / where extra-axis bug was fixed
            #
            if k in axisItems:

                axis = axisItems[k]
            else:
                axis = pg.AxisItem(orientation=k, parent=self)
            axis.linkToView(self.vb)
            self.axes[k] = {'item': axis, 'pos': pos}
            self.layout.addItem(axis, *pos)
            axis.setZValue(-1000)
            axis.setFlag(axis.ItemNegativeZStacksBehindParent)

        self.titleLabel = pg.LabelItem('', size='11pt', parent=self)
        self.layout.addItem(self.titleLabel, 0, 1)
        self.setTitle(None)

        for i in range(4):
            self.layout.setRowPreferredHeight(i, 0)
            self.layout.setRowMinimumHeight(i, 0)
            self.layout.setRowSpacing(i, 0)
            self.layout.setRowStretchFactor(i, 1)

        for i in range(3):
            self.layout.setColumnPreferredWidth(i, 0)
            self.layout.setColumnMinimumWidth(i, 0)
            self.layout.setColumnSpacing(i, 0)
            self.layout.setColumnStretchFactor(i, 1)

        self.layout.setRowStretchFactor(2, 100)
        self.layout.setColumnStretchFactor(1, 100)

        self.items = []
        self.curves = []
        self.itemMeta = weakref.WeakKeyDictionary()
        self.dataItems = []
        self.paramList = {}
        self.avgCurves = {}

        # Set up context menu
        #
        w = QtGui.QWidget()
        self.ctrl = c = ConfigTemplate.Ui_Form()
        c.setupUi(w)
        dv = QtGui.QDoubleValidator(self)

        menuItems = [
            ('Transforms', c.transformGroup),
            ('Downsample', c.decimateGroup),
            ('Average', c.averageGroup),
            ('Alpha', c.alphaGroup),
            ('Grid', c.gridGroup),
            ('Points', c.pointsGroup),
        ]

        self.ctrlMenu = QtGui.QMenu()

        self.ctrlMenu.setTitle('Plot Options')
        self.subMenus = []
        for name, grp in menuItems:
            sm = QtGui.QMenu(name)
            act = QtGui.QWidgetAction(self)
            act.setDefaultWidget(grp)
            sm.addAction(act)
            self.subMenus.append(sm)
            self.ctrlMenu.addMenu(sm)

        self.stateGroup = pg.WidgetGroup()
        for name, w in menuItems:
            self.stateGroup.autoAdd(w)

        self.fileDialog = None

        c.alphaGroup.toggled.connect(self.updateAlpha)
        c.alphaSlider.valueChanged.connect(self.updateAlpha)
        c.autoAlphaCheck.toggled.connect(self.updateAlpha)

        c.xGridCheck.toggled.connect(self.updateGrid)
        c.yGridCheck.toggled.connect(self.updateGrid)
        c.gridAlphaSlider.valueChanged.connect(self.updateGrid)

        c.fftCheck.toggled.connect(self.updateSpectrumMode)
        c.logXCheck.toggled.connect(self.updateLogMode)
        c.logYCheck.toggled.connect(self.updateLogMode)

        c.downsampleSpin.valueChanged.connect(self.updateDownsampling)
        c.downsampleCheck.toggled.connect(self.updateDownsampling)
        c.autoDownsampleCheck.toggled.connect(self.updateDownsampling)
        c.subsampleRadio.toggled.connect(self.updateDownsampling)
        c.meanRadio.toggled.connect(self.updateDownsampling)
        c.clipToViewCheck.toggled.connect(self.updateDownsampling)

        self.ctrl.avgParamList.itemClicked.connect(self.avgParamListClicked)
        self.ctrl.averageGroup.toggled.connect(self.avgToggled)

        self.ctrl.maxTracesCheck.toggled.connect(self.updateDecimation)
        self.ctrl.maxTracesSpin.valueChanged.connect(self.updateDecimation)

        self.hideAxis('right')
        self.hideAxis('top')
        self.showAxis('left')
        self.showAxis('bottom')

        if labels is None:
            labels = {}
        for label in list(self.axes.keys()):
            if label in args:
                labels[label] = args[label]
                del args[label]
        for k in labels:
            if isinstance(labels[k], basestring):
                labels[k] = (labels[k],)
            self.setLabel(k, *labels[k])

        if title is not None:
            self.setTitle(title)

        if len(args) > 0:
            self.plot(**args)

class DemoWaveformViewBox(pg.ViewBox):

    signal_region_selected=QtCore.Signal(float, float, float, float)
    sigRightClick=QtCore.Signal(float, float, object)
    sigLeftClick=QtCore.Signal(float, float, object)
    sigPanDrag=QtCore.Signal(float)

    def __init__(self,
                 *args,
                 **kwds):
        pg.ViewBox.__init__(self,
                            *args,
                            **kwds)
        self.setMouseMode(self.RectMode)
        self.mode = "add";

        self.pen_rect = pg.functions.mkPen((128, 128, 0), width=1.5)
        self.brush_rect = pg.functions.mkBrush(160, 128, 0, 45)

        self.rbScaleBox.setPen(self.pen_rect)
        self.rbScaleBox.setBrush(self.brush_rect)

    def set_mode(self,mode):
        if mode == "pan":
            self.mode = "pan";
            self.setMouseMode(self.PanMode)
        elif mode == "zoom":
            self.mode = "zoom"
            self.setMouseMode(self.RectMode)
        elif mode == "add":
            self.mode = "add"
            self.setMouseMode(self.RectMode)
        else:
            print ("mode is not supported by viewbox (CustomViewBox Class).")
            exit();

    # reimplement right-click to zoom out
    # def mouseClickEvent(self,
    #                     event):
    #     if self.mode == "zoom":
    #         if event.button() == QtCore.Qt.RightButton:
    #             modifiers = QtGui.QApplication.keyboardModifiers();
    #             position = event.screenPos()
    #             self.sigRightClick.emit(position.x(),
    #                                     position.y(),
    #                                     modifiers);

    #     elif self.mode == "add":
    #         if event.button() == QtCore.Qt.RightButton:
    #             modifiers = QtGui.QApplication.keyboardModifiers();
    #             position = event.screenPos()
    #             self.sigRightClick.emit(position.x(),position.y(),modifiers);

    #         if event.button() == QtCore.Qt.LeftButton:
    #             modifiers = QtGui.QApplication.keyboardModifiers();
    #             position = event.screenPos()
    #             self.sigLeftClick.emit(position.x(),position.y(),modifiers);


    def mouseDragEvent(self,
                       event,
                       axis=0):

        if self.mode == "add":

            # we accept all buttons in add mode
            #
            event.accept()

            position = event.pos()

            if self.state['mouseMode'] == self.RectMode:
                if event.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):

                    # final move in the drag; change the view scale
                    #
                    if event.isFinish():
                        self.rbScaleBox.hide()

                        point = pg.Point(event.buttonDownPos(event.button()))
                        rect = QtCore.QRectF(point, pg.Point(position))

                        mapped_rect = self.childGroup.mapRectFromParent(rect)
                        x_position_start, y_position_bottom,  \
                            x_position_end, y_position_top = \
                                mapped_rect.getCoords()

                        self.signal_region_selected.emit(
                            x_position_start,
                            y_position_bottom,
                            x_position_end,
                            y_position_top)
                    else:

                        # still dragging, not finished yet. Update
                        # shape of scale box
                        #
                        self.updateScaleBox(event.buttonDownPos(),
                                            event.pos())
        else:
            pg.ViewBox.mouseDragEvent(self,
                                      event,
                                      axis=0)
        # elif self.mode == "pan":

        #      posx=self.state["viewRange"][0][0];

        #      self.sigPanDrag.emit(posx)

        #      pg.ViewBox.mouseDragEvent(self,
        #                                event,
        #                                axis=0)

        # elif self.mode == "zoom":
        #     pg.ViewBox.mouseDragEvent(self,event)
