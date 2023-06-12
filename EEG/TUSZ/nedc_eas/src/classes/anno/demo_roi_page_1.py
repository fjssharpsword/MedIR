#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/anno/demo_roi_page_1.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------

from pyqtgraph import QtGui, QtCore, ROI, functions, Point, TextItem

# Customized ROI  used  for  plotting the annotations
# this class adds text, background and some other features to the ROI
class DemoROIPage1(ROI):

    sigMouseClicked=QtCore.Signal(int)
    sigShiftMouseClicked=QtCore.Signal(int)
    sigMoved=QtCore.Signal(float,
                           float,
                           float,
                           float)

    sigMouseHover=QtCore.Signal(float,
                                bool)

    signal_remove_item = QtCore.Signal()
    signal_edit_item = QtCore.Signal()

    def __init__(self,
                 channel_name_a,
                 pos,
                 anno_name_a=None,
                 cfg_dict_a=None,
                 **kwargs):
        ROI.__init__(self,
                     pos,
                     **kwargs)

        # get the position of upper left hand corner in terms of the
        # parent widget
        #
        self.y_position = pos[1]

        self.channel_name = channel_name_a

        # initialize some states
        #
        self.isMoving = False

        if "parent" in kwargs:
            self.parent = kwargs["parent"]

        self._init_menu()

        # set the size and color of the handle
        #
        self.handleSize = cfg_dict_a['handle_size']
        self.handlePen = QtGui.QPen(QtGui.QColor(
            *cfg_dict_a['pen_handle']))

        self.pen_hover = QtGui.QPen(QtGui.QColor(
            *cfg_dict_a['pen_hover']))
        self.parent.addItem(self)

        self.style_string = 'style="color: #' + str(cfg_dict_a['lbl_color']) \
                 + '; font-size: ' + str(cfg_dict_a['lbl_font_size']) + 'pt;">'
        self.text_item = TextItem()
        self.text_item.setParentItem(self)
        bwidth, height =  self.size()
        self.text_item.setPos(0,
                              height)

        center_for_left = [1, 0.5]
        center_for_right = [0, 0.5]
        left = [0, 0.5]
        right = [1, 0.5]
        self.handle_left = self.add_handle(left, center_for_left)
        self.handle_right = self.add_handle(right, center_for_right)

    def add_handle(self,
                   position_a,
                   center_a):

        pos = Point(position_a)
        center = Point(center_a)
        info = {'type': 's',
                'center': center,
                'pos': pos,
                'lockAspect':False}

        handle = self.addHandle(info)
        return handle

    # reimplementation of `ROI.mouseDragEvent` for some reason
    # positive y values in cursor_position was causing ROI to flip
    # vertically, which was very distracting
    #
    def mouseDragEvent(self,
                       event):
        # mouse has been pressed
        #
        if event.isStart():

            if event.button() == QtCore.Qt.LeftButton:

                # light self up and show handles (if any)
                #
                self.setSelected(True)

                # if self allows itself to be moved
                #
                if self.translatable:

                    self.isMoving = True

                    # store state for use in self.remove_move (if necessary)
                    #
                    self.preMoveState = self.getState()

                    # store the original location of click for use
                    # in translation at end of this function
                    #
                    self.cursorOffset = \
                        self.pos() - self.mapToParent(event.buttonDownPos())

                    # emit the signal that the self's region has changed
                    #
                    self.sigRegionChangeStarted.emit(self)

                    # accept the event (prevent propogation to parent region)
                    #
                    event.accept()

                # if not left click, then ignore
                #
                else:
                    event.ignore()

        # mouse has been released
        #
        elif event.isFinish():

            # if self allows itself to be moved
            #
            if self.translatable:

                # if we were moving, emit sigRegionChangeFinished
                #
                if self.isMoving:
                    self.stateChangeFinished()

                # whether we were moving or not, now we are def not moving
                #
                self.isMoving = False
            return

        # if self allows itself to be moved and if left button is
        # being held down and is moving
        #
        if self.translatable  \
           and self.isMoving \
           and event.buttons() == QtCore.Qt.LeftButton:

            # this is unnecessary, but kept around in case we ever do
            # want quantized annotating
            #
            snap = True if (event.modifiers()
                            & QtCore.Qt.ControlModifier) else None

            # extract the cursor position information from the event.
            #
            cursor_position = self.mapToParent(event.pos()) + self.cursorOffset

            # for some reason positive y values in new_position causes
            # vertical translations. this fixes that
            #
            cursor_position.setY(-abs(cursor_position.y()))

            # translate self to new position
            #
            translation = cursor_position - self.pos()
            self.translate(translation,
                           snap=snap,
                           finish=False)

    #  mouse click + shift + click
    
    def mouseClickEvent(self,
                         event):
         modifiers = QtGui.QApplication.keyboardModifiers()
         if (modifiers == QtCore.Qt.ShiftModifier
            and event.button() == QtCore.Qt.LeftButton):
             self.remove_single_annotation()
         else:
             pos = event.screenPos()
             self.menu.popup(QtCore.QPoint(pos.x(),
                                           pos.y()))

    def hoverEvent(self,
                   event):
        if not event.isExit() and event.acceptDrags(QtCore.Qt.LeftButton):
            #self.currentPen = self.pen_hover
            pnt = QtCore.QPoint(event.screenPos().x(),  event.screenPos().y())
            QtGui.QToolTip.showText(pnt,
                                    str(self.channel_name))

            self.sigMouseHover.emit(self.y_position,
                                    True)

        else:
            self.currentPen = self.pen
            ROI.hoverEvent(self,
                           event)
            self.sigMouseHover.emit(self.y_position,
                               False)

        self.update()

    # reimplementation of `GraphicsScene.mouseReleaseEvent` to get rid
    # of error (see try and except below)
    def mouseReleaseEvent(self,
                          event):
        if self.mouseGrabberItem() is None:

            if event.button() in self.dragButtons:
                if self.sendDragEvent(event, final=True):
                    event.accept()
                self.dragButtons.remove(event.button())
            else:
                cev = [e for e in self.clickEvents \
                       if int(e.button()) == int(event.button())]

                # try and except takes care of error
                #
                try:
                    if self.sendClickEvent(cev[0]):
                        event.accept()
                    self.clickEvents.remove(cev[0])
                except:
                    print ("Does this error matter?")

        if int(event.buttons()) == 0:
            self.dragItem = None
            self.dragButtons = []
            self.clickEvents = []
            self.lastDrag = None
        QtGui.QGraphicsScene.mouseReleaseEvent(self, event)

        # let items prepare for next click/drag
        #
        self.sendHoverEvents(event)

    def _init_menu(self):
        self.menu = QtGui.QMenu()

        self.remove_action = QtGui.QAction("Remove",
                                           self.menu)
        self.remove_action.triggered.connect(self.remove_single_annotation)
        self.menu.addAction(self.remove_action)

        self.edit_action = QtGui.QAction("Edit",
                                         self.menu)
        self.edit_action.triggered.connect(self.edit_single_annotation)
        self.menu.addAction(self.edit_action)

    def remove_move(self):
        self.isMoving = False

    def set_label(self,
                 label):

        if label == "":
            # self.text_item.setParentItem(None)
            # self.text_item = None
            self.update()
        html_text = '<div style="text-align: "><span ' \
                    + self.style_string + label + '</span></div>'
        self.text_item.setHtml(html_text)
        self.update()

    # just copied (for some reason without copying this here the
    # position of text changes when the size of ROI changes)
    #
    def stateChanged(self,
                     finish=True):
        """
        Process changes to the state of the ROI.

        If there are any changes, then the positions of handles are
        updated accordingly and sigRegionChanged is emitted. If finish
        is True, then sigRegionChangeFinished will also be emitted.
        """
        changed = False
        if self.lastState is None:
            changed = True
        else:
            for k in list(self.state.keys()):
                if self.state[k] != self.lastState[k]:
                    changed = True
        # print "changed is", changed
        self.prepareGeometryChange()

        if changed:
            ## Move all handles to match the current configuration of the ROI
            for handle in self.handles:
                if handle['item'] in self.childItems():
                    handle_position = handle['pos'] * self.state['size']
                    handle['item'].setPos(handle_position)
                    # print self.focusItem()
                    # print dir(handle['item'])
            self.update()
            self.sigRegionChanged.emit(self)

        elif self.freeHandleMoved:
            self.sigRegionChanged.emit(self)

        self.freeHandleMoved = False
        self.lastState = self.stateCopy()

        if finish:
            self.stateChangeFinished()

    # TODO: abstract into DemoAnnotation
    #
    def set_brush(self,
                 *br,
                 **kwargs):
        """
        Set the brush that fills the region. Can have any arguments that
        are valid for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = functions.mkBrush(*br, **kwargs)
        self.current_brush = self.brush

    # modify the paint method so it also take care of the brush
    #
    def paint(self,
              painter,
              opt,
              widget):
        painter.save()
        rectangle = self.boundingRect()
        painter.setRenderHint(QtGui.QPainter.Antialiasing,True)
        painter.setPen(self.currentPen)
        painter.setBrush(self.current_brush)
        painter.translate(rectangle.left(), rectangle.top())
        painter.scale(rectangle.width(), rectangle.height())
        painter.drawRect(0, 0, 1, 1)
        painter.restore()

    # remove this item. The parent item should be set in constructor
    # (e.g. pw.ItemPlot or anything with a viewbox that this object
    # is located) this method just remove the graphic Item from the
    # scene (if it is related to a logical item it does not affect
    # that)
    #
    # perhaps use event.removeTimer.stop() for a bug mentioned in:
    # https://groups.google.com/forum/#!searchin/pyqtgraph/remove/pyqtgraph/OyhbLmzsdk0/alDwTlVbut8J
    #
    def remove_item(self):
        self.parent.vb.scene().removeItem(self)

    def remove_single_annotation(self):
        self.signal_remove_item.emit()

    def edit_single_annotation(self):
        self.signal_edit_item.emit()
        
    def translate(self, *args, **kwargs):
        """
        Move the ROI to a new position.
        Accepts either (x, y, snap) or ([x,y], snap) as arguments

        If the ROI is bounded and the move would exceed boundaries,
        then the ROI is moved to the nearest acceptable position
        instead.

        snap can be:
           None (default): use self.translateSnap and self.snapSize
                           to determine whether/how to snap
           False:          do not snap
           Point(w,h)      snap to rectangular grid with spacing (w, h)
           True:           snap using self.snapSize
                           (and ignoring self.translateSnap)

        Also accepts *update* and *finish* arguments (see setPos() for
        a description of these).

        """

        if len(args) == 1:
            pt = args[0]
        else:
            pt = args

        new_state = self.stateCopy()
        new_state['pos'] = new_state['pos'] + pt

        snap = kwargs.get('snap', None)
        if snap is None:
            snap = self.translateSnap
        if snap is not False:
            new_state['pos'] = self.getSnapPosition(new_state['pos'],
                                                    snap=snap)

        if self.maxBounds is not None:
            r = self.stateRect(new_state)

            d = Point(0, 0)
            if self.maxBounds.left() > r.left():
                d[0] = self.maxBounds.left() - r.left()
            elif self.maxBounds.right() < r.right():
                d[0] = self.maxBounds.right() - r.right()
            if self.maxBounds.top() > r.top():
                d[1] = self.maxBounds.top() - r.top()
            elif self.maxBounds.bottom() < r.bottom():
                d[1] = self.maxBounds.bottom() - r.bottom()
            new_state['pos'] += d

        update = kwargs.get('update', True)
        finish = kwargs.get('finish', True)
        self.setPos(new_state['pos'],
                    update=update,
                    finish=finish)

        self.sigMoved.emit(new_state['pos'].x(),
                           new_state['pos'].y(),
                           self.size().x(),
                           self.size().y())


