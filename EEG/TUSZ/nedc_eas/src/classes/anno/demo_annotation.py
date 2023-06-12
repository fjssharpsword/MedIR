#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/anno/demo_annotation.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
from pyqtgraph import QtGui, QtCore, functions
from .demo_roi_page_1 import DemoROIPage1


# must inherit from QObject in order to be able to chain signals
#
class DemoAnnotation(QtCore.QObject):
    signal_region_changed=QtCore.Signal(float,
                                        float,
                                        int)

    signal_remove_unique_key = QtCore.Signal(int)
    signal_edit_unique_key = QtCore.Signal(int)

    def __init__(self,
                 position_a,
                 channel_a,
                 anno_name_a,
                 cfg_dict_a,
                 cfg_dict_annotation_colors_a,
                 page_1_parent,
                 unique_key_a,
                 size,
                 maxBounds):
        super(QtCore.QObject, self).__init__()

        self.anno_name = anno_name_a
        if unique_key_a is not None:
            self.unique_key = unique_key_a

        self.left_bound = position_a[0]
        self.right_bound = size[0] + self.left_bound
        self.channel_name = channel_a

        self.dict_anno_colors = cfg_dict_annotation_colors_a

        self.page_1_roi = DemoROIPage1(
            self.channel_name,
            position_a,
            size=size,
            anno_name_a=anno_name_a,
            cfg_dict_a=cfg_dict_a,
            maxBounds=maxBounds,
            parent=page_1_parent)

        self.page_1_roi.sigRegionChangeFinished.connect(
            self.deal_with_annotation_movement)

        self.page_1_roi.signal_remove_item.connect(
            self.deal_with_single_remove)

        self.page_1_roi.signal_edit_item.connect(
            self.deal_with_single_edit)

        self.pen_empty = functions.mkPen(
            color=cfg_dict_a['pen_color_empty'],
            width=cfg_dict_a['border_width_default'])
        self.pen_default = functions.mkPen(
            color=cfg_dict_a['pen_color_default'],
            width=cfg_dict_a['border_width_default'])
        self.pen_white = functions.mkPen(
            color=cfg_dict_a['pen_color_white'],
            widfth=cfg_dict_a['border_width_default'])
        self.pen_selected = functions.mkPen(
            color=cfg_dict_a['pen_color_selected'],
            width=cfg_dict_a['border_width_selected'])

        self.brush_empty = QtGui.QBrush(QtGui.QColor(
             *cfg_dict_a['brush_empty']))
        
        self.brush_default = QtGui.QBrush(QtGui.QColor(
            *cfg_dict_a['brush_default']))

        self.brush_selected = QtGui.QBrush(QtGui.QColor(
            *cfg_dict_a['brush_selected']))

        # object anno_name (used to refer to this object by parent or
        # other classes)
        #
        if anno_name_a:
            self.anno_name = anno_name_a
            self.page_1_roi.set_label(self.anno_name)
            self.set_type(anno_name_a)
        else:
            self.page_1_roi.set_brush(self.brush_empty)

    # this method was created to allow movement from annotations from
    # both pages to emit the same common signal (self.signal_region_changed)
    #
    def deal_with_annotation_movement(self,
                                      annotation_object_a):

        size_in_seconds = annotation_object_a.size().x()
        left_side = annotation_object_a.pos()[0]
        right_side = left_side + size_in_seconds

        self.signal_region_changed.emit(left_side,
                                        right_side,
                                        self.unique_key)

    def set_style(self,
                  is_empty=False,
                  is_selected=False):

        self.empty = is_empty
        self.selected = is_selected

        if is_empty is True:
            self.page_1_roi.currentPen = self.page_1_roi.pen = self.pen_empty
            self.page_1_roi.set_brush(self.brush_empty)
        if is_selected is True:
            self.page_1_roi.currentPen = self.page_1_roi.pen = self.pen_selected
            self.page_1_roi.set_brush(self.brush_selected)
        else:
            self.page_1_roi.currentPen = self.page_1_roi.pen = self.pen_default
        self.page_1_roi.update()

    def set_type(self,
                 label_string):
        self.anno_name = label_string
        self.page_1_roi.set_label(label_string)
        r, g, b, alpha = self.dict_anno_colors[label_string]
        self.brush = QtGui.QBrush(QtGui.QColor(r, g, b, alpha))
        self.page_1_roi.set_brush(self.brush)
        self.set_style(is_selected=False)
        self.page_1_roi.update()

    def remove(self):
        self.page_1_roi.remove_item()

    def deal_with_single_remove(self):
        self.signal_remove_unique_key.emit(self.unique_key)

    def deal_with_single_edit(self):
        self.signal_edit_unique_key.emit(self.unique_key)
        
