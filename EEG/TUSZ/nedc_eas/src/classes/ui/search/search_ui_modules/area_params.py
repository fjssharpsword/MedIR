from pyqtgraph.Qt import QtGui, QtCore

class DemoSearchAreaParams(QtGui.QFrame):
    def __init__(self,
                 parent):
        QtGui.QFrame.__init__(self, parent)

        self.layout_grid = QtGui.QGridLayout(self)

        self.label_area = QtGui.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(600)
        self.label_area.setFont(font)
        self.label_area.setText("Refine results for")
        self.layout_grid.addWidget(self.label_area, 0, 0, 1, 2)
        self.init_checkboxes()


    def init_checkboxes(self):

        self.label_epileptiform_discharges = QtGui.QLabel(
            self)
        self.layout_grid.addWidget(self.label_epileptiform_discharges,
                                         1, 0, 1, 2)

        self.check_box_gped = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_gped, 2, 0, 1, 2)
        self.check_box_gped.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_gped.setText("GPED")

        self.check_box_pled = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_pled, 3, 0, 1, 2)
        self.check_box_pled.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_pled.setText("PLED")

        self.check_box_spsw = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_spsw, 4, 0, 1, 2)
        self.check_box_spsw.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_spsw.setText("SPSW")

        self.label_seizure_events = QtGui.QLabel(self)
        self.layout_grid.addWidget(self.label_seizure_events, 6, 0, 1, 2)
        self.label_seizure_events.setText("Seizure Events")

        self.check_box_seiz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_seiz, 7, 0, 1, 1)
        self.check_box_seiz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_seiz.setText("SEIZ")

        self.check_box_fnsz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_fnsz, 7, 1, 1, 1)
        self.check_box_fnsz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_fnsz.setText("FNSZ")

        self.check_box_gnsz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_gnsz, 8, 0, 1, 1)
        self.check_box_gnsz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_gnsz.setText("GNSZ")

        self.check_box_spsz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_spsz, 8, 1, 1, 1)
        self.check_box_spsz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_spsz.setText("SPSZ")

        self.check_box_cpsz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_cpsz, 9, 0, 1, 1)
        self.check_box_cpsz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_cpsz.setText("CPSZ")

        self.check_box_absz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_absz, 9, 1, 1, 1)
        self.check_box_absz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_absz.setText("ABSZ")

        self.check_box_tnsz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_tnsz, 10, 0, 1, 1)
        self.check_box_tnsz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_tnsz.setText("TNSZ")

        self.check_box_cnsz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_cnsz, 10, 1, 1, 1)
        self.check_box_cnsz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_cnsz.setText("CNSZ")

        self.check_box_tcsz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_tcsz, 11, 0, 1, 1)
        self.check_box_tcsz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_tcsz.setText("TCSZ")

        self.check_box_atsz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_atsz, 11, 1, 1, 1)
        self.check_box_atsz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_atsz.setText("ATSZ")

        self.check_box_mysz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_mysz, 12, 0, 1, 1)
        self.check_box_mysz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_mysz.setText("MYSZ")

        self.check_box_nesz = QtGui.QCheckBox(self)
        self.layout_grid.addWidget(self.check_box_nesz, 12, 1, 1, 1)
        self.check_box_nesz.setFocusPolicy(QtCore.Qt.NoFocus)
        self.check_box_nesz.setText("NESZ")
        
        checkbox_spacer = QtGui.QSpacerItem(20,
                                            108,
                                            QtGui.QSizePolicy.Minimum,
                                            QtGui.QSizePolicy.Expanding)
        self.layout_grid.addItem(checkbox_spacer, 13, 0, 1, 1)

    def get_checkbox_dict(self):
        checkbox_dict = {}
        if self.check_box_gped.isChecked():
            checkbox_dict['gped'] = True
        else:
            checkbox_dict['gped'] = False

        if self.check_box_pled.isChecked():
            checkbox_dict['pled'] = True
        else:
            checkbox_dict['pled'] = False

        if self.check_box_spsw.isChecked():
            checkbox_dict['spsw'] = True
        else:
            checkbox_dict['spsw'] = False

        return checkbox_dict
