from pyqtgraph.Qt import QtGui

class DemoMenuFilter(QtGui.QMenu):
    def __init__(self,
                 menu_bar_parent_a):
        QtGui.QMenu.__init__(self,
                             menu_bar_parent_a)

        # create and name filter menu submenus
        #
        self.sub_menu_low_cut = QtGui.QMenu(self)
        self.sub_menu_high_cut = QtGui.QMenu(self)
        self.sub_menu_notch = QtGui.QMenu(self)
        self.sub_menu_rhythms = QtGui.QMenu(self)

        self.actions_setup()
        self.actions_connect_and_name()

    def actions_setup(self):
        # low_cut submenu
        # includes:
        # Off, 5 Hz, 10 Hz, 15 Hz, 20 Hz, 25 Hz, 30 Hz, Custom Frequency
        #
        self.action_low_cut = QtGui.QAction(self)
        
        self.action_low_cut_off = QtGui.QAction(self)
        self.action_low_cut_off.setEnabled(True)
        self.action_low_cut_off.setCheckable(True)
        self.action_low_cut_off.setChecked(True)

        self.action_low_cut_5hz = QtGui.QAction(self)
        self.action_low_cut_5hz.setEnabled(True)
        self.action_low_cut_5hz.setCheckable(True)
        self.action_low_cut_5hz.setChecked(False)

        self.action_low_cut_10hz = QtGui.QAction(self)
        self.action_low_cut_10hz.setEnabled(True)
        self.action_low_cut_10hz.setCheckable(True)
        self.action_low_cut_10hz.setChecked(False)

        self.action_low_cut_15hz = QtGui.QAction(self)
        self.action_low_cut_15hz.setEnabled(True)
        self.action_low_cut_15hz.setCheckable(True)
        self.action_low_cut_15hz.setChecked(False)

        self.action_low_cut_20hz = QtGui.QAction(self)
        self.action_low_cut_20hz.setEnabled(True)
        self.action_low_cut_20hz.setCheckable(True)
        self.action_low_cut_20hz.setChecked(False)

        self.action_low_cut_25hz = QtGui.QAction(self)
        self.action_low_cut_25hz.setEnabled(True)
        self.action_low_cut_25hz.setCheckable(True)
        self.action_low_cut_25hz.setChecked(False)

        self.action_low_cut_30hz = QtGui.QAction(self)
        self.action_low_cut_30hz.setEnabled(True)
        self.action_low_cut_30hz.setCheckable(True)
        self.action_low_cut_30hz.setChecked(False)
        
        self.action_low_cut_custom_frequency = QtGui.QAction(self)
        self.action_low_cut_custom_frequency.setEnabled(True)
        self.action_low_cut_custom_frequency.setCheckable(True)
        self.action_low_cut_custom_frequency.setChecked(False)

        self.action_group_low_cut = QtGui.QActionGroup(self)
        self.action_group_low_cut.addAction(self.action_low_cut_off)
        self.action_group_low_cut.addAction(self.action_low_cut_5hz)
        self.action_group_low_cut.addAction(self.action_low_cut_10hz)
        self.action_group_low_cut.addAction(self.action_low_cut_15hz)
        self.action_group_low_cut.addAction(self.action_low_cut_20hz)
        self.action_group_low_cut.addAction(self.action_low_cut_25hz)
        self.action_group_low_cut.addAction(self.action_low_cut_30hz)
        self.action_group_low_cut.addAction(
            self.action_low_cut_custom_frequency)

        # high cut submenu
        # includes:
        # Off, 100 Hz, 75 Hz, 50 Hz, 40 Hz, 30 Hz, 30 Hz
        #
        self.action_high_cut = QtGui.QAction(self)

        self.action_high_cut_off = QtGui.QAction(self)
        self.action_high_cut_off.setEnabled(True)
        self.action_high_cut_off.setCheckable(True)
        self.action_high_cut_off.setChecked(True)

        self.action_high_cut_100hz = QtGui.QAction(self)
        self.action_high_cut_100hz.setEnabled(True)
        self.action_high_cut_100hz.setCheckable(True)
        self.action_high_cut_100hz.setChecked(False)

        self.action_high_cut_75hz = QtGui.QAction(self)
        self.action_high_cut_75hz.setEnabled(True)
        self.action_high_cut_75hz.setCheckable(True)
        self.action_high_cut_75hz.setChecked(False)

        self.action_high_cut_50hz = QtGui.QAction(self)
        self.action_high_cut_50hz.setEnabled(True)
        self.action_high_cut_50hz.setCheckable(True)
        self.action_high_cut_50hz.setChecked(False)

        self.action_high_cut_40hz = QtGui.QAction(self)
        self.action_high_cut_40hz.setEnabled(True)
        self.action_high_cut_40hz.setCheckable(True)
        self.action_high_cut_40hz.setChecked(False)

        self.action_high_cut_30hz = QtGui.QAction(self)
        self.action_high_cut_30hz.setEnabled(True)
        self.action_high_cut_30hz.setCheckable(True)
        self.action_high_cut_30hz.setChecked(False)

        self.action_high_cut_20hz = QtGui.QAction(self)
        self.action_high_cut_20hz.setEnabled(True)
        self.action_high_cut_20hz.setCheckable(True)
        self.action_high_cut_20hz.setChecked(False)

        self.action_high_cut_10hz = QtGui.QAction(self)
        self.action_high_cut_10hz.setEnabled(True)
        self.action_high_cut_10hz.setCheckable(True)
        self.action_high_cut_10hz.setChecked(False)

        self.action_high_cut_custom_frequency = QtGui.QAction(self)
        self.action_high_cut_custom_frequency.setEnabled(True)
        self.action_high_cut_custom_frequency.setCheckable(True)
        self.action_high_cut_custom_frequency.setChecked(False)

        self.action_group_high_cut = QtGui.QActionGroup(self)
        self.action_group_high_cut.addAction(self.action_high_cut_off)
        self.action_group_high_cut.addAction(self.action_high_cut_100hz)
        self.action_group_high_cut.addAction(self.action_high_cut_75hz)
        self.action_group_high_cut.addAction(self.action_high_cut_50hz)
        self.action_group_high_cut.addAction(self.action_high_cut_40hz)
        self.action_group_high_cut.addAction(self.action_high_cut_30hz)
        self.action_group_high_cut.addAction(self.action_high_cut_20hz)
        self.action_group_high_cut.addAction(self.action_high_cut_10hz)
        self.action_group_high_cut.addAction(
            self.action_high_cut_custom_frequency)

        # notch submenu
        #
        self.action_notch = QtGui.QAction(self)

        self.action_notch_off = QtGui.QAction(self)
        self.action_notch_off.setEnabled(True)
        self.action_notch_off.setCheckable(True)
        self.action_notch_off.setChecked(False)

        self.action_notch_60hz = QtGui.QAction(self)
        self.action_notch_60hz.setEnabled(True)
        self.action_notch_60hz.setCheckable(True)
        self.action_notch_60hz.setChecked(True)

        self.action_notch_50hz = QtGui.QAction(self)
        self.action_notch_50hz.setEnabled(True)
        self.action_notch_50hz.setCheckable(True)
        self.action_notch_50hz.setChecked(False)

        self.action_notch_custom_frequency = QtGui.QAction(self)
        self.action_notch_custom_frequency.setEnabled(True)
        self.action_notch_custom_frequency.setCheckable(True)
        self.action_notch_custom_frequency.setChecked(False)

        self.action_group_notch = QtGui.QActionGroup(self)
        self.action_group_notch.addAction(self.action_notch_off)
        self.action_group_notch.addAction(self.action_notch_60hz)
        self.action_group_notch.addAction(self.action_notch_50hz)
        self.action_group_notch.addAction(
            self.action_notch_custom_frequency)

        # rhythms menu
        #
        self.action_rhythms_off = QtGui.QAction(self)
        self.action_rhythms_off.setEnabled(True)
        self.action_rhythms_off.setCheckable(True)
        self.action_rhythms_off.setChecked(True)
        
        self.action_rhythms_delta_select = QtGui.QAction(self)
        self.action_rhythms_delta_select.setEnabled(True)
        self.action_rhythms_delta_select.setCheckable(True)
        self.action_rhythms_delta_select.setChecked(False)

        self.action_rhythms_theta_select = QtGui.QAction(self)
        self.action_rhythms_theta_select.setEnabled(True)
        self.action_rhythms_theta_select.setCheckable(True)
        self.action_rhythms_theta_select.setChecked(False)

        self.action_rhythms_alpha_select = QtGui.QAction(self)
        self.action_rhythms_alpha_select.setEnabled(True)
        self.action_rhythms_alpha_select.setCheckable(True)
        self.action_rhythms_alpha_select.setChecked(False)

        self.action_rhythms_beta_select = QtGui.QAction(self)
        self.action_rhythms_beta_select.setEnabled(True)
        self.action_rhythms_beta_select.setCheckable(True)
        self.action_rhythms_beta_select.setChecked(False)

        self.action_rhythms_gamma_select = QtGui.QAction(self)
        self.action_rhythms_gamma_select.setEnabled(True)
        self.action_rhythms_gamma_select.setCheckable(True)
        self.action_rhythms_gamma_select.setChecked(False)

        self.action_group_rhythms = QtGui.QActionGroup(self)
        self.action_group_rhythms.addAction(self.action_rhythms_off)
        self.action_group_rhythms.addAction(self.action_rhythms_delta_select)
        self.action_group_rhythms.addAction(self.action_rhythms_theta_select)
        self.action_group_rhythms.addAction(self.action_rhythms_alpha_select)
        self.action_group_rhythms.addAction(self.action_rhythms_beta_select)
        self.action_group_rhythms.addAction(self.action_rhythms_gamma_select)

        self.list_action_filters = []
        self.list_action_filters.append(self.action_low_cut_off)
        self.list_action_filters.append(self.action_low_cut_5hz)
        self.list_action_filters.append(self.action_low_cut_10hz)
        self.list_action_filters.append(self.action_low_cut_15hz)
        self.list_action_filters.append(self.action_low_cut_20hz)
        self.list_action_filters.append(self.action_low_cut_25hz)
        self.list_action_filters.append(self.action_low_cut_30hz)
        self.list_action_filters.append(self.action_low_cut_custom_frequency)
        self.list_action_filters.append(self.action_high_cut_off)
        self.list_action_filters.append(self.action_high_cut_100hz)
        self.list_action_filters.append(self.action_high_cut_75hz)
        self.list_action_filters.append(self.action_high_cut_50hz)
        self.list_action_filters.append(self.action_high_cut_40hz)
        self.list_action_filters.append(self.action_high_cut_30hz)
        self.list_action_filters.append(self.action_high_cut_20hz)
        self.list_action_filters.append(self.action_high_cut_10hz)
        self.list_action_filters.append(self.action_high_cut_custom_frequency)
        self.list_action_filters.append(self.action_notch_off)
        self.list_action_filters.append(self.action_notch_60hz)
        self.list_action_filters.append(self.action_notch_50hz)
        self.list_action_filters.append(self.action_notch_custom_frequency)
        self.list_action_filters.append(self.action_rhythms_off)
        self.list_action_filters.append(self.action_rhythms_delta_select)
        self.list_action_filters.append(self.action_rhythms_theta_select)
        self.list_action_filters.append(self.action_rhythms_alpha_select)
        self.list_action_filters.append(self.action_rhythms_beta_select)
        self.list_action_filters.append(self.action_rhythms_gamma_select)

        # make sure that selection from the rhythm submenu enables or disables
        # the other filter submenus as specified in design document
        #
        #  Off  -> other filter submenus enabled
        #  !Off -> other filter submenus disabled
        #
        self.action_group_rhythms.triggered.connect(
            self.toggle_enable_non_rhythm_filters)

    def actions_connect_and_name(self):
        self.setTitle("Filter")

        # low cut submenu
        #
        self.addAction(self.sub_menu_low_cut.menuAction())
        self.sub_menu_low_cut.setTitle("Low Cut")

        self.sub_menu_low_cut.addAction(self.action_low_cut_off)
        self.sub_menu_low_cut.addAction(self.action_low_cut_5hz)
        self.sub_menu_low_cut.addAction(self.action_low_cut_10hz)
        self.sub_menu_low_cut.addAction(self.action_low_cut_15hz)
        self.sub_menu_low_cut.addAction(self.action_low_cut_20hz)
        self.sub_menu_low_cut.addAction(self.action_low_cut_25hz)
        self.sub_menu_low_cut.addAction(self.action_low_cut_30hz)
        self.sub_menu_low_cut.addAction(self.action_low_cut_custom_frequency)
        self.action_low_cut_off.setText("Off")
        self.action_low_cut_5hz.setText("5 Hz")
        self.action_low_cut_10hz.setText("10 Hz")
        self.action_low_cut_15hz.setText("15 Hz")
        self.action_low_cut_20hz.setText("20 Hz")
        self.action_low_cut_25hz.setText("25 Hz")
        self.action_low_cut_30hz.setText("30 Hz")
        self.action_low_cut_custom_frequency.setText("Custom Frequency")

        # high cut submenu
        #
        self.addAction(self.sub_menu_high_cut.menuAction())
        self.sub_menu_high_cut.setTitle("High Cut")

        self.sub_menu_high_cut.addAction(self.action_high_cut_off)
        self.sub_menu_high_cut.addAction(self.action_high_cut_100hz)
        self.sub_menu_high_cut.addAction(self.action_high_cut_75hz)
        self.sub_menu_high_cut.addAction(self.action_high_cut_50hz)
        self.sub_menu_high_cut.addAction(self.action_high_cut_40hz)
        self.sub_menu_high_cut.addAction(self.action_high_cut_30hz)
        self.sub_menu_high_cut.addAction(self.action_high_cut_20hz)
        self.sub_menu_high_cut.addAction(self.action_high_cut_10hz)
        self.sub_menu_high_cut.addAction(self.action_high_cut_custom_frequency)
        self.action_high_cut_off.setText("Off")
        self.action_high_cut_100hz.setText("100 Hz")
        self.action_high_cut_75hz.setText("75 Hz")
        self.action_high_cut_50hz.setText("50 Hz")
        self.action_high_cut_40hz.setText("40 Hz")
        self.action_high_cut_30hz.setText("30 Hz")
        self.action_high_cut_20hz.setText("20 Hz")
        self.action_high_cut_10hz.setText("10 Hz")
        self.action_high_cut_custom_frequency.setText("Custom Frequency")

        # notch submenu
        #
        self.addAction(self.sub_menu_notch.menuAction())
        self.sub_menu_notch.setTitle("Notch")

        self.sub_menu_notch.addAction(self.action_notch_off)
        self.sub_menu_notch.addAction(self.action_notch_60hz)
        self.sub_menu_notch.addAction(self.action_notch_50hz)
        self.sub_menu_notch.addAction(self.action_notch_custom_frequency)
        self.action_notch_off.setText("Off")
        self.action_notch_60hz.setText("60 Hz")
        self.action_notch_50hz.setText("50 Hz")
        self.action_notch_custom_frequency.setText("Custom Frequency")
      
        self.addSeparator()

        self.addAction(self.sub_menu_rhythms.menuAction())
        self.sub_menu_rhythms.setTitle("Rhythms")

        self.sub_menu_rhythms.addAction(self.action_rhythms_off)
        self.sub_menu_rhythms.addAction(self.action_rhythms_delta_select)
        self.sub_menu_rhythms.addAction(self.action_rhythms_theta_select)
        self.sub_menu_rhythms.addAction(self.action_rhythms_alpha_select)
        self.sub_menu_rhythms.addAction(self.action_rhythms_beta_select)
        self.sub_menu_rhythms.addAction(
            self.action_rhythms_gamma_select)

        self.action_rhythms_off.setText("Off")
        self.action_rhythms_delta_select.setText("Delta")
        self.action_rhythms_theta_select.setText("Theta")
        self.action_rhythms_alpha_select.setText("Alpha")
        self.action_rhythms_beta_select.setText("Beta")
        self.action_rhythms_gamma_select.setText("Gamma")

    def toggle_enable_non_rhythm_filters(self):
        if self.action_rhythms_off.isChecked():
            self.sub_menu_low_cut.setEnabled(True)
            self.sub_menu_high_cut.setEnabled(True)
            self.sub_menu_notch.setEnabled(True)
        else:
            self.sub_menu_low_cut.setEnabled(False)
            self.sub_menu_high_cut.setEnabled(False)
            self.sub_menu_notch.setEnabled(False)
