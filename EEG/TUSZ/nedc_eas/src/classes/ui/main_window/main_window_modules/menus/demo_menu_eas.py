# -*- coding: utf-8 -*-

from pyqtgraph.Qt import QtGui

class DemoMenuEAS(QtGui.QMenu):
    def __init__(self,
                 menu_bar_parent_a):
        QtGui.QMenu.__init__(self,
                             menu_bar_parent_a)

        self.actions_setup()
        self.actions_connect_and_name()


    def actions_setup(self):
        self.action_about_eas = QtGui.QAction(self)
        self.action_about_eas.setEnabled(True)

        self.action_preferences = QtGui.QAction(self)
        self.action_preferences.setEnabled(True)

        self.action_quit_eas = QtGui.QAction(self)
        self.action_quit_eas.setEnabled(True)

    def actions_connect_and_name(self):
        self.addAction(self.action_about_eas)
        # self.addSeparator()
        self.addAction(self.action_preferences)
        self.addAction(self.action_quit_eas)

        self.setTitle("EAS")
        self.action_about_eas.setText("About EAS")
        self.action_preferences.setText("Preferences...")
        self.action_quit_eas.setText("Quit EAS")
