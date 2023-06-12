from pyqtgraph.Qt import QtGui

class DemoMenuHelp(QtGui.QMenu):
    def __init__(self,
                 menu_bar_parent_a):
        QtGui.QMenu.__init__(self,
                             menu_bar_parent_a)

        self.actions_setup()
        self.actions_connect_and_name()

    def actions_setup(self):
        
        self.action_help_search = QtGui.QAction(self)
        self.action_help_search.setEnabled(False)

        self.action_help_eas = QtGui.QAction(self)
        self.action_help_eas.setEnabled(False)

        self.action_welcome = QtGui.QAction(self)
        self.action_welcome.setEnabled(False)

        self.action_getting_started_w_eas = QtGui.QAction(self)
        self.action_getting_started_w_eas.setEnabled(False)

        self.action_visit_website = QtGui.QAction(self)
        self.action_visit_website.setEnabled(False)

        self.action_check_for_updates = QtGui.QAction(self)
        self.action_check_for_updates.setEnabled(False)

        self.action_send_feedback = QtGui.QAction(self)
        self.action_send_feedback.setEnabled(False)

    def actions_connect_and_name(self):
        
        self.addAction(self.action_help_search)
        self.addAction(self.action_help_eas)
        self.addSeparator()
        self.addAction(self.action_welcome)
        self.addAction(self.action_getting_started_w_eas)
        self.addAction(self.action_visit_website)
        self.addSeparator()
        self.addAction(self.action_check_for_updates)
        self.addAction(self.action_send_feedback)

        self.setTitle("Help")
        self.action_help_search.setText("Search")
        self.action_help_eas.setText("EAS Help")
        self.action_welcome.setText("Welcome to EAS\n")
        self.action_getting_started_w_eas.setText(
            "Getting Started with EAS")
        self.action_visit_website.setText("Visit the Product Website")
        self.action_check_for_updates.setText("Check for Updates\n")
        self.action_send_feedback.setText("Send Feedback\n")
