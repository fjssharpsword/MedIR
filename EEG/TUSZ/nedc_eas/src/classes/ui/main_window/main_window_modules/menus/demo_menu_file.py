from pyqtgraph.Qt import QtGui

class DemoMenuFile(QtGui.QMenu):
    def __init__(self,
                 menu_bar_parent_a):
        QtGui.QMenu.__init__(self,
                             menu_bar_parent_a)
        self.actions_setup()
        self.actions_connect_and_name()

    def actions_setup(self):
        
        self.action_open = QtGui.QAction(self)

        self.action_open_recent = QtGui.QAction(self)
        self.action_open_recent.setEnabled(False)

        self.action_find = QtGui.QAction(self)
        self.action_find.setEnabled(False)

        self.action_file_save = QtGui.QAction(self)
        self.action_file_save.setEnabled(False)
        
        self.action_file_save_as = QtGui.QAction(self)
        self.action_file_save_as.setEnabled(False)

        self.action_search_for_file = QtGui.QAction(self)
        self.action_search_for_file.setEnabled(True)

        self.action_print = QtGui.QAction(self)
        self.action_print.setEnabled(True)

    def actions_connect_and_name(self):
        self.addAction(self.action_open)
        self.addAction(self.action_open_recent)
        self.addAction(self.action_find)
        self.addAction(self.action_search_for_file)
        self.addSeparator()

        self.addAction(self.action_file_save)
        self.addAction(self.action_file_save_as)

        self.addSeparator()
        self.addAction(self.action_print)

        self.setTitle("File")
        self.action_open.setText("Open")
        self.action_open_recent.setText("Open Recent")
        self.action_find.setText("Find")
        self.action_search_for_file.setText("Search")
        self.action_print.setText("Print")

        self.action_file_save.setText("Save")
        self.action_file_save_as.setText("Save As")