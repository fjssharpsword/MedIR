# <*** THERE IS A GLOBAL VARIABLE DECLARED AT THE END OF THIS FILE ***>
# <*** THAT IS IN AN INSTANCE OF THIS CLASS. THE REASONS FOR THIS  ***>
# <*** DECISION ARE LISTED BELOW                                   ***>


# This class stores references to all DemoSearch, DemoEventLoop.ui,
# and DemoTextReportWidget windows upon creation and deletes these
# references when the windows are closed. None of the other widgets
# require this.
#
# If we don't store these references then no widgets but the first
# will ever appear on screen - It appears that they are deleted
# immediately after creation. Maybe garbage collector gets them?
#
# This allows us to abuse PyQt into allowing us multiple "main
# windows".  This is not how PyQt is meant to be used, but it seems
# this is only way to get the multiple main window functionality.
#
# This decision was not made lightly.
#
class DemoWindowManager:
    def __init__(self):

        # initialize a widget list. This is used to store references
        # to all DemoSearch and DemoEventLoop.ui (DemoMainWindow)
        # widgets.
        #
        self.child_widget_list = []

    # method: manage
    #
    # arguments:
    #  -widget_a: a widget whose memory we will need to manage
    #
    # returns: none
    #
    # this is a simple interface that takes care of saving a reference to
    # widget_a, as well as connecting that widget's sig_closed signal so that
    # the reference will get deleted.
    #
    def manage(self,
               widget_a):
        self._append_to_widget_list(widget_a)

        # when the widget is closed, a closeEvent occurs. This has  been
        # made to emit a signal that we can capture here. When this signal
        # is captured, this class will delete its reference to the widget.
        # (allowing python's garbage collector to get too it)
        #
        widget_a.sig_closed.connect(self._delete_widget_ref)

    # method: _append_to_widget_list
    #
    # arguments:
    #  -widget_a: a widget that we will save a reference to
    #
    # returns: none
    #
    # store a reference to a widget in self.child_widget_list
    #
    def _append_to_widget_list(self,
                               widget_a):
        self.child_widget_list.append(widget_a)

    # method: _delete_widget_ref
    #
    # arguments:
    #  -widget_a: a widget what whose reference we will delete
    #
    # returns: none
    #
    # this method ensures that self doesn't hold references to widgets
    # that have already been closed. I suspect that if this is not
    # done, these widgets will stay in memory after they stopped
    # serving a purpose. I suspect that if this is not done, on long
    # sessions with multiple widgets, the program might start hogging huge
    # amounts of memory, or even start segfaulting.
    #
    def _delete_widget_ref(self,
                           widget_a):

        # iterate over list and remove widgets which have been closed.
        # This will allow python's garbage collector to delete them.
        #
        self.child_widget_list = [
            w for w in self.child_widget_list if not w == widget_a]


# <*** THIS IS THE WINDOW MANAGER FOR THE ENTIRE PROJECT ***>
#
global window_manager
window_manager = DemoWindowManager()
