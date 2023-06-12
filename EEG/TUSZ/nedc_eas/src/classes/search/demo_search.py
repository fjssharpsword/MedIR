#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/search/demo_search.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
import classes.demo_event_loop

from classes.ui.search.demo_search_user_interface import DemoSearchUserInterface
from classes.ui.search.demo_text_report_widget import DemoTextReportWidget
from .demo_search_helpers import *

# class: DemoSearch
#
# This class acts as a controller for the search ui, and uses helper
# functions to conduct its queries.
#
class DemoSearch:
    def __init__(self,
                 current_file_a=None):

        # remember the file to allow for calculating similarity to other files
        #
        self.current_file = current_file_a

        # initialize the search user interface. pass it the functions it needs
        # to be able to set up buttons that launch new text and signal widgets
        #
        self.ui = DemoSearchUserInterface(
            self.launch_new_demo_instance,
            self.launch_new_text_instance)

        # connect search button and enter key in the ui to self.do_search
        #
        self.ui.sig_do_search.connect(self.do_search)

    # method: launch_new_demo_instance
    #
    # arguments:
    #  -file_name: (optional) the path for the edf the demo should load
    #              probably the only time this might be called without a 
    #              file_name is the first instance in main, 
    #              if no initial file to open is specified in sys.argv
    #
    # returns: none
    #
    # this method creates a DemoEventLoop and optionally has it open an edf.
    # typically it is called by a button-press-event in self.ui,
    # but it is also called in main when the program is first starting.
    #
    def launch_new_demo_instance(self,
                                 file_name=None):

        # open a new event loop and associated main window
        #
        loop = classes.demo_event_loop.DemoEventLoop()

        # if the event loop has a file associated with it, have the
        # new loop open it, and remember the file to allow for
        # calculating similarity to other files
        #
        if file_name is not None:
            loop.open_edf_file(file_name)
            self.current_file = file_name

    # method: launch_new_text_instance
    #
    # arguments:
    #  -file_name_a: the path to the txt report to open
    #
    # returns: none
    #
    # this method launches a text report widget. it is only ever
    # called by a button-press-event in self.ui
    #
    def launch_new_text_instance(self,
                                 file_name_a):

        # remember the file to allow for calculating similarity to other files
        # 
        self.current_file = file_name_a

        # launch a text report widget
        #
        text_report_widget = DemoTextReportWidget()

        # get and display the contents of the text file
        #
        with open(file_name_a) as txt_source:
            text = txt_source.read()
        text_report_widget.text_area.setText(text)

    # method: do_search
    #
    # arguments: none
    #
    # returns: none
    #
    # this method is clears the search results area in the ui and collects the
    # new query, along with checkbox info. It then tries to get the search
    # results for the query / checkbox info. If it gets search results,
    # it will try to search via similarity, and then display the results. If it
    # does not get results, it will make the ui display an indicative message.
    #
    def do_search(self):

        # remove old search results
        #
        self.ui.clear_search_results()

        # get text from text entry field, bool dict from checkboxes
        #
        in_string = str(self.ui.search_area.search_line_edit.text())
        checkbox_dict = self.ui.param_area.get_checkbox_dict()

        # probably will rewrite this, but probably will be similar in effect
        #
        cohort_dir = get_cohort_dir(in_string)

        # if a hit is found, then make and display entries
        #
        if cohort_dir is not None:

            # get a dictionary of dictionarys (keys indicate values
            # similarity matrix row numbers)
            #
            results, similarity_file = get_results(
                cohort_dir,
                checkbox_dict)

            # check to see if we self.current_file matches any of the results
            # (matters for similarity sorting). If so, identify it via index.
            #
            current_edf_index_in_results = \
                self.get_index_of_current_edf_file_in_results_if_any(results)

            # sort the results according to the similarity matrix.
            # (doesn't do what it claims right now if
            # current_edf_index_in_results is None, as the helper
            # methods do not support this
            #
            sorted_results = sort_results(
                results,
                similarity_file,
                current_edf_index_in_results)

            # if we have results, iterate over them and:
            # 1) read the dictionary associated with the entry
            # 2) read the text file (report)
            # 3) pass this information to the ui so it can add an entry
            #
            if sorted_results:
                for i in sorted_results:

                    # read the dictionary associated with the file
                    #
                    edf_name = sorted_results[i]['edf_name']
                    edf_file = sorted_results[i]['edf_file']
                    txt_file = sorted_results[i]['txt']

                    # get the text from the text file
                    #
                    extract_text = self.extract_text_from_file(txt_file)

                    # pass the information to the ui to add an entry
                    #
                    self.ui.add_entry(edf_file,
                                      txt_file,
                                      extract_text,
                                      edf_name)

            # if no results, indicate none found
            #
            else:
                try:
                    self.ui.results_area.display_none_found()
                except:
                    pass

        # if no cohort_dir, of course no result are found! indicate none found
        #
        else:
            try:
                self.ui.results_area.display_none_found()
            except:
                pass

    # method: extract_text_from_file
    #
    # arguments:
    #  -text_file_name_a: the path to the text file
    #
    # returns:
    #  - extract_text: the text read in from the file
    #
    # this method
    #
    def extract_text_from_file(self,
                               text_file_name_a):

        # initialize empty string to hold text input
        #
        extract_text = ""

        # open text file and iterate line by line
        #
        with open(text_file_name_a) as txt_source:
            for line in txt_source:

                # prevents formatting issue (extra line breaks)
                #
                if len(line) > 2:

                    # store line with html line break
                    #
                    extract_text = extract_text + line + "<br>"

        return extract_text

    # method: get_index_of_current_edf_file_in_results_if_any
    #
    # arguments:
    #  -results_a: a (integer-keyed) dictionary of dictionaries, each of which
    #              hold paths to edf_file, text_file, and a name for edf
    #
    # returns: the index of matching result if found, other wise None
    #          
    # NOTE: This function will return None if no matching result is found.
    #
    def get_index_of_current_edf_file_in_results_if_any(self,
                                                        results_a):

        # iterate over results and check if self.current_file matches any
        #
        for i in results_a:

            # get path to edf file at current index
            #
            edf_file = results_a[i]['edf_file']

            # necessary check for case where self.current_file was not yet set
            #
            if hasattr(self, 'current_file'):

                # if result matches self.current file, return index
                #
                if edf_file == self.current_file:
                    return i

