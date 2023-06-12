#!/usr/bin/env python

# file: $(NEDC_NFC)/src/main.py
#
# revision history:
#  20210614 (JP): updated the location for the release
#  20160613 (MT): initial version
#
# This script calls the Main class to run a shell of the demo.
# ------------------------------------------------------------------------------

# import modules
#
from pyqtgraph.Qt import QtGui
import os
import sys
from classes.demo_event_loop import DemoEventLoop

# add bin directory path for importing necessary modules
#
import nedc_file_tools as nft
import nedc_cmdl_parser as ncp

current_path = os.path.dirname(os.path.realpath(__file__))

NEDC_HELP_FILE = current_path + "/resources/nedc_demo_help.txt"
NEDC_USAGE_FILE = current_path + "/resources/nedc_demo_usage.txt"

# ------------------------------------------------------------------------------
#
# the main program starts here
#
# ------------------------------------------------------------------------------

# method: main
#
# arguments: none
#
# return: none
#
# This method calls Main and the necessary functions to display the demo window
#


def main():

    # declare the parser
    #
    parser = ncp.Cmdl(NEDC_USAGE_FILE, NEDC_HELP_FILE)
    parser.add_argument("file_arguments",  type=str, nargs='*')
    parser.add_argument("--mtg", "-m", type=str)

    # parse the user's arguments
    #
    args = parser.parse_args()
    montage_file_to_use = args.mtg
    in_files = args.file_arguments

    # loop over files passed in accumulate edfs to open
    #
    files_to_open = []
    for file_argument in in_files:

        # if user passes in a line by line list of files, lets loop over
        # it and accumulate the files listed
        #
        if '.list' in file_argument or '.txt' in file_argument:
            with open(file_argument) as line_by_line_file_list:
                contents = line_by_line_file_list.readlines()
                contents = [line.strip() for line in contents]
                for line in contents:
                    files_to_open.append(line)
        else:
            files_to_open.append(file_argument)

    # Qt boilerplate - absolutely necessary
    #
    app = QtGui.QApplication([])

    # loop over files to open and open if file is an edf. If no
    # files_to_open have been accumulated, than launch a single event loop
    # without a file
    #
    if files_to_open:
        for file_to_open in files_to_open:
            if file_to_open.endswith(".edf"):
                file_to_read_on_init = os.path.abspath(file_to_open)
                initial_loop = DemoEventLoop(
                    montage_file_to_use=montage_file_to_use)
                initial_loop.open_edf_file(file_to_read_on_init)

    else:
        initial_loop = DemoEventLoop(montage_file_to_use=montage_file_to_use)

    # more Qt boilerplate
    #
    QtGui.QApplication.instance().exec_()

    # clean up and exit gracefully
    #
    sys.exit(0)


# begin gracefully
#
if __name__ == "__main__":
    main()
#
# end of file
