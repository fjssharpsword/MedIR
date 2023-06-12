#!/usr/bin/env python

# file: $nedc_nfc/class/python/nedc_sys_tools/nedc_cmdl_parser.py
#
# revision history:
#
# 20200516 (JP): changed the format of options to match the Posix standard
# 20200515 (JP): made debug, verbosity and help included automatically
# 20170716 (JP): Upgraded to using the new annotation tools.
# 20170709 (JP): cleaned up a few things that were hardcoded
# 20160412 (MT): changed the print_usage method to instead print usage
#                messages based on files
# 20151217 (MG): initial version
#
# This file contains classes that handle command line parsing.
#------------------------------------------------------------------------------

# import required system modules
#
import os
import sys
import argparse

# import NEDC modules
#
import nedc_file_tools as nft
import nedc_debug_tools as ndt

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define some standard argument values
#
ARG_OEXT = "--ext"
ARG_ABRV_OEXT = "-e"

ARG_ODIR = "--odir"
ARG_ABRV_ODIR = "-o"

ARG_RDIR = "--rdir"
ARG_ABRV_RDIR = "-r"

ARG_USAG = "--usage"
ARG_DBGL = "--debug_level"
ARG_VRBL = "--verbosity_level"

# define some default values
#
DEF_ODIR = "./output"
DEF_RDIR = None

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: Cmdl
#
# This class inherits the argparse ArgumentParser object.
# This class overloads several display functions to use our file-based
# versions of the help and usage messages.
#
class Cmdl(argparse.ArgumentParser):
    
    # method: Cmdl::constructor
    #
    # arguments:
    #  usage: a short explanation of the command that is printed when 
    #         it is run without argument.
    #  help: a full explanation of the command that is printed when 
    #        it is run with -help argument.
    #
    # return: None
    #
    def __init__(self, usage, help):

        # set the class name
        #
        Cmdl.__CLASS_NAME__ = self.__class__.__name__

        # declare class data
        #
        if self.set_msgs(usage, help) == False:
            print("Error: %s (line: %s) %s::%s: files not found (%s %s)" %
                  (__FILE__, ndt.__LINE__, Cmdl.__CLASS_NAME__, ndt.__NAME__,
                   usage, help))
            sys.exit(os.EX_SOFTWARE)

        # call the parent class constructor
        #
        argparse.ArgumentParser.__init__(self)

        # add the standard NEDC options debug and verbosity:
        #  note that help ("-h", "--help" are included by default)
        #  note also that we don't provide abbreviations for
        #  debug and verbosity
        #
        self.add_argument(ARG_USAG)
        self.add_argument(ARG_DBGL, type = str)
        self.add_argument(ARG_VRBL, type = str)

        # exit gracefully
        #
        return None
    #                                                                          
    # end of method

    # method: Cmdl::set_msgs
    #
    # arguments:
    #  usage: a file containing a one line usage message
    #  help: a file containing the help message
    #
    # return: none
    #
    # this method is used to set the usage and help messages. it is called
    # from the arg parser class. note that filename expansion is done.
    #
    def set_msgs(self, usage, help):

        # expand the filenames (because they might contain env variables)
        #
        self.usage_d = nft.get_fullpath(usage)
        self.help_d = nft.get_fullpath(help)

        if (self.usage_d is None) or (self.help_d is None):
            print("Error: %s (line: %s) %s::%s: files not found (%s %s)" %
                  (__FILE__, ndt.__LINE__, Cmdl.__CLASS_NAME__, ndt.__NAME__,
                   usage, help))
            return False
        else:
            return True
    #                                                                          
    # end of method

    # method: Cmdl::print_usage
    #
    # arguments:
    #  file: output stream
    #
    # return: a boolean value indicating status
    #
    # this method is used to print the usage message from a file. note that
    # this method must not return, so it exits directly. it is called
    # from within the argpars class - we don't call it directly.
    #
    def print_usage(self, file = 'stderr'): 
        
        # open the file
        #
        try: 
            fp = open(self.usage_d, nft.MODE_READ_TEXT) 
        except IOError: 
            print("Error: %s (line: %s) %s::%s: file not found (%s)" %
                  (__FILE__, ndt.__LINE__, Cmdl.__CLASS_NAME__, ndt.__NAME__,
                   self.usage_d))
            sys.exit(os.EX_SOFTWARE)

        # print the file without a linefeed at the end
        #
        usage_file = fp.read()
        print(usage_file, end = nft.STRING_EMPTY)

        # exit gracefully
        #
        sys.exit(os.EX_OK)
    #                                                                          
    # end of method

    # method: Cmdl::format_help
    #
    # arguments: none
    #
    # return:
    #  help_file: string containing text from help file
    #
    # this class is used to define the specific help message to be used. it
    # is called from within argparse - we don't call it directly.
    #
    def format_help(self):

        # open the file
        #
        try: 
            fp = open(self.help_d, nft.MODE_READ_TEXT) 
        except IOError: 
            print("Error: %s (line: %s) %s::%s: file not found (%s)" %
                  (__FILE__, ndt.__LINE__, Cmdl.__CLASS_NAME__, ndt.__NAME__,
                   self.help_d))
            sys.exit(os.EX_SOFTWARE)

        # read the help file
        #
        help_file = fp.read()
        
        # close the file
        #
        fp.close()

        # exit gracefully
        #
        return help_file
    #                                                                          
    # end of method

    # method: Cmdl::parse_args
    #
    # arguments: none
    #
    # return:
    #  args: a list containing the arguments
    #
    # this class is used to wrap the parent class parser function
    # and update the values of the debug and verbosity classes.
    #
    def parse_args(self):

        # call the parent class parser
        #
        args, unknown = argparse.ArgumentParser.parse_known_intermixed_args(self)

        if args is None:
            print("Error: %s (line: %s) %s::%s: error parsing arguments (%s)" %
                  (__FILE__, ndt.__LINE__, Cmdl.__CLASS_NAME__, ndt.__NAME__,
                   sys.argv))
            sys.exit(os.EX_SOFTWARE)

        # set debug level and verbosity
        #
        dbgl = ndt.Dbgl()
        dbgl.set(name = args.debug_level)
        vrbl = ndt.Vrbl()
        vrbl.set(name = args.verbosity_level)

        # check for a usage flag
        #
        if args.usage is not None:
            print_usage()

        # exit gracefully
        #
        return args
    #                                                                          
    # end of method

#
# end of file
