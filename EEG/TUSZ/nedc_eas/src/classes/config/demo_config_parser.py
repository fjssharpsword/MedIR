#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/config/demo_config_parser.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------

import configparser
import os

from ast import literal_eval as make_tuple
from .demo_montage_reader import DemoMontageModule

#----------------------------------------------------------------------------------
#
# file: DemoConfigParser
#
# this file parses all sections, barring [Montage], of preferences.cfg into
# dictionaries, also writes to preferences.cfg from a dictionary
#
class DemoConfigParser(configparser.ConfigParser, object):

    # method: __init__
    #
    # arguments: config_file_a is the file to parsed
    #
    # returns: None
    #
    # this method initializes DemoConfigParser and sets up DemoMontageModule
    #
    def __init__(self,
                 config_file_a=None):

        configparser.ConfigParser.__init__(self)

        # create a file for default configuration
        # (unnecessary, we could just point self directly to config_file_a)
        #
        self.config_file = config_file_a

        self.d_sep_char = os.sep

        self.readfp(open(self.config_file))

        self.montage_reader = DemoMontageModule(self.config_file)

        self.montage_lines = self.montage_reader.get_montage_lines_for_writing()
    #
    # end of method

    # method: get_sect_dict
    #
    # arguments:
    # - section: a string that should match a section label in the config file
    # - do_tuple: a boolean that determines if the function formats
    #   the values in the returned dictionary as tuples
    #
    # returns:
    # - config_section_dictionary: a dictionary for a section of
    #   configuration settings
    #
    # this method parses the config file for 'sections'
    # (sections delimited in the config file by labels like: [<SectionName>] )
    # and returns a dictionary of settings for that section
    #
    # TODO: Make do_tuple into a list that is empty by default, and
    # iterate over the list, so that an empty list is ignored, but the
    # values that need to be 'tupleized' are.
    #
    def get_sect_dict(self,
                      section,
                      do_tuple=False,
                      no_quotes=False):

        # declare empty dictionary
        #
        config_section_dictionary = {}

        # get list of options in a 'section' argument
        #
        options = self.options(section)

        # loop over all options (keys) and collect their associated values
        #
        for option in options:
            try:
                # collect the values associated with each key
                #
                config_section_dictionary[option] = self.get(section, option)

                if config_section_dictionary[option] == -1:
                    print("skip: %s" % option)

            # handle the case of malformed configuration text
            #
            except:
                print("exception on %s!" % option)
                config_section_dictionary[option] = None

            # if the call called for a dictionary of tuples
            #
            if do_tuple is True:

                # check if the option is a float/tuple
                #
                if config_section_dictionary[option].isalpha() is False:
                    # try to make a tuple out of the each value parsed
                    try:
                        config_section_dictionary[option] = make_tuple(
                            config_section_dictionary[option])
                    except Exception as e:
                        print("unable to make tuple on %s" % option)
                        
            # if the call called for a dictionary of tuples
            # TODO: delete this functionality if unused.
            #
            if no_quotes is True:

                # try to make a tuple out of the each value parsed
                try:
                    config_section_dictionary[option] = \
                        config_section_dictionary[option].replace('"', '')
                except:
                    print("unable to remove quotes on %s" % option)

        # return dictionary
        #
        return config_section_dictionary
    #
    # end of method

    # method: set_config_file_from_dict
    #
    # arguments:
    #  - section_a: section of preferences.cfg to be editted  e.g. [MainWindow]
    #  - dict_new_values_a: dictionary to be written e.g. {'initial_time_scale': 10}
    #
    # returns: None:
    #
    # this method sets up ConfigParser to write to preferences.cfg
    def set_config_file_from_dict(self,
                                 section_a,
                                 dict_new_values_a):

        for key in dict_new_values_a:
            try:
                self.set(section_a, key, dict_new_values_a[key])
            except:
                print("exception on %s!" % key)
    #
    # end of method

    # method: write_config_file
    #
    # arguments: None
    #
    # returns: None
    #
    # this method overwrites preferences.cfg with new information, also
    # appends [Montage] section to preferences.cfg
    #
    def write_config_file(self):

        self.remove_section('Montage')

        # write to config override file
        #
        self.config_file = os.path.expanduser('~') + self.d_sep_char + ".eas.cfg"
        self.write(open(self.config_file, 'w'))
    #
    # end of method

    # method: write_montage_section_to_file
    #
    # arguments: None
    #
    # returns: None
    #
    # appends [Montage] section manually with identical format to ConfigParser
    #
    def write_montage_to_file(self,
                              montage_file_a):
        self.remove_section('Montage')
        self.write(open(self.config_file, 'w'))
        with open(self.config_file, 'a') as config_file:
            config_file.write('[Montage]\n')
            config_file.write('prev_montage = ' + montage_file_a)
    #
    # end of method
