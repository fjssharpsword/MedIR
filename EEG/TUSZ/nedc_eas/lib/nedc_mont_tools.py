#!/usr/bin/env python

# files: $(NEDC_NFC)/class/python/.../nedc_mont_tools.py
#                                                                              
# revision history:                                                            
#
# 20220621 (PM): fixed get_subtrahends() to handle montages that only have one 
#                part 
# 20200708 (JP): code review
# 20200707 (TC): changed the interface
# 20200623 (TC): initial version
#
# This file contains a Python implementation of functions to manipulate
# montages.
#
#------------------------------------------------------------------------------

# import required system modules                                               
#
import os
import sys

# import NEDC modules
#
import nedc_debug_tools as ndt
import nedc_file_tools as nft

#------------------------------------------------------------------------------
#                                                                              
# global variables are listed here
#                                                                              
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define variables that delimit the parts of a line
#
DEF_DELIM_MONTAGE = "montage ="
DEF_DELIM_OPER = nft.DELIM_SPACE + nft.STRING_DASHDASH + nft.DELIM_SPACE

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: Montage
#
# This class is a Python implementation of our C++ code to manipulate montages.
#
class Montage:

    # define static variables for debug and verbosity
    #
    dbgl_d = ndt.Dbgl()
    vrbl_d = ndt.Vrbl()

    # method: Montage::constructor
    #
    # arguments:
    #   mfile: montage file
    #
    # returns: none
    #
    def __init__(self, fname = None):

        # set the class name
        #
        Montage.__CLASS_NAME__ = self.__class__.__name__
        
        # display debug information
        #
        if self.dbgl_d == ndt.FULL:
            print("%s (line: %s) %s::%s: contructing a montage (%s)" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__, fname))

        # opening the file
        #
        if fname is not None:
            self.montage_d = self.load(fname)
            if self.montage_d is None:
                print("Error: %s (line: %s) %s: %s (%s)" %
                      (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                       "cannot load file", fname))
                sys.exit(os.EX_SOFTWARE)
        
        #
        # end of method

    # method: Montage::load
    #
    # arguments:
    #  fname: montage filename
    #
    # returns: a montage as a dictionary
    #
    # This method loads (and parses) a montage file.
    #
    def load(self, fname):

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: loading a montage (%s)" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__, fname))

        # open the montage file
        #
        fp = open(fname, nft.MODE_READ_TEXT)
        if fp is None:
            print("Error: %s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__, "cannot open file", fname))
            return None

        # define montage dict
        #
        self.montage_d = {}
    
        # parse the montage file
        #
        flag_pblock = False
        for line in fp:

            # check for a delimiter
            #
            if line.startswith(DEF_DELIM_MONTAGE):
                try:
                    # clean up the line
                    #
                    str = line\
                            .replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                            .replace(nft.DELIM_TAB, nft.DELIM_NULL) \
                                                .split(nft.DELIM_COMMA)
            
                    # separate the fields:
                    #  remove montage numbers
                    #
                    parts = str[1].split(nft.DELIM_COLON)
                
                    # slip double dash between minuend and subtrahend
                    #
                    parts[1] = parts[1].split(DEF_DELIM_OPER)

                    # remove any unnecessary space between items
                    #
                    parts[0] = parts[0].strip()
                    parts[1] = [channel.strip() for channel in parts[1]]
                   
                    # append name and minuend/subtrahend to dict:
                    #  [('FP1-F7', ['EEG FP1-REF', 'EEG F7-REF']),
                    #   ('F7-T3', ['EEG F7-REF', 'EEG T3-REF']), ...]
                    #
                    self.montage_d.update({parts[0]:parts[1]})
                    flag_pblock = True
                
                except:
                    # return None when there is a syntax error
                    #
                    flag_pblock = False
                    print("Error: %s (line: %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                           "cannot parse montage", fname))
                    break
        
        # close the file
        #
        fp.close()    

        # make sure we found a montage block
        #
        if flag_pblock == False:
            fp.close()
            print("Error: %s (line: %s) %s::%s: invalid montage file (%s)" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__, fname))
            return None
        
        # exit gracefully
        #
        return self.montage_d

    # method: Montage::check
    #
    # arguments:
    #  isig: a dict of signal data
    #  montage: a montage dict
    #
    # returns: a boolean value indicating status
    #
    # This method checks if a list of channel labels is consistent
    # with the montage.
    #
    def check(self, isig, montage):
        
        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: checking a montage" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__), montage)
            
        # get a list of channels from input signal:
        #  use a fast technique:
        #   https://stackoverflow.com/questions/16819222/
        #    how-to-return-dictionary-keys-as-a-list-in-python
        #
        chan_labels = [*isig]
        
        # loop over a montage dict to find a missing channel
        #
        missing_channels = []
        
        for key in montage:

            # check minuend and subtrahend channels if it's not in
            # edf chan labels
            #
            for channel in montage[key]:
                if channel not in chan_labels:
                    missing_channels.append(channel)

        # check if there is a missing channel
        #
        if missing_channels:
            print("Error: %s (line: %s) %s::%s: missing channels" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__), missing_channels)
            return False
        
        # exit gracefully
        #
        return True

    # method: Montage::apply
    #
    # arguments:
    #  isig: signal data
    #  montage: a montage dict
    #
    # returns: a new signal that is a result of the montage operation
    #
    # This method applies montage to a signal.
    #
    def apply(self, isig, montage):

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: applying a montage" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__))

        # save the montage to class data
        #
        self.montage_d = montage
        
        # make sure every channel in the montage matches the signal
        #
        status = self.check(isig, montage)
        if status is False:
            print("Error: %s (line: %s) %s: montage invalid" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                  ndt.__NAME__))
            return None
        
        # loop over every channel in the montage
        #
        osig = {}
        for key in self.montage_d:

            # assign the operands and subtract the signals
            #
            noperands = len(self.montage_d[key])
            if noperands == int(2):
                minuend = isig[self.montage_d[key][0]]
                subtrahend = isig.get(self.montage_d[key][1])
                osig[key] = minuend - subtrahend
            else:
                osig[key] = isig[self.montage_d[key][0]]

        # exit gracefully
        #
        return osig
    
    # method: Montage::get_minuends
    #
    # arguments: None
    #
    # returns: a list of minuends from montage
    #
    # This method gets the minuends from a montage
    #
    def get_minuends(self):

        # display debug information
        #
        if self.dbgl_d == ndt.FULL:
            print("%s (line: %s) %s::%s: fetching minuends" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__))

        # check montage is loaded
        #
        if self.montage_d is None:
            print("%s (line: %s) %s::%s: no montage loaded" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__))
            return None

        # find the minuends
        #
        minuends = []
        for key in self.montage_d:
            minuends.append(self.montage_d[key][0])

        # exit gracefully
        #
        return minuends

    # method: Montage::get_subtrahends
    #
    # arguments:
    #  None
    #
    # returns: a list of subtrahends from a montage
    #
    # This method gets the subtrahends from a montage.
    #
    def get_subtrahends(self):
    
        # display debug information
        #
        if self.dbgl_d == ndt.FULL:
            print("%s (line: %s) %s::%s: fetching subtrahends" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__))

        # check montage is loaded
        #
        if self.montage_d is None:
            print("%s (line: %s) %s::%s: no montage loaded" %
                  (__FILE__, ndt.__LINE__, Montage.__CLASS_NAME__,
                   ndt.__NAME__))
            return None

        # find the subtrahends
        #
        subtrahends = []
        for key in self.montage_d:
            
            # check if there's a label there
            #
            try:
                subtrahends.append(self.montage_d[key][1])

            # if no label then move on
            #
            except:
                continue

        # exit gracefully
        #
        return subtrahends

#
# end of class

#
# end of file
