#!/usr/bin/env python

# file: $(NEDC_NFC)/class/python/nedc_edf_tools/nedc_edf_tools.py
#                                                                              
# revision history:                                                            
#
# 20210809 (JP): added get_header_from_file
# 20210809 (JP): fixed a bug with the cleanup method
# 20200607 (JP): started over using our own code because the open source
#                versions of this software were not adequately robust
# 20200219 (NS): updated to python3
# 20170622 (NC): refactored code into /class/ with more ISIP standards        
# 20160511 (MT): refactored code to comply with ISIP standards and to allow   
#                input from the command line                                  
# 20141212 (MG): modified to support edf files with non-eeg channels          
# 20141020 (MG): changed load_edf to read an edf file between two specified   
#                times (t1,t2)                                               
# 20140812 (MG): initial version                                              
#                                                                              
# This file contains a Python implementation of the C++ class Edf.
#------------------------------------------------------------------------------

# import required system modules                                               
#
from collections import OrderedDict
import scipy.signal as signal
import os
import numpy as np
import re
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

#------------------------------------------------------------------------------
#                                                                              
# this special section defines an Edf header byte by byte
#                                                                              
#------------------------------------------------------------------------------

# section (1): version information
#
EDF_VERS_NAME = "version"
EDF_VERS_BSIZE = int(8)
EDF_VERS = b"0       "
    
# section (2): patient information
#
EDF_LPTI_BSIZE = int(80)
EDF_LPTI_TSIZE = int(119)

EDF_LPTI_PATIENT_ID_NAME = "ltpi_patient_id"
EDF_LPTI_GENDER_NAME = "ltpi_gender"
EDF_LPTI_DOB_NAME = "ltpi_dob"
EDF_LPTI_FULL_NAME_NAME = "ltpi_full_name"
EDF_LPTI_AGE_NAME = "ltpi_age"

# section (3): local recording information
#
EDF_LRCI_BSIZE = int(80)
EDF_LRCI_TSIZE = EDF_LPTI_TSIZE
EDF_LRCI_RSIZE = EDF_LPTI_BSIZE

EDF_LRCI_START_DATE_LABEL = "lrci_start_date_label"
EDF_LRCI_START_DATE = "lrci_start_date"
EDF_LRCI_EEG_ID = "lrci_eeg_id"
EDF_LRCI_TECH = "lrci_tech"
EDF_LRCI_MACHINE = "lrci_machine"

# section (4): general header information
#
EDF_GHDI_BSIZE = int(8 + 8 + 8 + 5 + 39 + 8 + 8 + 4)
EDF_GHDI_TSIZE = EDF_LPTI_TSIZE

EDF_GHDI_START_DATE = "ghdi_start_date"
EDF_GHDI_START_TIME = "ghdi_start_time"
EDF_GHDI_HSIZE = "ghdi_hsize"
EDF_GHDI_FILE_TYPE = "ghdi_file_type"
EDF_GHDI_RESERVED = "ghdi_reserved"
EDF_GHDI_NUM_RECS = "ghdi_num_recs"
EDF_GHDI_DUR_REC = "ghdi_dur_rec"
EDF_GHDI_NSIG_REC = "ghdi_nsig_rec"

# section (5): channel-specific information
#
EDF_LABL_BSIZE = int(16)
EDF_TRNT_BSIZE = int(80)
EDF_PDIM_BSIZE = int( 8)
EDF_PMIN_BSIZE = int( 8)
EDF_PMAX_BSIZE = int( 8)
EDF_DMIN_BSIZE = int( 8)
EDF_DMAX_BSIZE = int( 8)
EDF_PREF_BSIZE = EDF_TRNT_BSIZE
EDF_RECS_BSIZE = int( 8)

EDF_CHAN_LABELS = "chan_labels"
EDF_CHAN_TRANS_TYPE = "chan_trans_type"
EDF_CHAN_PHYS_DIM = "chan_phys_dim"
EDF_CHAN_PHYS_MIN = "chan_phys_min"
EDF_CHAN_PHYS_MAX = "chan_phys_max"
EDF_CHAN_DIG_MIN = "chan_dig_min"
EDF_CHAN_DIG_MAX = "chan_dig_max"
EDF_CHAN_PREFILT = "chan_prefilt"
EDF_CHAN_REC_SIZE = "chan_rec_size"

# section (6): derived values
#
EDF_SAMPLE_FREQUENCY = "sample_frequency"
EDF_NUM_CHANNELS_SIGNAL = "num_channel_signal"
EDF_NUM_CHANNELS_ANNOTATION = "num_channels_annotation"

# other important definitions
#
EDF_SIZEOF_SHORT = int(2)
EDF_DEF_CHAN = int(-1)
EDF_DEF_DBG_NF = int(10)
EDF_FTYP_NAME = "ftype"
EDF_FTYP_BSIZE = int(5)
EDF_FTYP = "EDF  "
EDF_SIG_MAXVAL = int(32767)
EDF_ANNOTATION_KEY = "ANNOTATION"

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

# function: set_limits
#
# arguments:
#  long f1: desired first index (input)
#  long f2: desired number of items (input)
#  long fmax: maximum number available (input)
#
# return: a boolean value indicating status
#  long& n1: first index (output)
#  long& n2: last_index (output)
#
# This method returns a range [n1, n2] that is clipped based on the inputs.
#
def set_limits(f1, f2, fmax):

    # initial the outuput to the max range
    #
    n1 = int(0)
    n2 = int(fmax)
  
    # clip n1
    #
    if f1 > int(0):
        n1 = min(f1, fmax - 1)

    # clip n2
    #
    if f2 == int(0):
        n2 = n1
        return(n1, n2)
    elif f2 > int(0):
        n2 = min(n1 + f2, n2)

    # exit gracefully
    #
    return(n1, n2)

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: Edf
#
# This is class is a Python implementation of the C++ class Edf.
# Its interface parallels that class.
#
class Edf:

    # define static variables for debug and verbosity
    #
    dbgl_d = ndt.Dbgl()
    vrbl_d = ndt.Vrbl()

    # define a dictionary to hold header information
    #
    h_d = {}

    # method: Edf::constructor
    #
    # arguments: none
    #
    # returns: none
    #
    def __init__(self):

        # set the class name
        #
        Edf.__CLASS_NAME__ = self.__class__.__name__

    #
    # end of method

    # method: Edf::is_edf
    #
    # arguments:
    #  fname: path to edf file
    #
    # returns: True if file is an edf file
    #
    # This method looks at the beginning 8 bytes of the edf file, and decides
    # if the file is an edf file.
    #
    def is_edf(self, fname):

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: checking for an edf file (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))

        # open the file
        #
        fp = open(fname, nft.MODE_READ_BINARY)
        if fp is None:
            print("Error: %s (line: %s) %s::%s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))
            return False
            
        # make sure we are at the beginning of the file and read
        #
        fp.seek(0, os.SEEK_SET)
        barray = fp.read(EDF_VERS_BSIZE)

        # close the file and reset the pointer
        #
        fp.close()

        # if the beginning of the file contains the magic sequence
        # then it is an edf file
        #
        if barray == EDF_VERS:
            return True
        else:
            return False

    # method: Edf::print_header
    #
    # arguments:
    #  fp: stream to be used for printing
    #  prefix: a prefix character to use for printing
    #
    # returns: a boolean value indicating status
    #
    # This method assumes the header has been loaded and prints it.
    #
    def print_header(self, fp, prefix = nft.DELIM_TAB):

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: printing Edf header" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        # (1) version information
        #
        # note we conver this to a string to be compatible with the c++
        # version of this code
        #
        fp.write("%sBlock 1: Version Information\n" % (prefix))
        fp.write("%s version = [%s]\n\n" %
                 (prefix, str(self.h_d[EDF_VERS_NAME], nft.DEF_CHAR_ENCODING)))

        # (2) local patient information
        #
        fp.write("%sBlock 2: Local Patient Information\n" % (prefix))
        fp.write("%s lpti_patient_id = [%s]\n" % 
	         (prefix, self.h_d[EDF_LPTI_PATIENT_ID_NAME]))
        fp.write("%s lpti_gender = [%s]\n" %
	         (prefix, self.h_d[EDF_LPTI_GENDER_NAME]))
        fp.write("%s lpti_dob = [%s]\n" %
	         (prefix, self.h_d[EDF_LPTI_DOB_NAME]))
        fp.write("%s lpti_full_name = [%s]\n" %
	         (prefix, self.h_d[EDF_LPTI_FULL_NAME_NAME]))
        fp.write("%s lpti_age = [%s]\n\n" %
	         (prefix, self.h_d[EDF_LPTI_AGE_NAME]))

        # (3) local recording information
        #
        fp.write("%sBlock 3: Local Recording Information\n" % (prefix))
        fp.write("%s lrci_start_date_label = [%s]\n" %
	         (prefix, self.h_d[EDF_LRCI_START_DATE_LABEL]))
        fp.write("%s lrci_start_date = [%s]\n" %
	         (prefix, self.h_d[EDF_LRCI_START_DATE]))
        fp.write("%s lrci_eeg_id = [%s]\n" %
	         (prefix, self.h_d[EDF_LRCI_EEG_ID]))
        fp.write("%s lrci_tech = [%s]\n" %
	         (prefix, self.h_d[EDF_LRCI_TECH]))
        fp.write("%s lrci_machine = [%s]\n\n" %
	         (prefix, self.h_d[EDF_LRCI_MACHINE]))

        # (4) general header information
        #
        fp.write("%sBlock 4: General Header Information\n" % (prefix))
        fp.write("%s ghdi_start_date = [%s]\n" %
	         (prefix, self.h_d[EDF_GHDI_START_DATE]))
        fp.write("%s ghdi_start_time = [%s]\n" %
	         (prefix, self.h_d[EDF_GHDI_START_TIME]))
        fp.write("%s ghdi_hsize = [%ld]\n" %
	         (prefix, self.h_d[EDF_GHDI_HSIZE]))
        fp.write("%s ghdi_file_type = [%s]\n" %
	         (prefix, self.h_d[EDF_GHDI_FILE_TYPE]))
        fp.write("%s ghdi_reserved = [%s]\n" %
	         (prefix, self.h_d[EDF_GHDI_RESERVED]))
        fp.write("%s ghdi_num_recs = [%ld]\n" %
	         (prefix, self.h_d[EDF_GHDI_NUM_RECS]))
        fp.write("%s ghdi_dur_rec = [%lf]\n" %
	         (prefix, self.h_d[EDF_GHDI_DUR_REC]))
        fp.write("%s ghdi_nsig_rec = [%ld]\n\n" %
	         (prefix, self.h_d[EDF_GHDI_NSIG_REC]))

        # (5) channel-specific information
        #
        fp.write("%sBlock 5: Channel-Specific Information\n" % (prefix))
        fp.write("%s chan_labels (%ld) = " %
                 (prefix, self.h_d[EDF_GHDI_NSIG_REC]))

        last_chan = self.h_d[EDF_GHDI_NSIG_REC] - 1
        for i in range(0, last_chan):
            fp.write("[%s], " % (self.h_d[EDF_CHAN_LABELS][i]))
        fp.write("[%s]\n" % ((self.h_d[EDF_CHAN_LABELS])[last_chan]))

        fp.write("%s chan_trans_type (%ld) = " %
                 (prefix, self.h_d[EDF_GHDI_NSIG_REC]))
        for i in range(0, last_chan):
                fp.write("[%s], " % (self.h_d[EDF_CHAN_TRANS_TYPE][i]))
        fp.write("[%s]\n" % (self.h_d[EDF_CHAN_TRANS_TYPE][last_chan]))
        
        fp.write("%s chan_phys_dim (%ld) = " %
                 (prefix, self.h_d[EDF_GHDI_NSIG_REC]))
        for i in range(0, last_chan):
            fp.write("[%s], " % (self.h_d[EDF_CHAN_PHYS_DIM][i]))
        fp.write("[%s]\n" % (self.h_d[EDF_CHAN_PHYS_DIM][last_chan]))

        fp.write("%s chan_phys_min (%ld) = " %
                 (prefix, self.h_d[EDF_GHDI_NSIG_REC]))
        for i in range(0, last_chan):
            fp.write("[%10.3f], " % (self.h_d[EDF_CHAN_PHYS_MIN][i]))
        fp.write("[%10.3f]\n" % (self.h_d[EDF_CHAN_PHYS_MIN][last_chan]))

        fp.write("%s chan_phys_max (%ld) = " %
                 (prefix, self.h_d[EDF_GHDI_NSIG_REC]))
        for i in range(0, last_chan):
            fp.write("[%10.3f], " % (self.h_d[EDF_CHAN_PHYS_MAX][i]))
        fp.write("[%10.3f]\n" % (self.h_d[EDF_CHAN_PHYS_MAX][last_chan]))

        fp.write("%s chan_dig_min (%ld) = " %
                 (prefix, self.h_d[EDF_GHDI_NSIG_REC]))
        for i in range(0, last_chan):
            fp.write("[%10ld], " % (self.h_d[EDF_CHAN_DIG_MIN][i]))
        fp.write("[%10ld]\n" % (self.h_d[EDF_CHAN_DIG_MIN][last_chan]))

        fp.write("%s chan_dig_max (%ld) = " %
                 (prefix, self.h_d[EDF_GHDI_NSIG_REC]))
        for i in range(0, last_chan):
            fp.write("[%10ld], " % (self.h_d[EDF_CHAN_DIG_MAX][i]))
        fp.write("[%10ld]\n" % (self.h_d[EDF_CHAN_DIG_MAX][last_chan]))

        fp.write("%s chan_prefilt (%ld) = " %
                 (prefix, self.h_d[EDF_GHDI_NSIG_REC]))
        for i in range(0, last_chan):
            fp.write("[%s], " % (self.h_d[EDF_CHAN_PREFILT][i]))
        fp.write("[%s]\n" % (self.h_d[EDF_CHAN_PREFILT][last_chan]))

        fp.write("%s chan_rec_size (%ld) = " %
                 (prefix, self.h_d[EDF_GHDI_NSIG_REC]))
        for i in range(0, last_chan):
            fp.write("[%10ld], " % (self.h_d[EDF_CHAN_REC_SIZE][i]))
        fp.write("[%10ld]\n" % (self.h_d[EDF_CHAN_REC_SIZE][last_chan]))

        fp.write("%s\n" % (prefix))

        # (6) derived values
        #
        fp.write("%sBlock 6: Derived Values\n" % (prefix))
        fp.write("%s hdr_sample_frequency = %10.1f\n" %
	         (prefix, self.h_d[EDF_SAMPLE_FREQUENCY]))
        fp.write("%s hdr_num_channels_signal = %10ld\n" %
	         (prefix, self.h_d[EDF_NUM_CHANNELS_SIGNAL]))
        fp.write("%s hdr_num_channels_annotation = %10ld\n" %
	         (prefix, self.h_d[EDF_NUM_CHANNELS_ANNOTATION]))
        fp.write("%s duration of recording (secs) = %10.1f\n" %
	         (prefix, (float)(self.h_d[EDF_GHDI_DUR_REC] *
                                  self.h_d[EDF_GHDI_NUM_RECS])))
        
        fp.write("%s per channel sample frequencies:\n" % (prefix))
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            fp.write("%s  channel[%4ld]: %10.1f Hz (%s)\n" %
                         (prefix, i,
	                  self.get_sample_frequency(i),
                          self.h_d[EDF_CHAN_LABELS][i]))
        
        # exit gracfully
        #
        return True

    # method: Edf::print_header_from_file
    #
    # arguments:
    #  fname: input file
    #  fp: stream to be used for printing
    #  prefix: a prefix character to use for printing
    #
    # returns: a boolean value indicating status
    #
    # This opens a file, reads the header, and pretty prints it.
    #
    def print_header_from_file(self, fname, fp, prefix = nft.DELIM_TAB):

        # declare local variables
        #
        
        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: printing Edf header (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))

        # make sure this is an edf file
        #
        if self.is_edf(fname) == False:
            print("Error: %s (line: %s) %s::%s: not an Edf file (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))
            return False

        # open the file
        #
        fp_edf = open(fname, "rb")
        if fp_edf == None:
            print("Error: %s (line: %s) %s::%s: error opening (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))

        # read the header from a file:
        #  note that we will ignore the signal data
        #
        if self.get_header(fp_edf) == False:
            print("Error: %s (line: %s) %s::%s: error opening (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))
            return False

        # print the header
        #
        self.print_header(fp, prefix)

        # exit gracefully
        #
        return True

    # method: Edf::get_header_from_file
    #
    # arguments:
    #  fname: input filename
    #
    # returns: a boolean value indicating status
    #
    # This method reads the header of an edf file given a filename.
    #
    def get_header_from_file(self, fname):

        # open the file
        #
        fp_edf = open(fname, "rb")
        if fp_edf == None:
            print("Error: %s (line: %s) %s::%s: error opening (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))
            return False

        # read the header from a file:
        #  note that we will ignore the signal data
        #
        if self.get_header(fp_edf) == False:
            print("Error: %s (line: %s) %s::%s: error opening (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))
            return False

        # exit gracefully
        #
        return True

    # method: Edf::get_header
    #
    # arguments:
    #  fp: an open file pointer
    #
    # returns: a logical value indicating the status of the get operation
    #
    # This method reads the header of an edf file.
    #
    def get_header(self, fp):

        # declare local variables
        #
        nbytes = int(0)
        num_items = int(0)

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching an Edf header" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        # rewind the file
        #
        fp.seek(0, os.SEEK_SET)

        # (1) version information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (1)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        self.h_d[EDF_VERS_NAME] = fp.read(EDF_VERS_BSIZE)
        if self.h_d[EDF_VERS_NAME] != EDF_VERS:
            return False

        # (2) local patient information
        #
        # unfortunately, some edf files don't contain all the information
        # they should. this often occurs because the deidenitification
        # process overwrites this information. so we zero out the buffers
        # that won't be filled if the information is missing.
        #
        # note also that sometimes this field is blank, so split might
        # not return an adequate number of fields.
        #
        # finally, we want these stored as strings, not bytes
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (2)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        fields = (fp.read(EDF_LPTI_BSIZE)).split()

        if len(fields) > int(0):
            self.h_d[EDF_LPTI_PATIENT_ID_NAME] = str(fields[0],
                                                     nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LPTI_PATIENT_ID_NAME] = nft.STRING_EMPTY
        if len(fields) > int(1):
            self.h_d[EDF_LPTI_GENDER_NAME] = str(fields[1],
                                                 nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LPTI_GENDER_NAME] = nft.STRING_EMPTY
        if len(fields) > int(2):
            self.h_d[EDF_LPTI_DOB_NAME] = str(fields[2],
                                              nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LPTI_DOB_NAME] = nft.STRING_EMPTY
        if len(fields) > int(3):
            self.h_d[EDF_LPTI_FULL_NAME_NAME] = str(fields[3],
                                                    nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LPTI_FULL_NAME_NAME] = nft.STRING_EMPTY
        if len(fields) > int(4):
            self.h_d[EDF_LPTI_AGE_NAME] = str(fields[4],
                                              nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LPTI_AGE_NAME] = nft.STRING_EMPTY
        
        # (3) local recording information
        #
        # unfortunately, some edf files don't contain all the information
        # they should. this often occurs because the deidenitification
        # process overwrites this information. so we zero out the buffers
        # that won't be filled if the information is missing.
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (3)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        fields = (fp.read(EDF_LRCI_BSIZE)).split()

        if len(fields) > int(0):
            self.h_d[EDF_LRCI_START_DATE_LABEL] = str(fields[0],
                                                      nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LRCI_START_DATE_LABEL] = nft.STRING_EMPTY
        if len(fields) > int(1):
            self.h_d[EDF_LRCI_START_DATE] = str(fields[1],
                                                nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LRCI_START_DATE] = nft.STRING_EMPTY
        if len(fields) > int(2):
            self.h_d[EDF_LRCI_EEG_ID] = str(fields[2],
                                            nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LRCI_EEG_ID] = nft.STRING_EMPTY
        if len(fields) > int(3):
            self.h_d[EDF_LRCI_TECH] = str(fields[3],
                                          nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LRCI_TECH] = nft.STRING_EMPTY
        if len(fields) > int(4):
            self.h_d[EDF_LRCI_MACHINE] = str(fields[4],
                                             nft.DEF_CHAR_ENCODING)
        else:
            self.h_d[EDF_LRCI_MACHINE] = nft.STRING_EMPTY

        # (4) general header information
        #
        # get the fourth block of data (non-local information)
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (4)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        try:
            byte_buf = fp.read(EDF_GHDI_BSIZE)
            buf = str(byte_buf, nft.DEF_CHAR_ENCODING)
        except:
            print("Error: %s (line: %s) %s::%s: char encoding (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   byte_buf))
            return False

        self.h_d[EDF_GHDI_START_DATE] = buf[0:8]
        self.h_d[EDF_GHDI_START_TIME] = buf[8:8+8]
        self.h_d[EDF_GHDI_HSIZE] = nft.atoi(buf[16:16+8])
        self.h_d[EDF_GHDI_FILE_TYPE] = buf[24:24+5]
        self.h_d[EDF_GHDI_RESERVED] = buf[29:29+39]
        self.h_d[EDF_GHDI_NUM_RECS] = nft.atoi(buf[68:68+8])
        self.h_d[EDF_GHDI_DUR_REC] = nft.atof(buf[76:76+8])
        self.h_d[EDF_GHDI_NSIG_REC] = nft.atoi(buf[84:84+4])

        # (5) channel-specific information
        #
        # get the fifth block of data (channel-specific information)
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (4)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        # (5a) read channel labels
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5a)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        buf = fp.read(EDF_LABL_BSIZE * self.h_d[EDF_GHDI_NSIG_REC])

        self.h_d[EDF_NUM_CHANNELS_ANNOTATION] = int(0)
        self.h_d[EDF_CHAN_LABELS] = []
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):

            # grab the channel label
            #
            offset = EDF_LABL_BSIZE * i
            tstr = (str(buf[offset:offset+EDF_LABL_BSIZE],
                       nft.DEF_CHAR_ENCODING)).upper()
            self.h_d[EDF_CHAN_LABELS].append(nft.trim_whitespace(tstr))
        
            # look for the annotation labels:
            #  note that the label is already upper case
            #
            if EDF_ANNOTATION_KEY in self.h_d[EDF_CHAN_LABELS][i]:
                self.h_d[EDF_NUM_CHANNELS_ANNOTATION] += int(1)

        # (5b) read the transducer type
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5b)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        buf = fp.read(EDF_TRNT_BSIZE * self.h_d[EDF_GHDI_NSIG_REC])

        self.h_d[EDF_CHAN_TRANS_TYPE] = []
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            offset = EDF_LABL_BSIZE * i
            tstr = str(buf[offset:offset+EDF_TRNT_BSIZE],
                       nft.DEF_CHAR_ENCODING)
            self.h_d[EDF_CHAN_TRANS_TYPE].append(nft.trim_whitespace(tstr))

        # (5c) read the physical dimension
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5c)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        buf = fp.read(EDF_PDIM_BSIZE * self.h_d[EDF_GHDI_NSIG_REC])

        self.h_d[EDF_CHAN_PHYS_DIM] = []
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            offset = EDF_PDIM_BSIZE * i
            tstr = str(buf[offset:offset+EDF_PDIM_BSIZE],
                       nft.DEF_CHAR_ENCODING)
            self.h_d[EDF_CHAN_PHYS_DIM].append(nft.trim_whitespace(tstr))

        # (5d) read the physical minimum
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5d)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        buf = fp.read(EDF_PMIN_BSIZE * self.h_d[EDF_GHDI_NSIG_REC])

        self.h_d[EDF_CHAN_PHYS_MIN] = []
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            offset = EDF_PMIN_BSIZE * i
            tstr = str(buf[offset:offset+EDF_PMIN_BSIZE],
                       nft.DEF_CHAR_ENCODING)
            self.h_d[EDF_CHAN_PHYS_MIN].\
                append(nft.atof(nft.trim_whitespace(tstr)))

        # (5e) read the physical maximum
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5e)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        buf = fp.read(EDF_PMAX_BSIZE * self.h_d[EDF_GHDI_NSIG_REC])

        self.h_d[EDF_CHAN_PHYS_MAX] = []
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            offset = EDF_PMAX_BSIZE * i
            tstr = str(buf[offset:offset+EDF_PMAX_BSIZE],
                       nft.DEF_CHAR_ENCODING)
            self.h_d[EDF_CHAN_PHYS_MAX].\
                append(nft.atof(nft.trim_whitespace(tstr)))

        # (5f) read the digital minimum
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5f)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        buf = fp.read(EDF_DMIN_BSIZE * self.h_d[EDF_GHDI_NSIG_REC])

        self.h_d[EDF_CHAN_DIG_MIN] = []
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            offset = EDF_DMIN_BSIZE * i
            tstr = str(buf[offset:offset+EDF_DMIN_BSIZE],
                       nft.DEF_CHAR_ENCODING)
            self.h_d[EDF_CHAN_DIG_MIN].\
                append(nft.atoi(nft.trim_whitespace(tstr)))

        # (5g) read the digital maximum
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5g)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        buf = fp.read(EDF_DMAX_BSIZE * self.h_d[EDF_GHDI_NSIG_REC])

        self.h_d[EDF_CHAN_DIG_MAX] = []
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            offset = EDF_DMAX_BSIZE * i
            tstr = str(buf[offset:offset+EDF_DMAX_BSIZE],
                       nft.DEF_CHAR_ENCODING)
            self.h_d[EDF_CHAN_DIG_MAX].\
                append(nft.atoi(nft.trim_whitespace(tstr)))

        # (5h) read the prefilt labels
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5h)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        buf = fp.read(EDF_PREF_BSIZE * self.h_d[EDF_GHDI_NSIG_REC])

        self.h_d[EDF_CHAN_PREFILT] = []
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            offset = EDF_PREF_BSIZE * i
            tstr = str(buf[offset:offset+EDF_PREF_BSIZE],
                       nft.DEF_CHAR_ENCODING)
            self.h_d[EDF_CHAN_PREFILT].append(nft.trim_whitespace(tstr))

        # (5i) read the rec sizes
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5i)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        buf = fp.read(EDF_RECS_BSIZE * self.h_d[EDF_GHDI_NSIG_REC])

        self.h_d[EDF_CHAN_REC_SIZE] = []
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            offset = EDF_RECS_BSIZE * i
            tstr = str(buf[offset:offset+EDF_RECS_BSIZE],
                       nft.DEF_CHAR_ENCODING)
            self.h_d[EDF_CHAN_REC_SIZE].\
                append(nft.atoi(nft.trim_whitespace(tstr)))

        # (5j) the last chunk of the header is reserved space
        # that we don't need to read. however, we need to advance the
        # file pointer to be safe.
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (5j)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        fp.seek(self.h_d[EDF_GHDI_HSIZE], os.SEEK_SET)

        # (6) compute some derived values
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: fetching (6)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        self.h_d[EDF_SAMPLE_FREQUENCY] = \
            (float(self.h_d[EDF_CHAN_REC_SIZE][0]) /
             float(self.h_d[EDF_GHDI_DUR_REC]))
        self.h_d[EDF_NUM_CHANNELS_SIGNAL] = \
            int(self.h_d[EDF_GHDI_NSIG_REC] -
                self.h_d[EDF_NUM_CHANNELS_ANNOTATION])

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Edf::get_sample_frequency
    #
    # arguments:
    #  chan: the input channel index
    #
    # returns: a floating point value containing the sample frequency
    #
    def get_sample_frequency(self, chan = EDF_DEF_CHAN):
        if chan == EDF_DEF_CHAN:
           return self.sample_frequency_d
        else:
            return (float(self.h_d[EDF_CHAN_REC_SIZE][chan]) /
                    float(self.h_d[EDF_GHDI_DUR_REC]))

    # method: Edf::get_num_samples
    #
    # arguments:
    #  chan: the input channel index
    #
    # returns: an integer value containing the number of samples
    #
    def get_num_samples(self, chan = EDF_DEF_CHAN):
        return int(self.h_d[EDF_CHAN_REC_SIZE][chan] *
                   self.h_d[EDF_GHDI_NUM_RECS])

    # method: Edf::get_duration
    #
    # arguments: none
    #
    # returns: a float containing the duration in secs
    #
    def get_duration(self):
        return (float(self.h_d[EDF_GHDI_DUR_REC] *
                      float(self.h_d[EDF_GHDI_NUM_RECS])))

    # method: Edf::read_edf
    #
    # arguments:
    #  fname: input filename
    #  scale: if true, scale the signal based on the header data
    #  sflag: if true, read the signal data
    #
    # return: the header and the signal data as dictionaries
    #
    # This method reads an edf file, and returns the raw signal data.
    #
    def read_edf(self, fname, scale, sflag = True):

        # delcare local variables
        #
        sig = OrderedDict()
        
        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: opening an EDF file (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))

        # open the file
        #
        fp = open(fname, nft.MODE_READ_BINARY)
        if fp is None:
            print("Error: %s (line: %s) %s::%s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))
            return (None, None)
  
        # get the size of the file on disk
        #
        fp.seek(0, os.SEEK_END)
        file_size_in_bytes = fp.tell()
        fp.seek(0, os.SEEK_SET)
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: file size = %ld bytes" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
	           file_size_in_bytes))

        # load the header
        #
        if self.get_header(fp) == False:
            print("Error: %s (line: %s) %s::%s: error in get_header (%s)" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__,
                   fname))
            return (None, None)

        # exit if necessary
        #
        if sflag == False:
            fp.close()
            return (self.h_d, None)

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            self.print_header(sys.stdout)

        # position the file to the beginning of the data
        # using the header information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: positioning file pointer" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))
        
        fp.seek(self.h_d[EDF_GHDI_HSIZE], os.SEEK_SET)
        
        # create space to hold the entire signal:
        #  in python, we only need to size the numpy arrays
        #
        for i in range(0, self.h_d[EDF_GHDI_NSIG_REC]):
            sz = int(self.h_d[EDF_GHDI_NUM_RECS] *
                     self.h_d[EDF_CHAN_REC_SIZE][i])
            sig[self.h_d[EDF_CHAN_LABELS][i]] = \
                np.empty(shape = sz, dtype = np.float64)
            
            if (self.dbgl_d == ndt.FULL) and (i < EDF_DEF_DBG_NF):
                print("%s (line: %s) %s::%s %s (%s: %ld row, %ld cols)" %
                      (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__,
                       ndt.__NAME__, "sig dimensions",
                       self.h_d[EDF_CHAN_LABELS][i], i,
                       sig[self.h_d[EDF_CHAN_LABELS][i]].shape[0]))

        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s signal vector resized" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        # loop over all records
        #
        ns_read = np.zeros(shape = self.h_d[EDF_GHDI_NSIG_REC], dtype = int)
        for i in range(0, self.h_d[EDF_GHDI_NUM_RECS]):

            # loop over all channels
            #
            for j in range(0, self.h_d[EDF_GHDI_NSIG_REC]):

                # display debug message
                #
                if (self.dbgl_d == ndt.FULL) and (i < EDF_DEF_DBG_NF) and \
	           (j < EDF_DEF_DBG_NF):
                    print("%s (line: %s) %s::%s: %s [%ld %ld]" %
                          (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__,
                           ndt.__NAME__,
                           "reading record no.", i, j))

                # read the data:
                #  store the data after the last sample read
                #
                num_samps = self.h_d[EDF_CHAN_REC_SIZE][j]
                data = fp.read(num_samps * EDF_SIZEOF_SHORT)
                buf = np.frombuffer(data, dtype = "short", count = num_samps) \
                      .astype(np.float64)
                ns_read[j] += num_samps

                if num_samps != int(len(data) / EDF_SIZEOF_SHORT):
                    print("Error: %s (line: %s) %s::%s: %s [%d][%d]" %
                          (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__,
                           ndt.__NAME__, "read error",
                           num_samps, int(len(data)/EDF_SIZEOF_SHORT)))
                    return (None, None)


                # compute scale factors:
                #  this code is identical to the C++ version
                sum_n = float(self.h_d[EDF_CHAN_PHYS_MAX][j] - \
                              self.h_d[EDF_CHAN_PHYS_MIN][j])
                sum_d = float(self.h_d[EDF_CHAN_DIG_MAX][j] -
                              self.h_d[EDF_CHAN_DIG_MIN][j])
                sum = float(1.0)
                dc = float(0.0)
                if sum_d != float(0.0):
                    sum = sum_n / sum_d
                    dc = float(self.h_d[EDF_CHAN_PHYS_MAX][j] - sum * \
                               float(self.h_d[EDF_CHAN_DIG_MAX][j]))

                if (self.dbgl_d == ndt.FULL) and (i < EDF_DEF_DBG_NF) and \
	           (j < EDF_DEF_DBG_NF):
                    print("%s (line: %s) %s::%s:"
                          "%s [%ld %ld] %f (%f, %f, %f)" %
                          (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__,
	                   ndt.__NAME__, "dc offset = ",
	                   i, j, dc, sum_n, sum_d, sum))

                # scale the data
                #
                if scale == True:
                    buf = buf * sum
                    buf = buf + dc

                offset = i * self.h_d[EDF_CHAN_REC_SIZE][j]
                sig[self.h_d[EDF_CHAN_LABELS][j]][offset:offset+num_samps] = \
                    buf

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s closing an EDF file" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        # close the file
        #
        fp.close()

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: done closing an EDF file" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        # exit gracefully
        #
        return (self.h_d, sig)
    #
    # end of method

    # method: Edf::cleanup
    #
    # arguments: none
    #
    # return: a boolean value indicating status
    #
    # This method cleans up memory.
    #
    def cleanup(self):

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: starting clean up of memory" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))

        # clear the header structure
        #
        if self.h_d != None:
            self.h_d = {}

        # display debug information
        #
        if self.dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: done cleaning up memory" %
                  (__FILE__, ndt.__LINE__, Edf.__CLASS_NAME__, ndt.__NAME__))
  
        # exit gracefully
        #
        return True

    # method: channels_data
    #
    # arguments:
    #  channels_order: the order of raw channels that should be read from EDF
    #  montages_order: if specified, instead of raw channels, the montages will
    #                  be returned.
    #
    # return:
    #  samp_rate: the sample rate of signals
    #  sigbufs: the signals (raw or montages) as numpy array
    #
    def channels_data(self, edf_file, channels_order, montages_order=None):
        """
        Channels_order is the channels' list that are needed to be extracted 
        from edf file.
        Then if montages_order is given, raw channels will be converted to 
        montages and instead of raw signals, montages will be returned.
        Returned values are sample frequency and signals in numpy array format.
        """

        # read the EDF file
        #
        header, signal = self.read_edf(edf_file, False)

        # find the appropriate sample rate
        #
        samp_rates = np.zeros((len(channels_order), ))
        num_samples = samp_rates.copy()
        for ch_counter, channel in enumerate(channels_order):
            
            # find the specified channel in all the channels in edf file.
            # if nothing found, something is wrong. stop the whole process to
            # find the reason.
            #
            found = False

            # loop through the return signal data where signal is an 
            # OrderedDictionary:
            # Ex: {'EEG FP1-LE' : array([-118., ... , -88. ]) 
            #
            # If channel exist within the EDF file, we get its sample frequency
            # along its sample number
            #
            for sig_counter, signal_label in enumerate(signal):
                if (channel in signal_label):
                    samp_rates[ch_counter] = self.get_sample_frequency(sig_counter)
                    num_samples[ch_counter] = self.get_num_samples(sig_counter)
                    found = True
                    break

            # if we reach this, it means that we couldn't find the channel
            # in the EDF file 
            #
            assert found, \
                (f"Error: {__FILE__} (line: {ndt.__LINE__}) \
                {ndt.__NAME__}: Channel {channel} wasn't found \
                in {edf_file}")

        samp_rate = int(samp_rates.max())
        nSamples = int(num_samples.max())
        
        # now sample rate and number of samples are found, just fill the matrix.
        #
        sigbufs = np.zeros((len(channels_order), nSamples))
        for ch_counter, channel in enumerate(channels_order):
            # Find the index of channel
            for sig_counter, signal_label in enumerate(signal):
                if (channel in signal_label):
                    step = int(nSamples / self.get_num_samples(ch_counter))
                    sigbufs[ch_counter, 0::step] = signal[channel]

        # if montage order is specified, it means that instead of raw samples,
        # the montages should be returned. So, the montages will be extracted.
        #
        if montages_order is not None:
            
            # split raw channels in each montages and find raw channels index
            # in channels_order
            #
            monsig = np.zeros((len(montages_order), sigbufs.shape[1]))
            for mcounter, montage in enumerate(montages_order):
                rawch1, rawch2 = montage.split(nft.DELIM_DASH)
                index1, index2 = -1, -1
                for ch_counter, channel in enumerate(channels_order):
                    if rawch1 in channel:
                        index1 = ch_counter
                    if rawch2 in channel:
                        index2 = ch_counter

                assert not(index1 == -1 or index2 == -1 or index1 == index2),\
                    (f"Error: {__FILE__} (line: {ndt.__LINE__}) \
                        {ndt.__NAME__}: channels({rawch1},{rawch2}) \
                        hasn't been found or the indices are incorrect")
                
                monsig[mcounter, :] = sigbufs[index1] - sigbufs[index2]

            sigbufs = monsig

        return samp_rate, sigbufs
    #
    # end of method

    # method: down_sample
    #
    # arguments:
    #  sigmat: the raw signals or montages signals matrix
    #  old_samp_rate: old sample rate
    #  new_samp_rate: new sample rate
    #  filter_order: filter order for low pass filter before decimation
    #
    # return:
    #  lpfmat: low pass filter and down sampled channels signals matrix
    #          rows of matrix are channels and columns are samples.
    #
    def down_sample(self, sigmat, old_samp_rate, new_samp_rate,
                    filter_order=16):
        decimation_factor = int(old_samp_rate / new_samp_rate)
        filter_order = int(filter_order)
        lpfmat = signal.decimate(x=sigmat, q=decimation_factor, n=filter_order,
                                 ftype='fir', axis=1, zero_phase=True)
        return lpfmat
    #
    # end of method

#
# end of Edf

#
# end of file
