#!/usr/bin/env python
#
# file: $NEDC_NFC/class/python/nedc_sys_tools/nedc_file_tools.py
#                                                                              
# revision history:
#  20220225 (PM): added extract_comments function
#  20200623 (JP): reorganized
#  20200609 (JP): refactored the code and added atof and atoi
#  20170716 (JP): Upgraded to using the new annotation tools.
#  20170709 (JP): generalized some functions and abstracted more file I/O
#  20170706 (NC): refactored eval_tools into file_tools and display_tools
#  20170611 (JP): updated error handling
#  20170521 (JP): initial version
#                                                                              
# usage:                                                                       
#  import nedc_file_tools as nft
#                                                                              
# This class contains a collection of functions that deal with file handling
#------------------------------------------------------------------------------
#                                                                             
# imports are listed here                                                     
#                                                                             
#------------------------------------------------------------------------------

# import system modules
#
import errno
import os
import re
import sys

# import NEDC modules
#
import nedc_debug_tools as ndt

#------------------------------------------------------------------------------
#                                                                              
# global variables are listed here                                             
#                                                                              
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# set the default character encoding system
#
DEF_CHAR_ENCODING = "utf-8"

# file processing charater constants
#
DELIM_BLANK = '\x00'
DELIM_BOPEN = '{'
DELIM_BCLOSE = '}'
DELIM_CARRIAGE = '\r'
DELIM_CLOSE = ']'
DELIM_COLON = ':'
DELIM_COMMA = ','
DELIM_COMMENT = '#'
DELIM_DASH = '-'
DELIM_DOT = '.'
DELIM_EQUAL = '='
DELIM_GREATTHAN = '>'
DELIM_LESSTHAN = '<'
DELIM_NEWLINE = '\n'
DELIM_NULL = ''
DELIM_OPEN = '['
DELIM_QUOTE = '"'
DELIM_SEMI = ';'
DELIM_SLASH = '/'
DELIM_SPACE = ' '
DELIM_SQUOTE = '\''
DELIM_TAB = '\t'
DELIM_USCORE = '_'

# define default file extensions
#
DEF_EXT_CSV = "csv"
DEF_EXT_EDF = "edf"
DEF_EXT_LBL = "lbl"
DEF_EXT_REC = "rec"
DEF_EXT_SVS = "svs"
DEF_EXT_TXT = "txt"
DEF_EXT_XML = "xml"

# regular expression constants
#
DEF_REGEX_ASSIGN_COMMENT = '^%s([a-zA-Z:!?" _-]*)%s(.+?(?=\n))'

# file processing string constants
#
STRING_EMPTY = ""
STRING_DASHDASH = "--"

# file processing lists:
#  used to accelerate some functions
#
LIST_SPECIALS = [DELIM_SPACE, DELIM_BLANK]

# i/o constants
#
MODE_READ_TEXT = "r"
MODE_READ_BINARY = "rb"
MODE_WRITE_TEXT = "w"
MODE_WRITE_BINARY = "wb"

# parameter file constants
#
DELIM_VERSION = "version"
PFILE_VERSION = "param_v1.0.0"

# define constants for XML tags
#
DEF_XML_HEIGHT = "height"
DEF_XML_WIDTH = "width"
DEF_XML_CONFIDENCE = "confidence"
DEF_XML_COORDS = "coordinates"
DEF_XML_REGION_ID = "region_id"
DEF_XML_TEXT = "text"
DEF_XML_TISSUE_TYPE = "tissue_type"
DEF_XML_LABEL = "label"

# define constants for CSV tags
#

# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()

#------------------------------------------------------------------------------
#
# functions listed here: general string processing
#
#------------------------------------------------------------------------------

# function: trim_whitespace
#                                                                          
# arguments:
#  istr: input string
#
# return: an output string that has been trimmed
#
# This function removes leading and trailing whitespace.
# It is needed because text fields in Edf files have all
# sorts of junk in them.
#
def trim_whitespace(istr):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: trimming (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, istr))

    # declare local variables
    #
    last_index = len(istr)

    # find the first non-whitespace character
    #
    flag = False
    for i in range(last_index):
        if not istr[i].isspace():
            flag = True
            break

    # make sure the string is not all whitespace
    #
    if flag == False:
        return STRING_EMPTY
        
    # find the last non-whitespace character
    #
    for j in range(last_index - 1, -1, -1):
        if not istr[j].isspace():
            break

    # exit gracefully: return the trimmed string
    #
    return istr[i:j+1]
#                                                                          
# end of function

# function: first_substring
#
# arguments:
#  strings: list of strings (input)
#  substring: the substring to be matched (input)
#
# return: the index of the match in strings
#
# This function finds the index of the first string in strings that
# contains the substring. This is similar to running strstr on each
# element of the input list.
#
def first_substring(strings, substring):
    try:
        return next(i for i, string in enumerate(strings) if \
                    substring in string)
    except:
        return int(-1)
#
# end of function

# function: first_string
#
# arguments:
#  strings: list of strings (input)
#  substring: the string to be matched (input)
#
# return: the index of the match in strings
#
# This function finds the index of the first string in strings that
# contains an exact match. This is similar to running strstr on each
# element of the input list.
#
def first_string(strings, tstring):
    try:
        return next(i for i, string in enumerate(strings) if \
                    tstring == string)
    except:
        return int(-1)
#
# end of function

# function: atoi
#                                                                          
# arguments:
#  value: the value to be converted as a string
#                                              
# return: an integer value
#
# This function emulates what C++ atoi does by replacing
# null characters with spaces before conversion. This allows
# Python's integer conversion function to work properly.
#
def atoi(value):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: converting value (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, value))

    # replace all the null's with spaces:
    #  this code is complicated but can be found here:
    #   https://stackoverflow.com/a/30020228
    #
    ind = (min(map(lambda x: (value.index(x)
                              if (x in value) else len(value)),
                   LIST_SPECIALS)))
    tstr = value[0:ind]

    # try to convert the input
    #
    try:
        ival = int(tstr)
    except:
        print("Error: %s (line: %s) %s: string conversion error [%s][%s])" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, value, tstr))
        return None

    # exit gracefully
    #
    return ival
#
# end of function

# function: atof
#                                                                          
# arguments:
#  value: the value to be converted as a string
#                                              
# return: an integer value
#
# This function emulates what C++ atof does by replacing
# null characters with spaces before conversion. This allows
# Python's integer conversion function to work properly.
#
def atof(value):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: converting value (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, value))

    # replace all the null's with spaces:
    #  this code is complicated but can be found here:
    #   https://stackoverflow.com/a/30020228
    #
    ind = (min(map(lambda x: (value.index(x)
                              if (x in value) else len(value)),
                   LIST_SPECIALS)))
    tstr = value[0:ind]
    
    # try to convert the input
    #
    try:
        fval = float(tstr)
    except:
        print("Error: %s (line: %s) %s: string conversion error [%s][%s])" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, value, tstr))
        return None

    # exit gracefully
    #
    return fval
#
# end of function

#------------------------------------------------------------------------------
#
# functions listed here: manipulate filenames, lists and command line args
#
#------------------------------------------------------------------------------

# function: get_fullpath
#
# arguments:
#  path: path to directory or file
#
# return: the full path to directory/file path argument
#
# This function returns the full pathname for a file. It expands
# environment variables.
#
def get_fullpath(path):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: expanding name (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, path))

    # exit gracefully
    #
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
#
# end of function

# function: create_filename
#
# arguments:
#  iname: input filename (string)
#  odir: output directory (string)
#  oext: output file extension (string)
#  rdir: replace directory (string)
#  cdir: create directory (boolean - true means create the directory)
#
# return: the output filename
#
# This function creates an output file name based on the input arguments. It
# is a Python version of Edf::create_filename().
#
def create_filename(iname, odir, oext, rdir, cdir = False):
        
    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: creating (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, iname))

    # get absolute file name
    #
    abs_name = os.path.abspath(os.path.realpath(os.path.expanduser(iname)))

    # replace extension with ext
    #
    if oext is None:
        ofile = os.path.join(os.path.dirname(abs_name),
                             os.path.basename(abs_name))
    else:
        ofile = os.path.join(os.path.dirname(abs_name),
                             os.path.basename(abs_name).split(DELIM_DOT)[0]
                             + DELIM_DOT + oext)

    # get absolute path of odir
    #
    if odir is None:
        odir = DELIM_DOT
    else:
        odir = os.path.abspath(os.path.realpath(os.path.expanduser(odir)))

    # if the replace directory is valid and specified
    #
    if rdir is not None and rdir in ofile:

        # get absolute path of rdir
        #
        rdir = os.path.abspath(os.path.realpath(
            os.path.expanduser(rdir)))

        # replace the replace directory portion of path with 
        # the output directory
        #
        ofile = ofile.replace(rdir, odir)

    # if the replace directory is not valid or specified
    #
    else:

        # append basename of ofile to output directory
        #
        ofile = os.path.join(odir, os.path.basename(ofile))

    # create the directory if necessary
    #
    if cdir is True:
       if make_dir(odir) is False:
           print("Error: %s (line: %s) %s: make dir failed (%s)" %
                 (__FILE__, ndt.__LINE__, ndt.__NAME__, odir))
           sys.exit(os.EX_SOFTWARE)

    # exit gracefully
    #
    return ofile
#
# end of function

# function: concat_names
#                                                                          
# arguments:
#
#  odir: the output directory that will hold the file
#  fname: the output filename
#                                              
# return:
#  fname: a filename that is a concatenation of odir and fname
#
def concat_names(odir, fname):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: concatenating (%s %s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, odir, fname))

    # strip any trailing slashes
    #
    str = odir
    if str[-1] == DELIM_SLASH:
        str = str[:-1]

    # ceate the full pathname
    #
    new_name = str + DELIM_SLASH + fname

    # exit gracefully                                                     
    #                                                                      
    return new_name
#                                                                          
# end of function

# function: get_flist
#                                                                          
# arguments:
#  fname: full pathname of a filelist file
#                                              
# return:
#  flist: a list of filenames
#
# This function opens a file and reads filenames. It ignores comment
# lines and blank lines.
#
def get_flist(fname):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: opening (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

    # declare local variables
    #
    flist = []

    # open the file
    #
    try: 
        fp = open(fname, MODE_READ_TEXT) 
    except IOError: 
        print("Error: %s (line: %s) %s: file not found (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
        return None

    # iterate over lines
    #
    try:
        for line in fp:

            # remove spaces and newline chars
            #
            line = line.replace(DELIM_SPACE, DELIM_NULL) \
                       .replace(DELIM_NEWLINE, DELIM_NULL) \
                       .replace(DELIM_TAB, DELIM_NULL)

            # check if the line starts with comments
            #
            if line.startswith(DELIM_COMMENT) or len(line) == 0:
                pass
            else:
                flist.append(line)
    except:
        flist = None

    # close the file
    #
    fp.close()

    # exit gracefully
    #
    return flist
#                                                                          
# end of function

# function: make_fp
#                                                                          
# arguments:
#
#  fname: the filename
#                                              
# return:
#  fp: a file pointer
#
def make_fp(fname):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: creating (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

    # open the file
    #
    try:
        fp = open(fname, MODE_WRITE_TEXT)
    except:
        print("Error: %s (line: %s) %s: error opening file (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
        return None
 
    # exit gracefully                                                     
    #                                                                      
    return fp
#
# end of function

#------------------------------------------------------------------------------
#
# functions listed here: manipulate directories 
#
#------------------------------------------------------------------------------

# function: make_dirs
#
# arguments:
#  dirlist - the list of directories to create
#
# return: None
#
# This function creates all the directories in a given list
#
def make_dirs(dirlist):

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: creating (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, dirlist))

    # loop over the list
    #
    for directory in dirlist:

        # make the directory
        #
        make_dir(directory)

    # exit gracefully
    #
    return True
#
# end of function

# function: make_dir
#
# arguments:
#  path: new directory path (input)
#
# return: a boolean value indicating status
#
# This function emulates the Unix command "mkdir -p". It creates
# a directory tree, recursing through each level automatically.
# If the directory already exists, it continues past that level.
#
def make_dir(path):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: creating (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, path))

    # use a system call to make a directory
    #
    try:
        os.makedirs(path)

    # if the directory exists, and error is thrown (and caught)
    #
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

    # exit gracefully
    #
    return True
#
# end of function

# function: get_dirs
#
# arguments:
#  flist: list of files
#  odir: output directory
#  rdir: replace directory
#  oext: output extension
#
# return: set of unique directory paths
#
# This function returns a set containing unique directory paths
# from a given file list. This is done by replacing the rdir
# with odir and adding the base directory of the fname to the set
#
def get_dirs(flist, odir=DELIM_NULL, rdir=DELIM_NULL, oext=None):

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: fetching (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, flist))

    # generate a set of unique directory paths
    #
    unique_dirs = set()

    # for each file name in the list
    #
    for fname in flist:

        # generate the output file name
        #
        ofile = create_filename(fname, odir, oext, rdir)

        # append the base dir of the ofile to the set
        #
        unique_dirs.add(os.path.dirname(ofile))

    # exit gracefully
    #
    return unique_dirs
#
# end of function

#------------------------------------------------------------------------------
#
# functions listed here: manage parameter files
#
#------------------------------------------------------------------------------

# function: load_parameters
#                                                                          
# arguments:
#  pfile: path of a paramter file
#  keyword: section of the parameter file to load
#                                              
# return: a dict, containing the values in the section
#
def load_parameters(pfile, keyword):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: loading (%s %s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, pfile, keyword))

    # declare local variables
    #
    values = {}

    # make sure the file is a parameter file
    #
    if get_version(pfile) != PFILE_VERSION:
        return None

    # open the file
    #
    try: 
        fp = open(pfile, MODE_READ_TEXT) 
    except ioerror: 
        print("Error: %s (line: %s) %s: file not found (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, pfile))
        return None

    # loop over all lines in the file
    #
    flag_pblock = False
    for line in fp:

        # initialize empty value for each line
        value = ""
        
        # remove white spaces at the edges of the string
        #
        if DELIM_EQUAL in line:
            value = line.split(DELIM_EQUAL)[1]
            value = value.strip()
            
        # remove white spaces unless string starts with quotes
        #
        if (value.startswith(DELIM_QUOTE) and \
            value.endswith(DELIM_QUOTE) ):
            str = line.replace(DELIM_QUOTE, DELIM_NULL).strip()

        elif (value.startswith(DELIM_SQUOTE) and \
              value.endswith(DELIM_SQUOTE)):
            str = line.replace(DELIM_SQUOTE, DELIM_NULL).strip()

        else:
            str = line.replace(DELIM_SPACE, DELIM_NULL) \
                      .replace(DELIM_NEWLINE, DELIM_NULL) \
                      .replace(DELIM_TAB, DELIM_NULL)

        # throw away commented and blank lines
        #
        if ((str.startswith(DELIM_COMMENT) == True) or (len(str) == 0)):
            pass

        elif ((str.startswith(keyword) == True) and (DELIM_BOPEN in str)):
            flag_pblock = True

        elif ((flag_pblock == True) and (DELIM_BCLOSE in str)):
            fp.close();
            return values

        elif (flag_pblock == True):
            parts = str.split(DELIM_EQUAL)
            values[parts[0].strip()] = parts[1].strip()

    # make sure we found a block
    #
    if flag_pblock == False:
        fp.close()
        print("Error: %s (line: %s) %s: invalid parameter file (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, pfile))
        return None

    # exit gracefully
    #
    return values
#                                                                          
# end of function

# function: generate_map
#                                                                          
# arguments:
#  pblock: a dictionary containing a parameter block
#                                              
# return:
#  pmap: a parameter file map
#
# This function converts a dictionary returned from load_parameters to
# a dictionary containing a parameter map. Note that is lowercases the
# map so that text is normalized.
#
def generate_map(pblock):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: generating a map" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # declare local variables
    #
    pmap = {}

    # loop over the input, split the line and assign it to pmap
    #
    for key in pblock:
        lkey = key.lower()
        pmap[lkey] = pblock[key].split(DELIM_COMMA)
        pmap[lkey] = list(map(lambda x: x.lower(), pmap[lkey]))

    # exit gracefully
    #
    return pmap
#
# end of function

# function: permute_map
#                                                                          
# arguments:
#  map: the input map
#                                              
# return:
#  pmap: an inverted map
#
# this function permutes a map so symbol lookups can go fast.
#
def permute_map(map):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: permuting map" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # declare local variables
    #
    pmap = {}

    # loop over the input map:
    #  note there is some redundancy here, but every event should
    #  have only one output symbol
    #
    for sym in map:
        for event in map[sym]:
            pmap[event] = sym
 
    # exit gracefully                                                     
    #                                                                      
    return pmap
#
# end of function

# function: map_events
#                                                                          
# arguments:
#  elist: a list of events
#  pmap: a permuted map (look up symbols to be converted)
#                                              
# return:
#  mlist: a list of mapped events
#
# this function maps event labels to mapped values.
#
def map_events(elist, pmap):

    # display informational message
    #
    if dbgl == ndt.FULL:
        print("%s (line: %s) %s: mapping events" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # loop over the input list
    #
    mlist = []
    i = int(0)
    for event in elist:

        # copy the event
        #
        mlist.append([event[0], event[1], {}]);

        # change the label
        #
        for key in event[2]:
            mlist[i][2][pmap[key]] = event[2][key]

        # increment the counter
        #
        i += int(1)
 
    # exit gracefully                                                     
    #                                                                      
    return mlist
#
# end of function

# function: get_version
#
# arguments:
#  fname: input filename
#
# return: a string containing the type
#
# this function opens a file, reads the magic sequence and returns
# the string.
#
def get_version(fname):

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: opening file (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

    # open the file
    #
    try: 
        fp = open(fname, MODE_READ_TEXT) 
    except IOError: 
        print("%s (line: %s) %s: file not found (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
        return None

    # define version value
    #
    ver = None
    
    # iterate over lines until we find the magic string
    #
    for line in fp:

        # set every character to be lowercase
        #
        line = line.lower()

        # check if string contains "version"
        #
        if line.startswith("version") or line.startswith("<?xml version=") \
            or line.startswith("# version"):
            
            # only get version value after "version"
            #  for example, xxx_v1.0.0
            #
            ver = line.split(DELIM_VERSION, 1)[-1]
            ver = (ver.replace(DELIM_EQUAL, DELIM_NULL)).strip()
            ver = (ver.split())[0]

            #  remove "" if has
            #
            ver = ver.replace(DELIM_QUOTE, DELIM_NULL)
            
            # break after we find the version
            #
            break

    # close the file
    #
    fp.close()
    
    # substring "version" not found
    #
    if (ver is None):
        return None

    # exit gracefully
    #
    return ver
#                                                                          
# end of function

#------------------------------------------------------------------------------
#
# functions listed here: manage and manipulate data files
#
#------------------------------------------------------------------------------

# function: extract_comments
#
# arguments:
#   fname : the filename
#   cdelim: the character to check for the beginning of the comment
#   cassign: the character used the assignment operator in name/value pairs
#
# return: a dict
#
# this function extract a key-value comments from a file and returns a 
# dictionary
#
#   dict_comments = { "header" : "value" }
#   
#   note: everything is a string literal
#
def extract_comments(fname, cdelim = "#", cassign = "="):

    # create the regular expression pattern 
    #
    regex_assign_comment = re.compile(DEF_REGEX_ASSIGN_COMMENT %
                                      (cdelim, cassign),
                                      re.IGNORECASE | re.MULTILINE)
    # regex_regular_comment = re.compile(DEF_REGEX_REGULAR_COMMENT %
                                    #    (cdelim), re.IGNORECASE)

    # local dictionaries
    #
    dict_comments = {}

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: opening file (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

    # open the file
    #
    try: 
        fp = open(fname, MODE_READ_TEXT) 
    except IOError: 
        print("%s (line: %s) %s: file not found (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
        return None

    # loop through the file
    #
    for line in fp:

        # strip all the spaces within the line
        #
        line = line.replace(DELIM_CARRIAGE, DELIM_NULL)

        # skip all the line that is not a comment
        #
        if not line.startswith(cdelim):
            continue
        
        # extract all of the comments
        #
        assign_comment = re.findall(regex_assign_comment, line)

        # append it to the dictionary
        #
        if assign_comment:
            dict_comments[assign_comment[0][0].strip()] \
                            = assign_comment[0][1].strip()
        
    # close the file
    #
    fp.close()

    # exit gracefully
    #
    return dict_comments

#                                                                              
# end of file 

