#!/usr/bin/env python
#
# file: $NEDC_NFC/class/python/nedc_sys_tools/nedc_debug_tools.py
#                                                                              
# revision history:
#
# 20200531 (JP): refactored code
# 20200514 (JP): initial version
#                                                                              
# This file contains classes that facilitate debugging and information display.
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys

# import NEDC modules
#

#------------------------------------------------------------------------------
#                                                                              
# global variables are listed here                                             
#                                                                              
#------------------------------------------------------------------------------

# define a numerically ordered list of 'levels'
#
NONE = int(0)
BRIEF = int(1)
SHORT = int(2)
MEDIUM = int(3)
DETAILED = int(4)
FULL = int(5)

# define a dictionary indexed by name and reverse it so we have it by level
#
LEVELS = {'NONE': NONE, 'BRIEF': BRIEF, 'SHORT': SHORT,
          'MEDIUM': MEDIUM, 'DETAILED': DETAILED, 'FULL': FULL}
NAMES = {val: key for key, val in LEVELS.items()}

# define a constant that controls the amount of precision used
# to check floating point numbers. we use two constants - max precision
# for detailed checks and min precision for curosry checks.
#
MIN_PRECISION = int(4)
MAX_PRECISION = int(10)

# define a constant that is used to seed random number generators
# from a common starting point
#
RANDSEED = int(27)

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: __NAME__
#
# This class is used to get the function name. This is analogous to
# __NAME__ in C++. This class is hidden from the user.
#                                                                             
class __NAME__(object):

    # method: default constructor
    #
    #    def __init__(self):
    #        pass
    
    # method: a built-in function that returns an object representation
    #
    def __repr__(self):
        try:
            raise Exception
        except:
            return str(sys.exc_info()[2].tb_frame.f_back.f_code.co_name)

#
# end of class

# class: __LINE__
#
# This class is used to get the line number. This is analogous to
# __LINE__ in C++. This clas is hidden from the user.
#                                                                             
class __LINE__(object):

    # method: a built-in function that returns an object representation
    #
    def __repr__(self):
        try:
            raise Exception
        except:
            return str(sys.exc_info()[2].tb_frame.f_back.f_lineno)

#
# end of class

# define an abbreviations for the above classes:
#  These have to come after the class definitions, and are the symbols
#  that programmers will use.
#
# define an abbreviation for the above class:
#  __FILE__ must unfortunately be put in each file.
#
__NAME__ = __NAME__()
__LINE__ = __LINE__()
__FILE__ = os.path.basename(__file__)

# class: Dbgl
#
# This class is a parallel implementation of our C++ class Dbgl. Please see
# $NEDC_NFC/class/cpp/Dbgl/Dbgl.h for more information about this class. The
# definitions here need to be exactly the same as those in that class.
#
# Note that we prefer to manipulate this class using integer values
# rather than strings. Strings are only really used for the command line
# interface. All other operations should be done on integers.
#
class Dbgl:

    #--------------------------------------------------------------------------
    #
    # static data declarations
    #
    #--------------------------------------------------------------------------

    # define a static variable to hold the value
    #
    level_d = NONE

    #--------------------------------------------------------------------------
    #
    # constructors
    #
    #--------------------------------------------------------------------------

    # method: Dbgl::constructor
    #
    # note this method cannot set the value or this overrides
    # values set elsewhere in a program. the set method must be called.
    # 
    def __init__(self):
        Dbgl.__CLASS_NAME__ = self.__class__.__name__
    
    #--------------------------------------------------------------------------
    #
    # operator overloads:
    #  we keep the definitions concise
    #
    #--------------------------------------------------------------------------

    # method: Dbgl::int()
    #  cast conversion to int
    #
    def __int__(self):
        return int(self.level_d)

    # method:: Dbgl::>
    #  overload > (greater than) operator
    #
    def __gt__(self, level):
        if Dbgl.level_d > level:
            return True
        return False

    # method: Dbgl::>=
    #  overload >= (greater than or equal to) operator
    #
    def __ge__(self, level):
        if Dbgl.level_d >= level:
            return True
        return False

    # method: Dbgl::!=
    #  overload != (not equal to) operator
    #
    def __ne__(self, level):
        if Dbgl.level_d != level:
            return True
        return False

    # method: Dbgl::<
    #  overload < (less than) operator
    #
    def __lt__(self, level):
        if Dbgl.level_d < level:
            return True
        return False

    # method: Dbgl::<=
    #  overload <= (less than or equal to) operator
    #
    def __le__(self, level):
        if Dbgl.level_d <= level:
            return True
        return False

    # method:: Dbgl::==
    #  overload == (equal to) operator
    #
    def __eq__(self, level):
        if Dbgl.level_d == level:
            return True
        return False

    #--------------------------------------------------------------------------
    #
    # set and get methods
    #
    #--------------------------------------------------------------------------

    # method: Dbgl::set
    # 
    def set(self, level = None, name = None):

        # check and set the level by value
        #
        if level is not None:
            if self.check(level) == False:
                print("Error: %s (line: %s) %s::%s: invalid value (%d)" %
                      (__FILE__, __LINE__, Dbgl.__CLASS_NAME__, __NAME__,
                       level))
                sys.exit(os.EX_SOFTWARE)
            else:
                Dbgl.level_d = int(level)

        # check and set the level by name
        #
        elif name is not None:
            try:
                Dbgl.level_d = LEVELS[name.upper()]
            except KeyError as e:
                print("Error: %s (line: %s) %s::%s: invalid value (%s)" %
                      (__FILE__, __LINE__, Dbgl.__CLASS_NAME__, __NAME__,
                       name))
                sys.exit(os.EX_SOFTWARE)

        # if neither is specified, set to NONE
        #
        else:
            Dbgl.level_d = NONE

        # exit gracefully
        #
        return Dbgl.level_d

    # method: Dbgl::get
    #  note that we don't provide a method to return the integer value
    #  because int(), a pseudo-cast operator, can do this.
    # 
    def get(self):
        return NAMES[Dbgl.level_d]

    # method: Dbgl::check
    # 
    def check(self, level):
        if (level < NONE) or (level > FULL):
            return False;
        else:
            return True

#
# end of class

# class: Vrbl
#
# This class is a parallel implementation of our C++ class Vrbl. Please see
# $NEDC_NFC/class/cpp/Vrbl/Vrbl.h for more information about this class. The
# definitions here need to be exactly the same as those in that class.
#
class Vrbl:

    #--------------------------------------------------------------------------
    #
    # static data declarations
    #
    #--------------------------------------------------------------------------

    # define a static variable to hold the value
    #
    level_d = NONE

    #--------------------------------------------------------------------------
    #
    # constructors
    #
    #--------------------------------------------------------------------------

    # method: Vrbl::constructor
    # 
    # note this method cannot set the value or this overrides
    # values set elsewhere in a program. the set method must be called.
    # 
    def __init__(self):
        Vrbl.__CLASS_NAME__ = self.__class__.__name__
    
    #--------------------------------------------------------------------------
    #
    # operator overloads
    #
    #--------------------------------------------------------------------------

    # method: Vrbl::int()
    #  cast conversion to int
    #
    def __int__(self):
        return int(self.level_d)

    # method: Vrbl::>
    #  overload > (greater than) operator
    #
    def __gt__(self, level):
        if Vrbl.level_d > level:
            return True
        return False

    # method: Vrbl::>=
    #  overload >= (greater than or equal to) operator
    #
    def __ge__(self, level):
        if Vrbl.level_d >= level:
            return True
        return False

    # method: Vrbl::!=
    #  overload != (not equal to) operator
    #
    def __ne__(self, level):
        if Vrbl.level_d != level:
            return True
        return False

    # method: Vrbl::<
    #  overload < (less than) operator
    #
    def __lt__(self, level):
        if Vrbl.level_d < level:
            return True
        return False

    # method: Vrbl::<=
    #  overload <= (less than or equal to) operator
    #
    def __le__(self, level):
        if Vrbl.level_d <= level:
            return True
        return False

    # method: Vrbl::==
    #  overload == (equal to) operator
    #
    def __eq__(self, level):
        if Vrbl.level_d == level:
            return True
        return False

    #--------------------------------------------------------------------------
    #
    # set and get methods
    #
    #--------------------------------------------------------------------------

    # method: Vrbl::set
    # 
    def set(self, level = None, name = None):

        # check and set the level by value
        #
        if level is not None:
            if self.check(level) == False:
                print("Error: %s (line: %s) %s::%s: invalid value (%d)" %
                      (__FILE__, __LINE__, Vrbl.__CLASS_NAME__, __NAME__,
                       level))
                sys.exit(os.EX_SOFTWARE)
            else:
                Vrbl.level_d = int(level)

        # check and set the level by name
        #
        elif name is not None:
            try:
                Vrbl.level_d = LEVELS[name.upper()]
            except KeyError as e:
                print("Error: %s (line: %s) %s::%s: invalid value (%s)" %
                      (__FILE__, __LINE__, Vrbl.__CLASS_NAME__, __NAME__,
                       name))
                sys.exit(os.EX_SOFTWARE)

        # if neither is specified, set to NONE
        #
        else:
            Vrbl.level_d = NONE

        # exit gracefully
        #
        return Vrbl.level_d

    # method: Vrbl::get
    #  note that we don't provide a method to return the integer value
    #  because int(), a pseudo-cast operator, can do this.
    # 
    def get(self):
        return NAMES[Vrbl.level_d]

    # method: Vrbl::check
    # 
    def check(self, level):
        if (level < NONE) or (level > FULL):
            return False;
        else:
            return True

#
# end of class

#                                                                              
# end of file 

