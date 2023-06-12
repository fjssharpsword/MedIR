File: nedc_eas/v5.1.1/AAREADME.txt
Tool: The NEDC EEG Annotation System
Version: 5.1.1

-------------------------------------------------------------------------------
Change Log:
(20220505) clarifications for Windows users
(20220430) incorporates our new annotation libraries
(20210826) fixed some bugs in reading cvs files; changed the name to EAS
(20210529) updated to Python 3.7.x improved portability
-------------------------------------------------------------------------------

This directory contains all the Python code needed to run our annotation
tool. This tool has been used to annotate various NEDC EEG corpora. To
learn more about the origins of the tool, please see this publication:

 Capp, N., Krome, E., Obeid, I., & Picone, J. (2017). Facilitating the
 Annotation of Seizure Events Through an Extensible Visualization
 Tool. In I. Obeid, I. Selesnick, & J. Picone (Eds.), Proceedings of
 the IEEE Signal Processing in Medicine and Biology Symposium
 (p. 1). IEEE. https://doi.org/10.1109/SPMB.2017.8257043

Please cite this publication when referring to this tool.

A. WHAT'S NEW

Version 5.1.1 changes:
  + Update the README file to clarify the installation for Window User
  + Small cosmetic changes 

B. INSTALLATION REQUIREMENTS

Python code unfortunately often depends on a large number of add-ons, making
it very challenging to port code into new environments. This tool has been
tested extensively on Windows and Mac machines running Python v3.9.x.

Software tools required include:

 o Python 3.7.x or higher (we recommend installing Anaconda)
 o PyQt5: https://www.riverbankcomputing.com/software/pyqt/download5
 o Numpy/SciPy: http://www.numpy.org/
 o PyQtGraph: http://www.pyqtgraph.org/ (v0.12.1 or higher)
 o ReportLab: http://www.reportlab.com/
 o lxml: https://lxml.de
 o bs4: https://www.crummy.com/software/BeautifulSoup/

There is a requirements.txt included in the release that helps you automate
the process of updating your environment.

C. USER'S GUIDE

C.1. WINDOW USERS

For Window users, we recommend users install Anaconda in order to run
a bash emulator.

Through the Anaconda prompt, create a new environment and specify the proper
python version. We can install all dependencies with a single command:

 $ conda create -n <my_environment_name> python=3.9 pyqt pyqtgraph reportlab \
 lxml scipy bs4 m2-base -c conda-forge

Once the software has been installed, you need to do the following things if
you want to run this from any directory:

 - activate your conda environment

    $ conda activate <my_environment_name>

 - set three environment variables - NEDC_NFC, PATH, and PYTHONPATH

    $ setx NEDC_NFC "<nedc_eas_installation_path>"
    $ setx PATH "%PATH%;%NEDC_NFC%\bin"
    $ setx PYTHONPATH "%PYTHONPATH%;%NEDC_NFC%\lib"

 - restart your command prompt and activate your environment again. 
   Then launch a bash session:

    $ bash
 
You should be able to type:

 $ which nedc_eas

and see the command. Then you can simply type:

 $ nedc_eas

Note: Since this is on Windows which uses DOS path conventions, be careful
with the directory structure of where you place the application. Avoid spaces
and special characteries in parent directories.

C.2. LINUX/MAC USERS

For Mac users, since Mac OS X 10.8 comes with Python 2.7, you may 
need to utilize pip3 when attempting to install dependencies:

 $ pip3 install pyqt5
 $ pip3 install pyqtgraph
 $ pip3 install reportlab
 $ pip3 install lxml
 $ pip3 install scipy
 $ pip3 install bs4

The easiest way to run this is to change your current working directory
to the root directory of the installation and execute the tool as follows:
 $ cd <my_install_location>/nedc_eas/v5.1.1
 $ ./bin/nedc_eas
 
Once the software has been installed, you need to do the following things if
you want to run this from any directory:

 - set the environment variable NEDC_NFC to the root directory
   of the installation:
      $ export NEDC_NFC='<my_install_location>/nedc_eas/v5.1.1'

 - put $NEDC_NFC/bin in your path:
      $ export PATH=$PATH:$NEDC_NFC
 
You should be able to type:

 $ which nedc_eas

and see the command. Then you can simply type:

 $ nedc_eas

After loading the tool, click on open under file on the navigation bar and
load the edf file that you wish to annotate.

A short tutorial on how to use the tool is available here:

https://www.isip.piconepress.com/projects/tuh_eeg/downloads/nedc_eas/v5.0.5/videos/getting_started/getting_started_v00.mp4

-----------------------------

If you have any additional comments or questions about the data,
please direct them to help@nedcdata.org.

Best regards,

Joe Picone
