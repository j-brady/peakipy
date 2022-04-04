======================
How to install peakipy
======================

Here are instructions for installing peakipy.

Installation
------------

I recommend using a virtual environment::
        
        python3 -m venv peakipy_venv


Then activate (bash)::
        
        source peakipy_venv/bin/activate

or (csh)::
        
        source peakipy_venv/bin/activate.csh
        

Once activated you can install peakipy in any of the following ways. 

With pip
^^^^^^^^
::

        pip install peakipy


Below is an example of an installation script and a basic use case ::

        #!/bin/bash

        ##############################
        # make a virtual environment #
        ##############################
        python3.7 -m venv peakipy_env;
        source peakipy_env/bin/activate;

        ##############################
        # install peakipy            #
        ##############################
        pip install --upgrade pip;
        pip install peakipy;

        ##############################
        #  process some data!        #
        ##############################
        peakipy read peaks.a2 test.ft2 --a2 --f1radius=0.213 --show;
        peakipy edit peaks.csv test.ft2; # adjust fitting parameters
        peakipy fit peaks.csv test.ft2 fits.csv --vclist=vclist; # assuming you saved edited peaklist as peaks.csv
        # interactive checking
        peakipy check fits.csv test.ft2 --clusters=86,96,104 --colors=purple,green --show --outname=~tmp.pdf;
        # plots all the fits (first plane only)
        peakipy check fits.csv test.ft2 --first --colors=purple,green;



Run this above code by sourcing the file e.g. ``source file_containing_commands``


With poetry
^^^^^^^^^^^

Clone the `peakipy <https://github.com/j-brady/peakipy>`_ repository from github::

        git clone https://github.com/j-brady/peakipy.git
        cd peakipy; poetry install


If you don't have poetry you can install it with the following command::

        curl -sSL https://install.python-poetry.org | python3 -


Otherwise refer to the `poetry documentation <https://poetry.eustace.io/docs/>`_ for more details.


With setup.py
^^^^^^^^^^^^^

After cloning the `peakipy <https://github.com/j-brady/peakipy>`_ repository from github.::
        
        cd peakipy; python setup.py install


At this point the package should be installed and the main scripts (``peakipy read``, ``peakipy edit``, ``peakipy fit`` and ``peakipy check``) should have been added to your path.


Requirements
------------

* Python3.6 or above 
* pandas>=0.24.0
* numpy>=1.16
* matplotlib>=3.0
* PyYAML>=5.1
* nmrglue>=0.6.0
* scipy>=1.2
* docopt>=0.6.2
* lmfit>=0.9.12
* scikit-image>=0.14.2
* bokeh>=1.0.4
* schema>=0.7.0
