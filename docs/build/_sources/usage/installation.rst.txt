======================
How to install peakipy
======================

Here are instructions for installing peakipy.

Installation
------------

I recommend using a virtual environment

``python3 -m venv peakipy_venv``

Then activate

``source peakipy_venv/bin/activate`` if using bash

or 

``source peakipy_venv/bin/activate.csh`` if using c-shell

Once activated 

``pip install peakipy``

You can also clone the `peakipy <https://github.com/j-brady/peakipy>`_ repository from github.

With poetry
^^^^^^^^^^^

``cd peakipy; poetry install``

If you don't have poetry you can install it with the following command

``curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python``

Otherwise refer to the `poetry documentation <https://poetry.eustace.io/docs/>`_ for more details.

With setup.py
^^^^^^^^^^^^^

``cd peakipy; python setup.py install``


At this point the package should be installed and the main scripts (``read_peaklist.py``, ``edit_fits.py``, ``fit_peaks.py`` and ``check_fits.py``) should have been added to your path.


Requirements
------------

* Python3.6 or above 
* pandas>=0.24.0
* numpy>=1.16
* matplotlib>=3.0
* PyYAML>=3.13
* nmrglue>=0.6.0
* scipy>=1.2
* docopt>=0.6.2
* lmfit>=0.9.12
* scikit-image>=0.14.2
* bokeh>=1.0.4
