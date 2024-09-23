Quickstart instructions
=======================

Inputs
------

1.  Peak list (see [instructions](./instructions.md))
2.  NMRPipe frequency domain dataset (2D or Pseudo 3D)

There are four main commands.

1.  `peakipy read` converts your peak list into a `.csv` file and
    selects clusters of peaks.
2.  `peakipy edit` is used to check and adjust fit parameters
    interactively (i.e clusters and mask radii) if initial clustering is
    not satisfactory.
3.  `peakipy fit` fits clusters of peaks using the `.csv` peak list
    generated (or edited) by the `read (edit)` script(s).
4.  `peakipy check` is used to check individual fits or groups of fits
    and make plots.

Head to the [instructions](./instructions.md) section for a
description of how to run these scripts. You can also use the
`--help` flag for instructions on how to run the programs from the
command line (e.g `peakipy read --help`).

How to install peakipy
----------------------

=== "poetry"
    !!! note ""
        Clone the [peakipy](https://github.com/j-brady/peakipy) repository from
        github:

            git clone https://github.com/j-brady/peakipy.git
            cd peakipy; poetry install

        If you don't have poetry I refer you to the [poetry documentation](https://poetry.eustace.io/docs/) for more details.

        At this point the package should be installed and the main scripts
        (`peakipy read`, `peakipy edit`, `peakipy fit` and `peakipy check`)
        should have been added to your path.

=== "pip"

    !!! note ""
            pip install peakipy

        Below is an example of an installation script and a basic use case :

            #!/bin/bash
            ##############################
            # make a virtual environment #
            ##############################
            python3.10 -m venv peakipy_env;
            source peakipy_env/bin/activate;

            ##############################
            # install peakipy            #
            ##############################
            pip install --upgrade pip;
            pip install peakipy;

            ##############################
            #  process some data!        #
            ##############################
            peakipy read peaks.a2 test.ft2 a2 --y-radius-ppm 0.213 --show;
            peakipy edit peaks.csv test.ft2; # adjust fitting parameters
            peakipy fit peaks.csv test.ft2 fits.csv --vclist vclist --max-cluster-size 15; # assuming you saved edited peaklist as peaks.csv
            # interactive checking
            peakipy check fits.csv test.ft2 --clusters 1 --clusters 2 --clusters 3 --colors purple green --show --outname tmp.pdf;
            # plots all the fits (first plane only)
            peakipy check fits.csv test.ft2 --first --colors purple green --show;

        Run this above code by sourcing the file e.g.
        `source file_containing_commands`

!!! note
    I recommend using a virtual environment:

        python3 -m venv peakipy_venv

    Then activate (bash):

        source peakipy_venv/bin/activate

    or (csh):

        source peakipy_venv/bin/activate.csh

    Once activated you can install peakipy with pip or poetry.


Requirements
-----------

=== "Latest"

    The latest version (2.0.0) of `peakipy` requires Python 3.10 or above (see `pyproject.toml` for details).

=== "0.2.0"

    `peakipy` version 0.2.0 which runs on Python 3.8 can be installed in the following ways:

        git clone --branch v0.2 https://github.com/j-brady/peakipy.git
        cd peakipy
        poetry install

    or:

        pip install peakipy==0.2.0
