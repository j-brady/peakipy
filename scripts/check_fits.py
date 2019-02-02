#!/usr/bin/env python3
""" Script for checking fits and editing fit params

    Usage:
        check_fits.py <fits> <data> [options]

    Options:

        --peak=<ASSIGNMENT>  check specific peak
        --clusters=<number>  check clusters

    Arguments:
        <fits>  fits output from fit_pipe_peaks.py csv, tab or pkl

"""

import os

from docopt import docopt

from peak_deconvolution.core import pvoigt2d

args = docopt(__doc__)

print(args)

fits = args.get("<fits>")
ext = os.path.splitext(fits)

dic, data = ng.pipe.read(args.get("<data>"))

# READ FITS
if ext == ".csv":
    fits = pd.read_csv(fits)

elif ext == ".tab":
    fits = pd.read_csv(fits, sep="\t")

else:
    fits = pd.read_pickle(fits)

clusters = fits.groupby("CLUSTID")
