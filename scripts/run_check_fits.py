#!/usr/bin/env python3
""" Script for checking fits and editing fit params

    Usage:
        run_check_fits.py <peaklist> <data> [options]

    Arguments:
        <peaklist>  peaklist output from read_peaklist.py (csv, tab or pkl)
        <data>      NMRPipe data

    Options:
        --dims=<id,f1,f2>  order of dimensions [default: 0,1,2]

"""
import os
from pathlib import Path
from shutil import which
from docopt import docopt
#import subprocess

args = docopt(__doc__)
peaklist = Path(args.get("<peaklist>"))
data = args.get("<data>")
dims = args.get("--dims")

script = which("check_fits.py")

#p = subprocess.Popen(['bokeh','serve', '--show', script, '--args', peaklist, data, f'--dims={dims}' ])

os.system(f"bokeh serve --show {script} --args {peaklist} {data} --dims={dims}")
