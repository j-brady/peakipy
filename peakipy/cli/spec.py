#!/usr/bin/env python3
"""
	Usage: spec.py <yaml_file>
           spec.py make <new_yaml_file>

	Plot NMRPipe spectra overlays using nmrglue and matplotlib. This is my attempt to make a general script for
    plotting NMR data.


    Below is an example yaml file for input

        # This first block is global parameters which can be overridden by adding the desired argument
        # to your list of spectra. One exception is "colors" which if set in global params overrides the
        # color option set for individual spectra as the colors will now cycle through the chosen matplotlib
        # colormap
		cs: 10e5                        # contour start
		contour_num: 10                 # number of contours
		contour_factor: 1.2             # contour factor
		colors: Set1                    # must be matplotlib.cm colormap

		outname: ["overlay.pdf","overlay.png"] # either single value or list of output names

        # Here is where your list of spectra to plot goes
		spectra:

				- fname: test.ft2
				  label: write legend here
				  contour_num: 1
				  linewidths: 1

	Options:
	    -h --help
        -v --version


    Dependencies:

        -- python3
        -- matplotlib, pyyaml, numpy, nmrglue, pandas and docopt

    
    peakipy - deconvolute overlapping NMR peaks
    Copyright (C) 2019  Jacob Peter Brady

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

yaml_file = """
##########################################################################################################
#  This first block is global parameters which can be overridden by adding the desired argument          #
#  to your list of spectra. One exception is "colors" which if set in global params overrides the        #
#  color option set for individual spectra as the colors will now cycle through the chosen matplotlib    #
#  colormap                                                                                              #
##########################################################################################################

cs: 10e5                        # contour start
contour_num: 10                 # number of contours
contour_factor: 1.2             # contour factor
colors: Set1                    # must be matplotlib.cm colormap

outname: ["overlay.pdf","overlay.png"] # either single value or list of output names
ncol: 1 #  tells matplotlib how many columns to give the figure legend - if not set defaults to 2

# Here is where your list of spectra to plot goes
spectra:

        - fname: test.ft2
          label: some information
          contour_num: 1
          linewidths: 1
"""


def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print("onpick points:", points)
