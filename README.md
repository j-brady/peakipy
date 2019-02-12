# Peak deconvolution

## Description

Simple deconvolution of NMR peaks for extraction of intensities. Provided an NMRPipe format spectrum (2D or Pseudo 3D) and a peak list (NMRPipe, Sparky or Analysis2), overlapped peaks are automatically/interactively clustered and groups of overlapped peaks are fitted together using Gaussian, Lorentzian or Pseudo-Voigt (Gaussian + Lorentzian).

## Installation

With poetry...

```bash
cd peak_deconvolution; poetry install
```

With setup.py you will need python3.6 or greater installed.

```bash
cd peak_deconvolution; python setup.py install
```

## Inputs

There are three main scripts.

1. read_peaklist.py is used to convert peak list and select clusters peaks.
2. run_check_fits.py is used to check fit parameters (i.e clusters and mask radii)
3. fit_peaks.py is used to fit clusters of peaks


### Running read_peaklist.py

First you need a peak list either Sparky or Analysis2 format.

Here is an example of how to run read_peaklist.py

```bash
read_peaklist.py peaks.sparky test.ft2 --show --outfmt=csv
```

This will convert your peaklist to into a `pandas DataFrame` use `threshold_otsu` from `scikit-learn` to determine a cutoff for selecting overlapping peaks.
These are subsequently grouped into clusters ("CLUSTID" column a la NMRPipe!)


```bash
read_peaklist.py peaks.sparky test.ft2 --dims=0,1,2 --show --outfmt=csv
```
  




## Protocol

Initial parameters for FWHM,  



## Outputs

1. Pandas DataFrame containing fitted intensities/linewidths/centers etc.
2. If --plot=<path> option selected the first plane of each fit will be plotted in <path> with the files named according to the cluster ID (CLUSTID) of the fit.


## Pseudo-Voigt model

Fraction parameter is fraction of Lorentzian lineshape.

## Test data

To test the program for yourself cd into the test dir and 

## To do

1. Add function to read_peaklist.py that allows output to FuDA format parameter file

2. add flag to include vclist to output dataframe

3. Normalize peaks

4. Multiprocessor

5. Script to check fits and add column with x_radius and y_radius so that you can edit the parameters and rerun specific fits.
