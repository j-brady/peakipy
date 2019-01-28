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

1. First you need a peak list

## Outputs

1. Pandas DataFrame containing fitted intensities/linewidths/centers etc.


## Pseudo-Voigt model

Fraction parameter is fraction of Lorentzian lineshape.


## To do

1. Add function to read_peaklist.py that allows output to FuDA format parameter file

2. Add flag to define lineshapes

3. starting amplitude is not close to fitted amplitude need to sort this out
