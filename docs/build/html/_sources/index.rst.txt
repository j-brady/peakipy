.. peakipy documentation master file, created by
   sphinx-quickstart on Sat Mar 30 12:13:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================
Say hello to peakipy!
=====================

Simple deconvolution of NMR peaks for extraction of intensities.
Provided an NMRPipe format spectrum (2D or Pseudo 3D) and a peak list (NMRPipe, Sparky or Analysis2), overlapped peaks are automatically/interactively clustered and groups of overlapped peaks are fitted together using Gaussian, Lorentzian or Pseudo-Voigt (Gaussian + Lorentzian) lineshape.

.. image:: bokeh.png

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   usage/installation
   usage/quickstart
   usage/instructions
   usage/examples
   usage/lineshapes 
   usage/code

Search
==================

* :ref:`search`
