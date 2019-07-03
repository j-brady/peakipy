=======================
Quickstart instructions
=======================

Inputs
------

1. Peak list (see :doc:`instructions`)
2. NMRPipe frequency domain dataset (2D or Pseudo 3D)

There are four main scripts.

1. ``peakipy read`` converts your peak list into a ``.csv`` file and selects clusters of peaks.
2. ``peakipy edit`` is used to check and adjust fit parameters interactively (i.e clusters and mask radii) if initial clustering is not satisfactory.
3. ``peakipy fit`` fits clusters of peaks using the ``.csv`` peak list generated (or edited) by the ``read (edit)`` script(s).
4. ``peakipy check`` is used to check individual fits or groups of fits and make plots.

Head to the :doc:`instructions` section for a description of how to run these scripts.
You can also use the ``-h`` or ``--help`` flags for instructions on how to run the programs from the command line (e.g ``peakipy read -h``).


