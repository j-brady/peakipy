==========
Lineshapes
==========

Here are the lineshapes used by peakipy. To select a specific lineshape to fit you can add ``--lineshape=`` flag when running ``fit_peaks``.

For example, ::

    fit_peaks test.csv test1.ft2 fits.csv --lineshape=G



Would fit to a Gaussian lineshape in both dimensions. Other options are L or PV for Lorentzian or Pseudo-Voigt in both dimensions, respectively.
If you want to fit a seperate lineshape for the indirect dimension then PV_PV, PV_G, PV_L, G_L allow you to fit Pseudo-Voigt with seperate X and Y fraction parameters, Pseudo-Voigt in X and Gaussian/Lorentzian in Y or Gaussian in X and Lorentzian in Y, respectively. 

Gaussian
--------

:math:`\frac{1}{\sigma_g\sqrt{2\pi}}\exp \frac{-(x - center)^2 } { 2 \sigma_g^2}`

Lorentzian
----------

:math:`\frac{1}{\pi} \left( \frac{\sigma}{(x - center)^2 + \sigma^2}\right)`

Pseudo-Voigt
------------

:math:`\frac{(1-fraction)}{\sigma_g\sqrt{2\pi}}\exp \frac{-(x - center)^2 }{ 2 \sigma_g^2} + \frac{fraction}{\pi} \left( \frac{\sigma}{(x - center)^2 + \sigma^2}\right)`

This is the default lineshape used for fitting and the fraction of G or L is assumed to be the same for both dimensions.


Fit quality
-----------

Currently, I don't have a good method for determining the fit quality as it is difficult to have an accurate estimate of the noise.
However, peakipy does perform a crude :math:`\chi^2` estimate (using normalised intensity values) and also calculates the linear correlation between the NMR data and the simulated data from the fit. If the slope deviates by more than 0.05 from 1.0 then it is advised that you check the fit. However, this is not totally robust and it is best to check the fit quality by plotting the data using the ``check_fits`` script.

:math:`\sum \frac{(y - y_{sim})^2}{\left| y_{sim} \right|}`
