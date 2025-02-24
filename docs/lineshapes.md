Lineshapes
==========

Here are the lineshapes used by peakipy. Use the `--lineshape` option in `peakipy fit` to select a lineshape for fitting.

For example,

    peakipy fit test.csv test1.ft2 fits.csv --lineshape G

would fit to a Gaussian lineshape in both dimensions. Other options are
V, L or PV for Voigt Lorentzian or Pseudo-Voigt in both dimensions,
respectively. If you want to fit a seperate lineshape for the indirect
dimension then PV_PV allows you to fit a Pseudo-Voigt with seperate X
and Y fraction parameters

Gaussian
--------

$$A\frac{1}{\sigma_g\sqrt{2\pi}} e^{\frac{-(x - center)^2 } { 2 \sigma_g^2}}$$


Where $A$ is the amplitude (amp), $\sigma_g$ is the standard deviation and $center$ is the center of the Gaussian function.
The full width at half maximum (FWHM) is $2\sigma_g\sqrt{2ln2}$.

Lorentzian
----------

$$A\frac{1}{\pi} \left( \frac{\sigma}{(x - center)^2 + \sigma^2}\right)$$

Where $A$ is the amplitude (amp), $\sigma$ is the standard deviation of the gaussian function and $center$ is the center of the Gaussian peak.
The full width at half maximum (FWHM) is $2\sigma$

Pseudo-Voigt
------------

$$A\left[\frac{(1-fraction)}{\sigma_g\sqrt{2\pi}} e^{\frac{-(x - center)^2 }{ 2 \sigma_g^2}} + \frac{fraction}{\pi} \left( \frac{\sigma}{(x - center)^2 + \sigma^2}\right)\right]$$

This is the default lineshape used for fitting and the fraction of G or
L is assumed to be the same for both dimensions. The `--lineshape PV_PV`
option will fit a seperate pseudo-voigt lineshape in each dimension
(i.e. fraction_x and fraction_y parameters).

Fit quality
-----------

Fit quality can be evaluated by inspecting the contour plot of residuals
that is generated when viewing fits interactively. $\chi^2$ and
$\chi_{red}^2$ are calculated using the noise estimate from `--noise` or
the threshold value calculated from `threshold_otsu` if `--noise` is not
set explicitly. Peakipy does calculate the linear correlation between
the NMR data and the simulated data from the fit. If the slope deviates
by more than 0.05 from 1.0 then it is advised that you check the fit.
However, this is not totally robust and it is best to check fit quality
by plotting the data using the `peakipy check` command.

Amplitudes
----------

The amplitude parameter ($A$) for each peak can be thought of as a scaling factor on the above lineshape functions. The lineshape functions themselves (i.e. everything after the $A$) are normalized between 0 and 1. You should ensure that the baseline of your data is around 0 otherwise this could result in systematic errors in fitted lineshape parameters (consider adding polynomial baseline corrections to your data if this is the case). For clarity, the `amp` column in the output fits `.csv` refers to the amplitude parameter and is synonymous with the volume of the peak. The `height` column is calculated based on the fitted lineshape parameters. 

!!! note
    
	Changing peak fitting settings such as masking radius could result differences in the fitted linewidths.
