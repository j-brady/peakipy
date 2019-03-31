==========
Lineshapes
==========

Here are the lineshapes used by peakipy.

Gaussian
--------

:math:`\frac{1}{\sigma_g\sqrt{2\pi}}\exp \frac{-(x - center)^2 } { 2 \sigma_g^2}`

Lorentzian
----------

:math:`\frac{1}{\pi} \left( \frac{\sigma}{(x - center)^2 + \sigma^2}\right)`

Pseudo-Voigt
------------

:math:`\frac{(1-fraction)}{\sigma_g\sqrt{2\pi}}\exp \frac{-(x - center)^2 }{ 2 \sigma_g^2} + \frac{fraction}{\pi} \left( \frac{\sigma}{(x - center)^2 + \sigma^2}\right)`


