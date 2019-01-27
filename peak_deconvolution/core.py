""" Functions for NMR peak deconvolution """

import numpy as np
from numpy import sqrt,log,pi,exp
from lmfit import Model

# constants
log2 = log(2)
s2pi = sqrt(2*pi)
spi = sqrt(pi)

π = pi
#√π = sqrt(π)
#√2π =  sqrt(2*π)

s2 = sqrt(2.0)
tiny = 1.e-13


def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.
    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))
    """
    return (amplitude/(sqrt(2*π)*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))


def lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Lorentzian function.
    lorentzian(x, amplitude, center, sigma) =
        (amplitude/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)
    """
    return (amplitude/(1 + ((1.0*x-center)/sigma)**2)) / (π*sigma)


def pvoigt2d(XY,amplitude=1.0, center_x=0.5,center_y=0.5, sigma_x=1.0,sigma_y=1.0, fraction=0.5):
    """ 2D pseudo-voigt model

        Arguments:
            -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
            -- amplitude: peak amplitude (gaussian and lorentzian)
            -- center_x: position of peak in x
            -- center_y: position of peak in y
            -- sigma_x: linewidth in x
            -- sigma_y: linewidth in y
            -- fraction: fraction of lorenztian in fit

        Returns:
            -- flattened array of Z values (use Z.reshape(X.shape) for recover)

    """
    x,y = XY
    sigma_gx = sigma_x / sqrt(2*log2)
    sigma_gy = sigma_y / sqrt(2*log2)
    # fraction same for both dimensions
    # super position of gaussian and lorentzian
    # then convoluted for x y
    pv_x = ((1-fraction)*gaussian(x, amplitude, center_x, sigma_gx) +
            fraction*lorentzian(x, amplitude, center_x, sigma_x))
    pv_y = ((1-fraction)*gaussian(y, amplitude, center_y, sigma_gy) +
            fraction*lorentzian(y, amplitude, center_y, sigma_y))
    return pv_x * pv_y


def make_mask(data,c_x,c_y,r_x,r_y):
    """ Create and elliptical mask

        ToDo:
            write explanation of function

        Arguments:
            data -- 2D array
            c_x  -- x center
            c_y  -- y center
            r_x  -- radius in x
            r_y  -- radius in y

        Returns:
            boolean mask of data.shape


    """
    a, b = c_y, c_x
    n_y, n_x = data.shape
    y, x = np.ogrid[-a:n_y-a, -b:n_x-b]
    mask = x**2./r_x**2. + y**2./r_y**2. <= 1.
    return mask


# ERROR CALCULATION
def r_square(data,residuals):
    """ Calculate R^2 value for fit

        Arguments:
            data -- array of data used for fitting
            residuals -- residuals for fit

        Returns:
            R^2 value
    """
    SS_tot = np.sum((data-data.mean())**2.)
    SS_res = np.sum(residuals**2.)
    r2 = 1 - SS_res/SS_tot
    return r2


# FUNCTIONS FOR DEALING WITH PARAMETERS
def update_params(params, peaks):
    """ Update lmfit parameters with values from Peak

        Arguments:
             -- params: lmfit parameter object
             -- peaks: list of Peak objects that parameters correspond to

        ToDo:
             -- deal with boundaries
             -- currently positions in points
             --

    """
    for peak in peaks:
        print(peak)
        for k,v in peak.param_dict().items():
            params[k].value = v
            print("update", k, v)
            if "center" in k:
                params[k].min = v-3
                params[k].max = v+3
                print("setting limit of %s, min = %.3e, max = %.3e" % (k, params[k].min, params[k].max))
            elif "sigma" in k:
                params[k].min = 0.0
                params[k].max = 1e6
                print("setting limit of %s, min = %.3e, max = %.3e" % (k, params[k].min, params[k].max))
            elif "fraction" in k:
                # fix weighting between 0 and 1
                params[k].min = 0.0
                params[k].max = 1.0


def make_models(model, peaks):
    """ Make composite models for multiple peaks

        Arguments:
            -- models
            -- peaks: list of Peak objects [<Peak1>,<Peak2>,...]

        Returns:
            -- mod: Composite model containing all peaks
            -- p_guess: params for composite model with starting values

    """
    if len(peaks)<2:
        # make model for first peak
        mod = Model(model, prefix=peaks[0].prefix)
        # add parameters
        p_guess = mod.make_params(**peaks[0].param_dict())

    elif len(peaks)>1:
        # make model for first peak
        mod = Model(model, prefix=peaks[0].prefix)

        for i in peaks[1:]:
            mod += Model(model, prefix=i.prefix)

        p_guess = mod.make_params()
        # add Peak params to p_guess
        update_params(p_guess, peaks)

    return mod, p_guess


def fix_params(params,to_fix):
    """ Set parameters to fix

        Arguments:
             -- params: lmfit parameters
             -- to_fix: parameter name to fix

        Returns:
            -- params: updated parameter object
    """
    for k in params:
        for p in to_fix:
            if p in k:
                params[k].vary = False

    return params


def get_params(params,name):
    ps = []
    ps_err = []
    names = []
    for k in params:
        if name in k:
            ps.append(params[k].value)
            ps_err.append(params[k].stderr)
            names.append(k)
    return ps, ps_err, names




class Peak():


    def __init__(self,center_x,center_y,amplitude,prefix="",sigma_x=1.0,sigma_y=1.0,assignment="None"):
        """ Peak class 
            
            Data structure for nmrpeak
        """
        self.center_x = center_x
        self.center_y = center_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.amplitude = amplitude
        self.prefix = prefix 
        self.assignment = assignment
                      
    def param_dict(self):
        """ Make dict of parameter names using prefix """
        str_form = lambda x: "%s%s"%(self.prefix,str(x))
        par_dict = {str_form("center_x"):self.center_x,
                    str_form("center_y"):self.center_y,
                    str_form("sigma_x"):self.sigma_x,
                    str_form("sigma_y"):self.sigma_y,
                    str_form("amplitude"):self.amplitude,
                    str_form("fraction"):0.5}
        return par_dict

    def mask(self,data,r_x,r_y):
        # data is spectrum containing peak
        a, b = self.center_y, self.center_x
        n_y, n_x = data.shape
        y,x = np.ogrid[-a:n_y-a, -b:n_x-b]
        # create circular mask
        mask = x**2./r_x**2. + y**2./r_y**2. <= 1.0
        return mask

    def __str__(self):
        return "Peak: x=%.1f, y=%.1f, amp=%.1f, fraction=%.1f, prefix=%s, assignment=%s"%\
                (self.center_x,self.center_y,self.amplitude,0.5,self.prefix,self.assignment)

