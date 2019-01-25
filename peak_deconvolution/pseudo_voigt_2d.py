from lmfit.models import PseudoVoigtModel
from lmfit import Model, report_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sqrt,log,pi,exp
import nmrglue as ng
import pandas as pd
from collections import namedtuple



log2 = log(2)
s2pi = sqrt(2*pi)
spi = sqrt(pi)
s2 = sqrt(2.0)
tiny = 1.e-13


def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.
    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))
    """
    return (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))


def lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Lorentzian function.
    lorentzian(x, amplitude, center, sigma) =
        (amplitude/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)
    """
    return (amplitude/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)


def voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None):
    """Return a 1-dimensional Voigt function.
    voigt(x, amplitude, center, sigma, gamma) =
        amplitude*wofz(z).real / (sigma*s2pi)
    see https://en.wikipedia.org/wiki/Voigt_profile
    """
    if gamma is None:
        gamma = sigma
    z = (x-center + 1j*gamma) / (sigma*s2)
    return amplitude*wofz(z).real / (sigma*s2pi)


def pvoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5):
    """Return a 1-dimensional pseudo-Voigt function.
    pvoigt(x, amplitude, center, sigma, fraction) =
       amplitude*(1-fraction)*gaussion(x, center, sigma_g) +
       amplitude*fraction*lorentzian(x, center, sigma)
    where sigma_g (the sigma for the Gaussian component) is
        sigma_g = sigma / sqrt(2*log(2)) ~= sigma / 1.17741
    so that the Gaussian and Lorentzian components have the
    same FWHM of 2*sigma.
    """
    sigma_g = sigma / sqrt(2*log2)
    return ((1-fraction)*gaussian(x, amplitude, center, sigma_g) +
             fraction*lorentzian(x, amplitude, center, sigma))


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


def update_params(params,peaks):
    for peak in peaks:
        print(peak)
        for k,v in peak.param_dict().items():
            params[k].value = v
            print("update",k,v)
            if "center" in k:
                params[k].min = v-3
                params[k].max = v+3
            elif "sigma" in k:
                params[k].min = 0.0
                print("setting limit of sigma")
                params[k].max = 1e6
            elif "fraction" in k:
                # fix weighting between 0 and 1
                params[k].min = 0.0
                params[k].max = 1.0


def make_models(model,peaks):
    """ Make composite models for multiple peaks 
        
        Arguments:
            -- models
            -- peaks: list of namedtuples
    """
    if len(peaks)<2:
        # make model for first peak
        mod = Model(model,prefix=peaks[0].prefix)
        # add parameters
        p_guess = mod.make_params(**peaks[0].param_dict())

    elif len(peaks)>1:

        mod = Model(model,prefix=peaks[0].prefix)

        for i in peaks[1:]:
            mod += Model(model,prefix=i.prefix)

        p_guess = mod.make_params()

        update_params(p_guess,peaks)

    return mod, p_guess


def fix_params(params,to_fix):
    """ Set parameters to fix """
    for k in params:
        for p in to_fix:
            if p in k:
                params[k].vary = False


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


# Peak = namedtuple("Peak",["center_x","center_y","amplitude","prefix"])
def data_mask(center_x,center_y,shape,r_x,r_y):
    a, b = center_y, center_x
    n_y, n_x = shape
    y,x = np.ogrid[-a:n_y-a, -b:n_x-b]
    # create circular mask
    mask = x**2./r_x**2. + y**2./r_y**2. <= 1.0
    return mask

class Peak():

    def __init__(self,center_x,center_y,amplitude,prefix):
        self.center_x = center_x
        self.center_y = center_y
        self.amplitude = amplitude
        self.prefix = prefix

    def param_dict(self):
        """ Make dict of parameter names using prefix """
        str_form = lambda x: "%s%s"%(self.prefix,str(x))
        par_dict = {str_form("center_x"):self.center_x,
                    str_form("center_y"):self.center_y,
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
        return "Peak: x=%d, y=%d, amp=%.1f, fraction=%.1f, prefix=%s"%\
                (self.center_x,self.center_y,self.amplitude,0.5,self.prefix)

if __name__ == "__main__":

    # # Fitting real data

    dic,data = ng.pipe.read("test.ft2")
    # read peak lists
    path = "test.tab"

    with open(path) as f:
        for line in f:
            if line.startswith("VARS"):
                names = line.split()[1:]

    df =  pd.read_table(path,skiprows=6,names=names,delim_whitespace=True)
    groups = [[0],[1],[2,3,4],[15,16]]
    groups_as_peaks = []
    for g in groups:
        group = []
        for i in g:
            cen_x = df.iloc[i].X_AXIS
    #        print("cen_x",cen_x)
            cen_y = df.iloc[i].Y_AXIS
            amp = df.iloc[i].VOL/100.
            prefix = "_%s_"%str(i)
            print("prefix",prefix)
            group.append(Peak(center_x=cen_x,center_y=cen_y,amplitude=amp,prefix=prefix))
        groups_as_peaks.append(group)

    #print(df.X1)
    #print(df.Y1)
    #print(groups_as_peaks)
    x_radius = 5
    y_radius = 5
    mod, p_guess = make_models(pvoigt2d,groups_as_peaks[0])

    #for k in p_guess:
    #    if "center_x" in k: 
    #        print("center_x",p_guess[k].value)
    x = np.arange(1,data.shape[1]+1)
    y = np.arange(1,data.shape[0]+1)
    XY = np.meshgrid(x,y)
    X,Y = XY
    print("X",X.shape,"Y",Y.shape)
    i = 1
    for group in groups_as_peaks:

        mask = np.zeros(data.shape,dtype=bool)
        mod, p_guess = make_models(pvoigt2d,group)
        cen_x = [p_guess[k].value for k in p_guess if "center_x" in k ]
        cen_y = [p_guess[k].value for k in p_guess if "center_y" in k ]
        print("Group %d"%i)
        i +=1
        for peak in group:
            mask += peak.mask(data,x_radius,y_radius)
            print(peak)
    #    print(cen_x,min(cen_x))
    #    print(cen_y,min(cen_y))
        max_x, min_x = int(round(max(cen_x)))+x_radius, int(round(min(cen_x)))-x_radius
        max_y, min_y = int(round(max(cen_y)))+y_radius, int(round(min(cen_y)))-y_radius
    #    mask += data_mask(cen_x,cen_y,data.shape,x_radius,y_radius)
        #print(min_x,max_x,min_y,max_y)
    #    peak_slice = data[min_y:max_y,
    #                      min_x:max_x]
    #    x,y = np.arange(min_x,max_x), np.arange(min_y,max_y)
    #    XY = np.meshgrid(x,y)
    #    X,Y = XY

        peak_slices = data.copy()[mask]
        XY_slices = [X.copy()[mask], Y.copy()[mask]]
        out = mod.fit(peak_slices,XY=XY_slices,params=p_guess)
        Zsim = mod.eval(XY=XY,params=out.params)
        print(report_fit(out.params))
        Zsim[~mask] = np.nan
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Z_plot = data.copy()
        Z_plot[~mask] = np.nan
        ax.plot_wireframe(X[min_y-1:max_y,min_x-1:max_x],Y[min_y-1:max_y,min_x-1:max_x],Z_plot[min_y-1:max_y,min_x-1:max_x])
        ax.set_xlabel("x")
        ax.plot_wireframe(X[min_y-1:max_y,min_x-1:max_x],Y[min_y-1:max_y,min_x-1:max_x],Zsim[min_y-1:max_y,min_x-1:max_x],color="r")
    #    ax.set_ylim(min_y,max_y)
    #    ax.set_xlim(min_x,max_x)
        plt.show()

    #mod = Model(pvoigt2d)
    #
    #def fit_peak(peak):
    #    
    #    x_radius = 5
    #    y_radius = 4
    #
    #    params = mod.make_params(sigma_x=1.0,sigma_y=1.0,
    #            amplitude=peak.VOL,center_y=peak.Y1,center_x=peak.X1)
    #    params["fraction"].min = 0.0
    #    params["fraction"].max = 1.0
    #
    #    x = np.arange(peak.X1-x_radius,peak.X1+x_radius)
    #    y = np.arange(peak.Y1-y_radius,peak.Y1+y_radius)
    #    XY = np.meshgrid(x,y)
    #    X,Y = XY
    #
    #    peak_slice = data[peak.Y1-y_radius:peak.Y1+y_radius,
    #                      peak.X1-x_radius:peak.X1+x_radius]
    #
    #    out = mod.fit(peak_slice,XY=XY,params=params)
    #    report_fit(out.params)
    #
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111, projection='3d')
    #    ax.plot_wireframe(X,Y,peak_slice)
    #    ax.set_xlabel("x")
    #    ax.plot_wireframe(X,Y,out.best_fit,color="r")
    #    plt.show()
    #
    #df.apply(lambda x: fit_peak(x),axis=1)
    #
    ## # deconvolute
    #peaks = data[460:480,212:220]
    #x,y = np.arange(212,220),np.arange(460,480)
    #X,Y = np.meshgrid(x,y)
    #XY = np.meshgrid(x,y)
    ##plt.imshow(peaks)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_wireframe(X,Y,peaks)
    #ax.set_xlabel("x")
    #
    #from collections import namedtuple
    #
    #Peak = namedtuple("Peak",["x","y","amplitude","prefix"])
    #p1 = Peak(x=215,y=462,amplitude=1e5,prefix="_1_") 
    #p2 = Peak(x=216,y=466,amplitude=5e4,prefix="_2_")
    #p3 = Peak(x=217,y=469,amplitude=1e4,prefix="_3_")
    #p4 = Peak(x=215,y=474,amplitude=1e4,prefix="_4_")
    #for p in [p1,p2,p3,p4]:
    #    ax.plot([p.x],[p.y],[1],"ro")
    #
    #mod = Model(pvoigt2d,prefix=p1.prefix)
    #mod += Model(pvoigt2d,prefix=p2.prefix)
    #mod += Model(pvoigt2d,prefix=p3.prefix)
    #mod += Model(pvoigt2d,prefix=p4.prefix)
    #params = mod.make_params(_1_center_x=p1.x,
    #                        _1_center_y=p1.y,
    #                        _1_amplitude=p1.amplitude,
    #                        _1_sigma_x = 1.0,
    #                        _1_sigma_y = 1.0,
    #                        _2_center_x=p2.x,
    #                        _2_center_y=p2.y,
    #                        _2_amplitude=p2.amplitude,
    #                        _2_sigma_x = 1.0,
    #                        _2_sigma_y = 1.0,
    #                        _3_center_x=p3.x,
    #                        _3_center_y=p3.y,
    #                        _3_amplitude=p3.amplitude,
    #                        _3_sigma_x = 1.0,
    #                        _3_sigma_y = 1.0,
    #                        _4_center_x=p4.x,
    #                        _4_center_y=p4.y,
    #                        _4_amplitude=p4.amplitude,
    #                        _4_sigma_x = 1.0,
    #                        _4_sigma_y = 1.0,
    #                       )
    #
    #
    #out = mod.fit(peaks,XY=XY,params=params)
    #print(report_fit(out.params))
    #
    #ax.plot_wireframe(X,Y,out.best_fit,color="r")
    #plt.show()
