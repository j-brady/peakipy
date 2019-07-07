""" 

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


import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import pandas as pd

from numba import jit
from numpy import sqrt, log, pi, exp
from lmfit import Model
from lmfit.model import ModelResult
from lmfit.models import LinearModel
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button


# constants
log2 = log(2)
π = pi


@jit(nopython=True)
def gaussian(x, center=0.0, sigma=1.0):
    """ 1-dimensional Gaussian function.

        gaussian(x, center, sigma) = 
            (1/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))
        
        :math:`\\frac{1}{ \sqrt{2\pi} } exp \left( \\frac{-(x-center)^2}{2 \sigma^2} \\right)`
        
        :param x: x
        :param center: center
        :param sigma: sigma
        :type x: numpy.array
        :type center: float
        :type sigma: float

        :return: 1-dimensional Gaussian
        :rtype: numpy.array

    """
    return (1.0 / (sqrt(2 * π) * sigma)) * exp(
        -(1.0 * x - center) ** 2 / (2 * sigma ** 2)
    )


@jit(nopython=True)
def lorentzian(x, center=0.0, sigma=1.0):
    """ 1-dimensional Lorentzian function.

        lorentzian(x, center, sigma) =
            (1/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)

        :math:`\\frac{1}{ 1+ \left( \\frac{x-center}{\sigma}\\right)^2} / (\pi\sigma)`

        :param x: x
        :param center: center
        :param sigma: sigma
        :type x: numpy.array
        :type center: float
        :type sigma: float

        :return: 1-dimensional Lorenztian 
        :rtype: numpy.array

    """
    return (1.0 / (1 + ((1.0 * x - center) / sigma) ** 2)) / (π * sigma)


@jit(nopython=True)
def pseudo_voigt(x, center=0.0, sigma=1.0, fraction=0.5):
    """ 1-dimensional Pseudo-voigt function
    
        Superposition of Gaussian and Lorentzian function

        :math:`(1-\phi) G(x,center,\sigma_g) + \phi L(x, center, \sigma)`

        Where :math:`\phi` is the fraction of Lorentzian lineshape and :math:`G` and :math:`L` are Gaussian and
        Lorentzian functions, respectively.

        :param x: data
        :type x: numpy.array
        :param center: center of peak
        :type center: float
        :param sigma: sigma of lineshape
        :type sigma: float
        :param fraction: fraction of lorentzian lineshape (between 0 and 1)
        :type fraction: float

        :return: pseudo-voigt function
        :rtype: numpy.array

    """
    sigma_g = sigma / sqrt(2 * log2)
    pv = (1 - fraction) * gaussian(x, center, sigma_g) + fraction * lorentzian(
        x, center, sigma
    )
    return pv


# @jit(nopython=True)
def pvoigt2d(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """ 2D pseudo-voigt model

        :math:`(1-fraction) G(x,center,\sigma_{gx}) + (fraction) L(x, center, \sigma_x) * (1-fraction) G(y,center,\sigma_{gy}) + (fraction) L(y, center, \sigma_y)`

        :param XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
        :type XY: numpy.array

        :param center_x: center of peak in x
        :type center_x: float

        :param center_y: center of peak in x
        :type center_y: float

        :param sigma_x: sigma of lineshape in x
        :type sigma_x: float

        :param sigma_y: sigma of lineshape in y
        :type sigma_y: float

        :param fraction: fraction of lorentzian lineshape (between 0 and 1)
        :type fraction: float

        :return: flattened array of Z values (use Z.reshape(X.shape) for recovery)
        :rtype: numpy.array

    """

    x, y = XY
    # sigma_gx = sigma_x / sqrt(2 * log2)
    # sigma_gy = sigma_y / sqrt(2 * log2)
    # fraction same for both dimensions
    # super position of gaussian and lorentzian
    # then convoluted for x y
    # pv_x = (1 - fraction) * gaussian(x, center_x, sigma_gx) + fraction * lorentzian(
    #    x, center_x, sigma_x
    # )
    pv_x = pseudo_voigt(x, center_x, sigma_x, fraction)
    pv_y = pseudo_voigt(y, center_y, sigma_y, fraction)
    # pv_y = (1 - fraction) * gaussian(y, center_y, sigma_gy) + fraction * lorentzian(
    #    y, center_y, sigma_y
    # )
    return amplitude * pv_x * pv_y


# @jit(nopython=True)
def pv_l(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """ 2D lineshape model with pseudo-voigt in x and lorentzian in y

        Arguments
        =========

            -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
            -- amplitude: peak amplitude (gaussian and lorentzian)
            -- center_x: position of peak in x
            -- center_y: position of peak in y
            -- sigma_x: linewidth in x
            -- sigma_y: linewidth in y
            -- fraction: fraction of lorentzian in fit

        Returns
        =======

            -- flattened array of Z values (use Z.reshape(X.shape) for recovery)

    """

    x, y = XY
    pv_x = pseudo_voigt(x, center_x, sigma_x, fraction)
    pv_y = pseudo_voigt(y, center_y, sigma_y, 1.0)  # lorentzian
    return amplitude * pv_x * pv_y


# @jit(nopython=True)
def pv_g(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """ 2D lineshape model with pseudo-voigt in x and gaussian in y

        Arguments
        ---------

            -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
            -- amplitude: peak amplitude (gaussian and lorentzian)
            -- center_x: position of peak in x
            -- center_y: position of peak in y
            -- sigma_x: linewidth in x
            -- sigma_y: linewidth in y
            -- fraction: fraction of lorentzian in fit

        Returns
        -------

            -- flattened array of Z values (use Z.reshape(X.shape) for recovery)

    """
    x, y = XY
    pv_x = pseudo_voigt(x, center_x, sigma_x, fraction)
    pv_y = pseudo_voigt(y, center_y, sigma_y, 0.0)  # gaussian
    return amplitude * pv_x * pv_y


# @jit(nopython=True)
def pv_pv(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction_x=0.5,
    fraction_y=0.5,
):
    """ 2D lineshape model with pseudo-voigt in x and pseudo-voigt in y
        i.e. fraction_x and fraction_y params

        Arguments
        =========

            -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
            -- amplitude: peak amplitude (gaussian and lorentzian)
            -- center_x: position of peak in x
            -- center_y: position of peak in y
            -- sigma_x: linewidth in x
            -- sigma_y: linewidth in y
            -- fraction_x: fraction of lorentzian in x
            -- fraction_y: fraction of lorentzian in y

        Returns
        =======

            -- flattened array of Z values (use Z.reshape(X.shape) for recovery)

    """

    x, y = XY
    pv_x = pseudo_voigt(x, center_x, sigma_x, fraction_x)
    pv_y = pseudo_voigt(y, center_y, sigma_y, fraction_y)
    return amplitude * pv_x * pv_y


# @jit(nopython=True)
def gaussian_lorentzian(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """ 2D lineshape model with gaussian in x and lorentzian in y

        Arguments
        =========

            -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
            -- amplitude: peak amplitude (gaussian and lorentzian)
            -- center_x: position of peak in x
            -- center_y: position of peak in y
            -- sigma_x: linewidth in x
            -- sigma_y: linewidth in y
            -- fraction: fraction of lorentzian in fit

        Returns
        =======

            -- flattened array of Z values (use Z.reshape(X.shape) for recovery)

    """
    x, y = XY
    pv_x = pseudo_voigt(x, center_x, sigma_x, 0.0)  # gaussian
    pv_y = pseudo_voigt(y, center_y, sigma_y, 1.0)  # lorentzian
    return amplitude * pv_x * pv_y


def make_mask(data, c_x, c_y, r_x, r_y):
    """ Create and elliptical mask

        Generate an elliptical boolean mask with center c_x/c_y in points
        with radii r_x and r_y. Used to generate fit mask

        :param data: 2D array
        :type data: np.array

        :param c_x: x center
        :type c_x: float

        :param c_y: y center
        :type c_y: float

        :param r_x: radius in x
        :type r_x: float

        :param r_y: radius in y
        :type r_y: float

        :return: boolean mask of data.shape
        :rtype: numpy.array

    """
    a, b = c_y, c_x
    n_y, n_x = data.shape
    y, x = np.ogrid[-a : n_y - a, -b : n_x - b]
    mask = x ** 2.0 / r_x ** 2.0 + y ** 2.0 / r_y ** 2.0 <= 1.0
    return mask


def rmsd(residuals):
    return np.sqrt(np.sum(residuals ** 2.0) / len(residuals))


def fix_params(params, to_fix):
    """ Set parameters to fix

         
        :param params: lmfit parameters
        :type params: lmfit.Parameters

        :param to_fix: list of parameter name to fix
        :type to_fix: list

        :return: updated parameter object
        :rtype: lmfit.Parameters

    """
    for k in params:
        for p in to_fix:
            if p in k:
                params[k].vary = False

    return params


def get_params(params, name):
    ps = []
    ps_err = []
    names = []
    for k in params:
        if name in k:
            ps.append(params[k].value)
            ps_err.append(params[k].stderr)
            names.append(k)
    return ps, ps_err, names


def make_param_dict(peaks, data, lineshape="PV"):
    """ Make dict of parameter names using prefix """

    param_dict = {}

    for index, peak in peaks.iterrows():

        str_form = lambda x: "%s%s" % (to_prefix(peak.ASS), x)
        # using exact value of points (i.e decimal)
        param_dict[str_form("center_x")] = peak.X_AXISf
        param_dict[str_form("center_y")] = peak.Y_AXISf
        #  linewidth esimate
        param_dict[str_form("sigma_x")] = peak.XW / 2.0
        param_dict[str_form("sigma_y")] = peak.YW / 2.0
        # estimate peak volume
        amplitude_est = data[
            int(peak.Y_AXIS) - int(peak.YW) : int(peak.Y_AXIS) + int(peak.YW) + 1,
            int(peak.X_AXIS) - int(peak.XW) : int(peak.X_AXIS) + int(peak.XW) + 1,
        ].sum()

        param_dict[str_form("amplitude")] = amplitude_est

        if lineshape == "G":
            param_dict[str_form("fraction")] = 0.0
        elif lineshape == "L":
            param_dict[str_form("fraction")] = 1.0
        elif lineshape == "PV_PV":
            param_dict[str_form("fraction_x")] = 0.5
            param_dict[str_form("fraction_y")] = 0.5
        else:
            param_dict[str_form("fraction")] = 0.5

    return param_dict


def to_prefix(x):
    """
    Peak assignments with characters that are not compatible lmfit model naming
    are converted to lmfit "safe" names.

    :param x: Peak assignment to be used as prefix for lmfit model
    :type x: str

    :returns: lmfit model prefix (_Peak_assignment_)
    :rtype: str

    """
    # must be string
    if type(x) != str:
        x = str(x)

    prefix = "_" + x
    to_replace = [
        [".", "_"],
        [" ", ""],
        ["{", "_"],
        ["}", "_"],
        ["[", "_"],
        ["]", "_"],
        ["-", ""],
        ["/", "or"],
        ["?", "maybe"],
        ["\\", ""],
        ["(", "_"],
        [")", "_"],
    ]
    for p in to_replace:
        prefix = prefix.replace(*p)
    return prefix + "_"


def make_models(model, peaks, data, lineshape="PV", xy_bounds=None):
    """ Make composite models for multiple peaks

        :param model: lineshape function
        :type model: function

        :param peaks: instance of pandas.df.groupby("CLUSTID")
        :type peaks: pandas.df.groupby("CLUSTID")

        :param data: NMR data
        :type data: numpy.array

        :param lineshape: lineshape to use for fit (PV/G/L/PV_PV)
        :type lineshape: str

        :param xy_bounds: bounds for peak centers (+/-x, +/-y)
        :type xy_bounds: tuple

        :return mod: Composite lmfit model containing all peaks
        :rtype mod: lmfit.CompositeModel

        :return p_guess: params for composite model with starting values
        :rtype p_guess: lmfit.Parameters

    """
    if len(peaks) == 1:
        # make model for first peak
        mod = Model(model, prefix="%s" % to_prefix(peaks.ASS.iloc[0]))
        # add parameters
        param_dict = make_param_dict(peaks, data, lineshape=lineshape)
        p_guess = mod.make_params(**param_dict)

    elif len(peaks) > 1:
        # make model for first peak
        first_peak, *remaining_peaks = peaks.iterrows()
        mod = Model(model, prefix="%s" % to_prefix(first_peak[1].ASS))
        for index, peak in remaining_peaks:
            mod += Model(model, prefix="%s" % to_prefix(peak.ASS))

        param_dict = make_param_dict(peaks, data, lineshape=lineshape)
        p_guess = mod.make_params(**param_dict)
        # add Peak params to p_guess

    update_params(p_guess, param_dict, lineshape=lineshape, xy_bounds=xy_bounds)

    return mod, p_guess


def update_params(params, param_dict, lineshape="PV", xy_bounds=None):
    """ Update lmfit parameters with values from Peak

        :param params: lmfit parameters 
        :type params: lmfit.Parameters object
        :param param_dict: parameters corresponding to each peak in fit
        :type param_dict: dict
        :param lineshape: lineshape (PV, G, L, PV_PV etc.)
        :type lineshape: str
        :param xy_bounds: bounds on xy peak positions
        :type xy_bounds: tuple

        :returns: None
        :rtype: None

        ToDo:
             -- deal with boundaries
             -- currently positions in points

    """
    for k, v in param_dict.items():
        params[k].value = v
        # print("update", k, v)
        if "center" in k:
            if xy_bounds == None:
                # no bounds set
                pass
            else:
                if "center_x" in k:
                    # set x bounds
                    x_bound = xy_bounds[0]
                    params[k].min = v - x_bound
                    params[k].max = v + x_bound
                elif "center_y" in k:
                    # set y bounds
                    y_bound = xy_bounds[1]
                    params[k].min = v - y_bound
                    params[k].max = v + y_bound
                # pass
                print(
                    "setting limit of %s, min = %.3e, max = %.3e"
                    % (k, params[k].min, params[k].max)
                )
        elif "sigma" in k:
            params[k].min = 0.0
            params[k].max = 1e4
            # print(
            #    "setting limit of %s, min = %.3e, max = %.3e"
            #    % (k, params[k].min, params[k].max)
            # )
        elif "fraction" in k:
            # fix weighting between 0 and 1
            params[k].min = 0.0
            params[k].max = 1.0

            if lineshape == "G":
                params[k].vary = False
            elif lineshape == "L":
                params[k].vary = False

    # return params


def run_log(log_name="run_log.txt"):
    """ Write log file containing time script was run and with which arguments"""
    with open(log_name, "a") as log:
        sys_argv = sys.argv
        sys_argv[0] = Path(sys_argv[0]).name
        run_args = " ".join(sys_argv)
        time_stamp = datetime.now()
        time_stamp = time_stamp.strftime("%A %d %B %Y at %H:%M")
        log.write(f"# Script run on {time_stamp}:\n{run_args}\n")


def fit_first_plane(
    group,
    data,
    uc_dics,
    lineshape="PV",
    xy_bounds=None,
    verbose=False,
    log=None,
    noise=1.0,
):
    """ Deconvolute group of peaks

        :param group: pandas data from containing group of peaks using groupby("CLUSTID")
        :type group: pandas.core.groupby.generic.DataFrameGroupBy

        :param data: NMR data
        :type data: numpy.array

        :param uc_dics: nmrglue unit conversion dics {"f1":uc_f1,"f2":uc_f2}
        :type uc_dics: dict
        
        :param lineshape: lineshape to fit (PV, G, L, G_L, PV_L, PV_G or PV_PV)
        :type lineshape: str

        :param xy_bounds: set bounds on x y positions. None or (x_bound, y_bound)
        :type xy_bounds: tuple

        :param plot: dir to save wireframe plots
        :type plot: str

        :param show: interactive matplotlib plot
        :type show: bool

        :param verbose: print what is happening to terminal
        :type verbose: bool

        :param log: file
        :type log: str

        :param noise: estimate of spectral noise for calculation of :math:`\chi^2` and :math:`\chi^2_{red}`
        :type noise: float

        :return: FitResult
        :rtype: FitResult

    """
    shape = data.shape
    mask = np.zeros(shape, dtype=bool)

    if (lineshape == "PV") or (lineshape == "G") or (lineshape == "L"):
        mod, p_guess = make_models(
            pvoigt2d, group, data, lineshape=lineshape, xy_bounds=xy_bounds
        )

    elif lineshape == "G_L":
        mod, p_guess = make_models(
            gaussian_lorentzian, group, data, lineshape="PV", xy_bounds=xy_bounds
        )

    elif lineshape == "PV_G":
        mod, p_guess = make_models(
            pv_g, group, data, lineshape="PV", xy_bounds=xy_bounds
        )

    elif lineshape == "PV_L":
        mod, p_guess = make_models(
            pv_l, group, data, lineshape="PV", xy_bounds=xy_bounds
        )

    elif lineshape == "PV_PV":
        mod, p_guess = make_models(
            pv_pv, group, data, lineshape="PV_PV", xy_bounds=xy_bounds
        )

    # get initial peak centers
    cen_x = [p_guess[k].value for k in p_guess if "center_x" in k]
    cen_y = [p_guess[k].value for k in p_guess if "center_y" in k]

    for index, peak in group.iterrows():
        mask += make_mask(
            data, peak.X_AXISf, peak.Y_AXISf, peak.X_RADIUS, peak.Y_RADIUS
        )

    x_radius = group.X_RADIUS.max()
    y_radius = group.Y_RADIUS.max()

    max_x, min_x = (
        int(np.ceil(max(group.X_AXISf) + x_radius + 1)),
        int(np.floor(min(group.X_AXISf) - x_radius)),
    )
    max_y, min_y = (
        int(np.ceil(max(group.Y_AXISf) + y_radius + 1)),
        int(np.floor(min(group.Y_AXISf) - y_radius)),
    )

    #  deal with peaks on the edge of spectrum
    if min_y < 0:
        min_y = 0

    if min_x < 0:
        min_x = 0

    if max_y > shape[-2]:
        max_y = shape[-2]

    if max_x > shape[-1]:
        max_x = shape[-1]

    peak_slices = data.copy()[mask]
    # must be a better way to make the meshgrid
    x = np.arange(shape[-1])
    y = np.arange(shape[-2])
    XY = np.meshgrid(x, y)
    X, Y = XY

    XY_slices = np.array([X.copy()[mask], Y.copy()[mask]])
    weights = 1.0 / np.array([noise] * len(np.ravel(peak_slices)))
    out = mod.fit(peak_slices, XY=XY_slices, params=p_guess, weights=weights, method="leastsq")

    if verbose:
        print(out.fit_report())

    z_sim = mod.eval(XY=XY, params=out.params)
    z_sim[~mask] = np.nan
    z_plot = data.copy()
    z_plot[~mask] = np.nan
    #  also if peak position changed significantly from start then add warning

    _z_plot = z_plot[~np.isnan(z_plot)]
    _z_sim = z_sim[~np.isnan(z_sim)]

    linmod = LinearModel()
    linpars = linmod.guess(_z_sim, x=_z_plot)
    linfit = linmod.fit(_z_sim, x=_z_plot, params=linpars)
    slope = linfit.params["slope"].value
    #  number of peaks in cluster
    n_peaks = len(group)

    chi2 = out.chisqr
    redchi = out.redchi

    fit_str = f"""
    Cluster {peak.CLUSTID} containing {n_peaks} peaks - slope={slope:.3f}

        chi^2 = {chi2:.5f}
        redchi = {redchi:.5f}

    """
    if (slope > 1.05) or (slope < 0.95):
        fit_str += """
        🧐 NEEDS CHECKING 🧐
        """
        print(fit_str)
    else:
        print(fit_str)

    if log != None:
        log.write("".join("#" for _ in range(60)) + "\n\n")
        log.write(fit_str + "\n\n")
        # pass
    else:
        pass

    return FitResult(
        out=out,
        mask=mask,
        fit_str=fit_str,
        log=log,
        group=group,
        uc_dics=uc_dics,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        X=X,
        Y=Y,
        Z=z_plot,
        Z_sim=z_sim,
        peak_slices=peak_slices,
        XY_slices=XY_slices,
        weights=weights,
        mod=mod,
    )


class FitResult:
    """ Data structure for storing fit results """

    def __init__(
        self,
        out: ModelResult,
        mask: np.array,
        fit_str: str,
        log: str,
        group: pd.core.groupby.generic.DataFrameGroupBy,
        uc_dics: dict,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        X: np.array,
        Y: np.array,
        Z: np.array,
        Z_sim: np.array,
        peak_slices: np.array,
        XY_slices: np.array,
        weights: np.array,
        mod: Model,

    ):
        """ Store output of fit_first_plane function """
        self.out = out
        self.mask = mask
        self.fit_str = fit_str
        self.log = log
        self.group = group
        self.uc_dics = uc_dics
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Z_sim = Z_sim
        self.peak_slices = peak_slices
        self.XY_slices = XY_slices
        self.weights = weights
        self.mod = mod

    def check_shifts(self):
        """ Calculate difference between initial peak positions 
            and check whether they moved too much from original
            position
            
        """
        pass

    def jackknife(self):
        """ perform jackknife sampling to estimate fitting errors

        """
        jk_results = []
        for i in range(len(self.peak_slices)):
            peak_slices = np.delete(self.peak_slices, i, None)
            X = np.delete(self.XY_slices[0], i, None)
            Y = np.delete(self.XY_slices[1], i, None)
            weights = np.delete(self.weights, i, None)
            jk_results.append(self.mod.fit(peak_slices, XY=[X,Y], params=self.out.params, weights=weights))

        #print(jk_results)
        amps = []
        sigmas = []
        names = []
        with open("test_jackknife","w") as f:
            for i in jk_results:
                f.write(i.fit_report())
                amp, amp_err, name = get_params(i.params, "amp")
                sigma, sigma_err, name = get_params(i.params, "sigma_x")
                f.write(f"{amp},{amp_err},{name}\n")
                amps.extend(amp)
                names.extend(name)
                sigmas.extend(sigma)

            df = pd.DataFrame({"amp":amps, "name":names, "sigma":sigmas})
            grouped = df.groupby("name")
            mean_amps = grouped.amp.mean()
            std_amps = grouped.amp.std()
            mean_sigmas = grouped.sigma.mean()
            std_sigmas = grouped.sigma.std()
            f.write("#####################################\n")
            f.write(f"{mean_amps}, {std_amps}, {mean_sigmas}, {std_sigmas}")
            f.write(self.out.fit_report())
            f.write("#####################################\n")
        #print(amps)
        #mean = np.mean(amps)
        #std =  np.std(amps)
        return JackKnifeResult(mean=mean_amps, std=std_amps)


    def plot(self, plot_path=None, show=False, nomp=True):
        """ Matplotlib interactive plot of the fits """

        if plot_path != None:
            plot_path = Path(plot_path)
            plot_path.mkdir(parents=True, exist_ok=True)
            # plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            # slice out plot area
            x_plot = self.uc_dics["f2"].ppm(
                self.X[self.min_y : self.max_y, self.min_x : self.max_x]
            )
            y_plot = self.uc_dics["f1"].ppm(
                self.Y[self.min_y : self.max_y, self.min_x : self.max_x]
            )
            z_plot = self.Z[self.min_y : self.max_y, self.min_x : self.max_x]

            z_sim = self.Z_sim[self.min_y : self.max_y, self.min_x : self.max_x]

            ax.set_title(
                "$\chi^2$="
                + f"{self.out.chisqr:.3f}, "
                + "$\chi_{red}^2$="
                + f"{self.out.redchi:.4f}"
            )

            residual = z_plot - z_sim
            cset = ax.contourf(
                x_plot,
                y_plot,
                residual,
                zdir="z",
                offset=np.nanmin(z_plot) * 1.1,
                alpha=0.5,
                cmap=cm.coolwarm,
            )
            fig.colorbar(cset, ax=ax, shrink=0.5, format="%.2e")
            # plot raw data
            ax.plot_wireframe(x_plot, y_plot, z_plot, color="#03353E", label="data")

            ax.set_xlabel("F2 ppm")
            ax.set_ylabel("F1 ppm")
            ax.plot_wireframe(
                x_plot, y_plot, z_sim, color="#C1403D", linestyle="--", label="fit"
            )

            # axes will appear inverted
            ax.view_init(30, 120)

            # Annotate plots
            labs = []
            Z_lab = []
            Y_lab = []
            X_lab = []
            for k, v in self.out.params.valuesdict().items():
                if "amplitude" in k:
                    Z_lab.append(v)
                    # get prefix
                    labs.append(" ".join(k.split("_")[:-1]))
                elif "center_x" in k:
                    X_lab.append(self.uc_dics["f2"].ppm(v))
                elif "center_y" in k:
                    Y_lab.append(self.uc_dics["f1"].ppm(v))
            #  this is dumb as !£$@
            Z_lab = [
                self.Z[
                    int(round(self.uc_dics["f1"](y, "ppm"))),
                    int(round(self.uc_dics["f2"](x, "ppm"))),
                ]
                for x, y in zip(X_lab, Y_lab)
            ]

            for l, x, y, z in zip(labs, X_lab, Y_lab, Z_lab):
                # print(l, x, y, z)
                ax.text(x, y, z * 1.2, l, None)

            # plt.colorbar(contf)
            plt.legend(bbox_to_anchor=(1.2, 1.1))

            name = self.group.CLUSTID.iloc[0]
            if show and nomp:
                plt.savefig(plot_path / f"{name}.png", dpi=300)

                def exit_program(event):
                    exit()

                def next_plot(event):
                    plt.close()

                axexit = plt.axes([0.81, 0.05, 0.1, 0.075])
                bnexit = Button(axexit, "Exit")
                bnexit.on_clicked(exit_program)

                axnext = plt.axes([0.71, 0.05, 0.1, 0.075])
                bnnext = Button(axnext, "Next")
                bnnext.on_clicked(next_plot)

                plt.show()
            else:
                print(
                    "Cannot use interactive matplotlib in multiprocess mode. Use --nomp flag."
                )
                plt.savefig(plot_path / f"{name}.png", dpi=300)
            #    print(p_guess)
            # close plot
            plt.close()
        else:
            pass

class JackKnifeResult:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

class Pseudo3D:
    """Read dic, data from NMRGlue and dims from input to create a Pseudo3D dataset

       :param dic: from nmrglue.pipe.read
       :type dic: dict

       :param data: data from nmrglue.pipe.read
       :type data: numpy.array

       :param dims: dimension order i.e [0,1,2] where 0 = planes, 1 = f1, 2 = f2
       :type dims: list
       """

    def __init__(self, dic, data, dims):
        # check dimensions
        self._udic = ng.pipe.guess_udic(dic, data)
        self._ndim = self._udic["ndim"]

        if self._ndim == 1:
            err = f"""
            ##########################################
                NMR Data should be either 2D or 3D
            ##########################################
            """
            raise TypeError(err)

        # check that spectrum has correct number of dims
        elif self._ndim != len(dims):
            err = f"""
            #################################################################
               Your spectrum has {self._ndim} dimensions with shape {data.shape}
               but you have given a dimension order of {dims}...
            #################################################################
            """
            raise ValueError(err)

        elif (self._ndim == 2) and (len(dims) == 2):
            self._f1_dim, self._f2_dim = dims
            self._planes = 0
            self._uc_f1 = ng.pipe.make_uc(dic, data, dim=self._f1_dim)
            self._uc_f2 = ng.pipe.make_uc(dic, data, dim=self._f2_dim)
            # make data pseudo3d
            self._data = data.reshape((1, data.shape[0], data.shape[1]))
            self._dims = [self._planes, self._f1_dim + 1, self._f2_dim + 1]

        else:
            self._planes, self._f1_dim, self._f2_dim = dims
            self._dims = dims
            self._data = data
            # make unit conversion dicts
            self._uc_f2 = ng.pipe.make_uc(dic, data, dim=self._f2_dim)
            self._uc_f1 = ng.pipe.make_uc(dic, data, dim=self._f1_dim)

        #  rearrange data if dims not in standard order
        if self._dims != [0, 1, 2]:
            # np.argsort returns indices of array for order 0,1,2 to transpose data correctly
            # self._dims = np.argsort(self._dims)
            self._data = np.transpose(data, self._dims)

        self._dic = dic

        self._f1_label = self._udic[self._f1_dim]["label"]
        self._f2_label = self._udic[self._f2_dim]["label"]

    @property
    def uc_f1(self):
        """ Return unit conversion dict for F1"""
        return self._uc_f1

    @property
    def uc_f2(self):
        """ Return unit conversion dict for F2"""
        return self._uc_f2

    @property
    def dims(self):
        """ Return dimension order """
        return self._dims

    @property
    def data(self):
        """ Return array containing data """
        return self._data

    @property
    def dic(self):
        return self._dic

    @property
    def udic(self):
        return self._udic

    @property
    def ndim(self):
        return self._ndim

    @property
    def f1_label(self):
        # dim label
        return self._f1_label

    @property
    def f2_label(self):
        # dim label
        return self._f2_label

    # size of f1 and f2 in points
    @property
    def f2_size(self):
        """ Return size of f2 dimension in points """
        return self._udic[self._f2_dim]["size"]

    @property
    def f1_size(self):
        """ Return size of f1 dimension in points """
        return self._udic[self._f1_dim]["size"]

    # points per ppm
    @property
    def pt_per_ppm_f1(self):
        return self.f1_size / (
            self._udic[self._f1_dim]["sw"] / self._udic[self._f1_dim]["obs"]
        )

    @property
    def pt_per_ppm_f2(self):
        return self.f2_size / (
            self._udic[self._f2_dim]["sw"] / self._udic[self._f2_dim]["obs"]
        )

    # points per hz
    @property
    def pt_per_hz_f1(self):
        return self.f1_size / self._udic[self._f1_dim]["sw"]

    @property
    def pt_per_hz_f2(self):
        return self.f2_size / self._udic[self._f2_dim]["sw"]

    # ppm per point
    @property
    def ppm_per_pt_f1(self):
        return 1.0 / self.pt_per_ppm_f1

    @property
    def ppm_per_pt_f2(self):
        return 1.0 / self.pt_per_ppm_f2

    # get ppm limits for ppm scales


#    uc_f1 = ng.pipe.make_uc(dic, data, dim=f1)
#    ppm_f1 = uc_f1.ppm_scale()
#    ppm_f1_0, ppm_f1_1 = uc_f1.ppm_limits()
#
#    uc_f2 = ng.pipe.make_uc(dic, data, dim=f2)
#    ppm_f2 = uc_f2.ppm_scale()
#    ppm_f2_0, ppm_f2_1 = uc_f2.ppm_limits()
