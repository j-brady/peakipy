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

from numpy import sqrt, log, pi, exp
from lmfit import Model, report_fit
from lmfit.models import LinearModel
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button


# constants
log2 = log(2)
s2pi = sqrt(2 * pi)
spi = sqrt(pi)

π = pi
# √π = sqrt(π)
# √2π =  sqrt(2*π)

s2 = sqrt(2.0)
tiny = 1.0e-13


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

        Arguments:
            -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
            -- amplitude: peak amplitude (gaussian and lorentzian)
            -- center_x: position of peak in x
            -- center_y: position of peak in y
            -- sigma_x: linewidth in x
            -- sigma_y: linewidth in y
            -- fraction: fraction of lorenztian in fit

        Returns:
            -- flattened array of Z values (use Z.reshape(X.shape) for recovery)

    """

    def gaussian(x, center=0.0, sigma=1.0):
        """Return a 1-dimensional Gaussian function.
        gaussian(x, center, sigma) =
            (1/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))
        """
        return (1.0 / (sqrt(2 * π) * sigma)) * exp(
            -(1.0 * x - center) ** 2 / (2 * sigma ** 2)
        )

    def lorentzian(x, center=0.0, sigma=1.0):
        """Return a 1-dimensional Lorentzian function.
        lorentzian(x, center, sigma) =
            (1/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)
        """
        return (1.0 / (1 + ((1.0 * x - center) / sigma) ** 2)) / (π * sigma)

    x, y = XY
    sigma_gx = sigma_x / sqrt(2 * log2)
    sigma_gy = sigma_y / sqrt(2 * log2)
    # fraction same for both dimensions
    # super position of gaussian and lorentzian
    # then convoluted for x y
    pv_x = (1 - fraction) * gaussian(x, center_x, sigma_gx) + fraction * lorentzian(
        x, center_x, sigma_x
    )
    pv_y = (1 - fraction) * gaussian(y, center_y, sigma_gy) + fraction * lorentzian(
        y, center_y, sigma_y
    )
    return amplitude * pv_x * pv_y


def make_mask(data, c_x, c_y, r_x, r_y):
    """ Create and elliptical mask

        Description:
            Generate an elliptical boolean mask with center c_x/c_y in points
            with radii r_x and r_y. Used to generate fit mask

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
    y, x = np.ogrid[-a : n_y - a, -b : n_x - b]
    mask = x ** 2.0 / r_x ** 2.0 + y ** 2.0 / r_y ** 2.0 <= 1.0
    return mask


def rmsd(residuals):
    return np.sqrt(np.sum(residuals ** 2.0) / len(residuals))


def fix_params(params, to_fix):
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
        else:
            param_dict[str_form("fraction")] = 0.5

    return param_dict


def to_prefix(x):
    """
    Peak assignments with characters that are not compatible lmfit model naming
    are converted to lmfit "safe" names.

    Arguments:
        -- x: Peak assignment to be used as prefix for lmfit model

    Returns:
        -- prefix: _Peak_assignment_

    """
    prefix = "_" + x
    to_replace = [
        [" ", ""],
        ["{", "_"],
        ["}", "_"],
        ["[", "_"],
        ["]", "_"],
        ["-", ""],
        ["/", "or"],
        ["?", "maybe"],
        ["\\", ""],
    ]
    for p in to_replace:
        prefix = prefix.replace(*p)
    return prefix + "_"


def make_models(model, peaks, data, lineshape="PV", xy_bounds=None):
    """ Make composite models for multiple peaks

        Arguments:
            -- model
            -- peaks: instance of pandas.df.groupby("CLUSTID")
            -- data: NMR data
            -- lineshape: PV/G/L
            -- xy_bounds: tuple containing bounds for peak centers (+/-x, +/-y)

        Returns:
            -- mod: Composite lmfit model containing all peaks
            -- p_guess: params for composite model with starting values

        Maybe add mask making to this function
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

        Arguments:
             -- params: lmfit parameter object
             -- peaks: list of Peak objects that parameters correspond to

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
                #pass
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
    with open(log_name,'a') as log:
        sys_argv = sys.argv
        sys_argv[0] = Path(sys_argv[0]).name
        run_args = " ".join(sys_argv)
        time_stamp = datetime.now()
        time_stamp = time_stamp.strftime("%A %d %B %Y at %H:%M")
        log.write(f"# Script run on {time_stamp}:\n{run_args}\n")


def fit_first_plane(
    group, data, uc_dics, lineshape="PV", xy_bounds=None, plot=None, show=True, verbose=False, log=None, noise=1.,
):
    """
        Arguments:

            group -- pandas data from containing group of peaks using groupby("CLUSTID")
            data  -- NMR data
            uc_dics -- unit conversion dics
            lineshape -- PV/G/L
            xy_bounds -- None or (x_bound, y_bound)
            plot -- if True show wireframe plots

    """
    shape = data.shape
    mask = np.zeros(shape, dtype=bool)
    mod, p_guess = make_models(pvoigt2d, group, data, lineshape=lineshape, xy_bounds=xy_bounds)

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
        int(np.floor(min(group.X_AXISf) - x_radius )),
    )
    max_y, min_y = (
        int(np.ceil(max(group.Y_AXISf) + y_radius + 1)),
        int(np.floor(min(group.Y_AXISf) - y_radius )),
    )

    # deal with peaks on the edge of spectrum
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

    XY_slices = [X.copy()[mask], Y.copy()[mask]]
    out = mod.fit(peak_slices, XY=XY_slices, params=p_guess)
    if verbose:
        print(out.fit_report())

    # calculate chi2
    z_sim = mod.eval(XY=XY, params=out.params)
    z_sim[~mask] = np.nan
    z_plot = data.copy()
    z_plot[~mask] = np.nan
    #print(z_plot.shape,z_sim.shape)
    # calculate difference between fitted height 
    # also if peak position changed significantly from start then add warning
    # figure out tolerence 
    _z_plot = z_plot[~np.isnan(z_plot)]
    _z_sim = z_sim[~np.isnan(z_sim)]
    _z_plot_min = np.min(_z_plot)
    _z_plot_max = np.max(_z_plot)

    norm_z = (_z_plot - _z_plot_min) / (_z_plot_max - _z_plot_min)
    norm_sim = (_z_sim - _z_plot_min) / (_z_plot_max - _z_plot_min)
    chi2 = np.sum((norm_z - norm_sim) ** 2.0 / np.abs(norm_sim))
    #_norm_z = norm_z[~np.isnan(norm_z)]
    #_norm_sim = norm_sim[~np.isnan(norm_sim)]

    linmod = LinearModel()
    linpars = linmod.guess(_z_sim,x=_z_plot)
    linfit = linmod.fit(_z_sim,x=_z_plot,params=linpars)
    #_sigma = np.sqrt(np.sum((_z_plot-_z_sim)**2.)/len(_z_plot))
    #plt.plot(_z_plot,linfit.best_fit,"--",label=f"{linfit.fit_report()}:Sigma={_sigma}")
    #plt.scatter(_z_plot,_z_sim,marker="o")
    #plt.xlabel("z_plot")
    #plt.ylabel("z_sim")
    #plt.legend()
    #plt.savefig(f"test/{peak.CLUSTID}.pdf")
    #plt.close()
    #chi2 = np.sum(np.abs(_z_sim - _z_plot)/ np.abs(_z_plot)) / len(_z_plot)
    #chi2 = np.sqrt(np.sum((_z_plot - _z_sim) ** 2.0) / np.sum(_z_sim ** 2.0 ))# / len(_z_plot)
    #chi2 = (np.sum((_z_plot - _z_sim) ** 2.0 / _z_sim**2)/len(_z_sim))**0.5
    slope = linfit.params["slope"].value
    #  number of peaks in cluster
    n_peaks = len(group)
    fit_str = f"Cluster {peak.CLUSTID} containing {n_peaks} peaks - slope={slope:.3f}" 
    if (slope > 1.05) or (slope < 0.95): 
        fit_str += " - NEEDS CHECKING"
        print(fit_str)
    else:
        print(fit_str)
    
    #for index, peak in group.iterrows():

    #    init_prefix = peak.prefix
    #    init_cenx = peak.X_AXISf
    #    init_ceny = peak.Y_AXISf
    #    print(init_prefix, init_cenx, init_ceny)

        #if out.
    chi2 = chi2 / n_peaks
    chi_str = f"Cluster {peak.CLUSTID} containing {n_peaks} peaks - chi2={chi2:.3f}"
    #if chi2 < 1:
    #    print(chi_str)
    #print(f"mean_err = {mean_err:.3f}, std_err = {std_err:.3f}, mean-std = {mean_err-std_err:.3f} ")
    #else:
    #    chi_str += " - NEEDS CHECKING"
    #    print(chi_str)
        #print(f"mean_err = {mean_err:.3f}, std_err = {std_err:.3f} ")

    if log != None:
        log.write("".join("#" for _ in range(60))+"\n\n")
        log.write(fit_str + "\n")
        log.write(chi_str + "\n\n")
        #pass
    else:
        pass

    if plot != None:
        plot_path = Path(plot)

        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # slice out plot area
        x_plot = uc_dics["f2"].ppm(X[min_y:max_y, min_x:max_x])
        y_plot = uc_dics["f1"].ppm(Y[min_y:max_y, min_x:max_x])
        z_plot = z_plot[min_y:max_y, min_x:max_x]
        z_sim = z_sim[min_y:max_y, min_x:max_x]

        ax.set_title("$\chi^2$=" + f"{chi2:.3f}")

        # plot raw data
        ax.plot_wireframe(x_plot, y_plot, z_plot, color="k")

        ax.set_xlabel("F2 ppm")
        ax.set_ylabel("F1 ppm")
        ax.plot_wireframe(
            x_plot, y_plot, z_sim, colors="r", linestyle="--", label="fit"
        )
        ax.invert_xaxis()
        ax.invert_yaxis()
        # Annotate plots
        labs = []
        Z_lab = []
        Y_lab = []
        X_lab = []
        for k, v in out.params.valuesdict().items():
            if "amplitude" in k:
                Z_lab.append(v)
                # get prefix
                labs.append(" ".join(k.split("_")[:-1]))
            elif "center_x" in k:
                X_lab.append(uc_dics["f2"].ppm(v))
            elif "center_y" in k:
                Y_lab.append(uc_dics["f1"].ppm(v))
        #  this is dumb as !£$@
        Z_lab = [
            data[
                int(round(uc_dics["f1"](y, "ppm"))), int(round(uc_dics["f2"](x, "ppm")))
            ]
            for x, y in zip(X_lab, Y_lab)
        ]

        for l, x, y, z in zip(labs, X_lab, Y_lab, Z_lab):
            # print(l, x, y, z)
            ax.text(x, y, z * 1.4, l, None)

        # plt.colorbar(contf)
        plt.legend()

        name = group.CLUSTID.iloc[0]
        if show:
            plt.savefig(plot_path / f"{name}.png", dpi=300)

            def exit_program(event):
                exit()

            axexit = plt.axes([0.81, 0.05, 0.1, 0.075])
            bnexit = Button(axexit, "Exit")
            bnexit.on_clicked(exit_program)
            plt.show()
        else:
            plt.savefig(plot_path / f"{name}.png", dpi=300)
        #    print(p_guess)
        # close plot
        plt.close()
    return out, mask


class Pseudo3D:
    """ Read dic, data from NMRGlue and dims from input to create a
        Pseudo3D dataset

        Arguments:
            dic  -- dic from nmrglue.pipe.read
            data -- data from nmrglue.pipe.read
            dims -- dimension order i.e [0,1,2]
                    0 = planes, 1 = f1, 2 = f2

        Methods:


    """

    def __init__(self, dic, data, dims):
        # check dimensions
        self._udic = ng.pipe.guess_udic(dic, data)
        self._ndim = self._udic["ndim"]

        if self._ndim == 1:
            raise TypeError("NMR Data should be either 2D or 3D")

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

class Fit:
    """ Class for fitting planes: NOT CURRENTLY USED """
    def __init__(
        self,
        group,
        data,
        udics,
        model=pvoigt2d,
        lineshape="PV",
        plot=None,
        show=True,
        verbose=False,
        log=None,
    ):

        """
            Arguments:

                group -- pandas data from containing group of peaks using groupby("CLUSTID")
                data  -- NMR data cube
                uc_dics -- unit conversion dics
                lineshape -- PV/G/L
                plot -- if True show wireframe plots
                show -- whether or not to show the plot using plt.show()
                verbose -- whether or not to print results
                log -- filehandle for log file
        """
        self.group = group
        self.data = data
        self.udics = udics
        self.model = model
        self.lineshape = lineshape
        self.log = log
        self.show = show
        self.plot = plot

    def first_plane(self):

        """ Fit first plane """

        summed_planes = self.data.sum(axis=0)
        # create boolean mask
        self.mask = np.zeros(self.data.shape, dtype=bool)
        # make models
        self.mod, self.p_guess = make_models(self.model, self.group, self.data, lineshape=self.lineshape)

        ## get initial peak centers
        #self.cen_x = [self.p_guess[k].value for k in self.p_guess if "center_x" in k]
        #self.cen_y = [self.p_guess[k].value for k in self.p_guess if "center_y" in k]

        for index, peak in self.group.iterrows():
            # generate boolean mask based on peak locations and radii
            self.mask += make_mask(
                self.data, peak.X_AXISf, peak.Y_AXISf, peak.X_RADIUS, peak.Y_RADIUS
            )

        # needs checking since this may not center peaks
        x_radius = self.group.X_RADIUS.max()
        y_radius = self.group.Y_RADIUS.max()
        self.max_x, self.min_x = (
            int(np.ceil(max(self.group.X_AXISf) + x_radius + 1)),
            int(np.floor(min(self.group.X_AXISf) - x_radius )),
        )
        self.max_y, self.min_y = (
            int(np.ceil(max(self.group.Y_AXISf) + y_radius + 1)),
            int(np.floor(min(self.group.Y_AXISf) - y_radius )),
        )
        #self.max_x, self.min_x = (
        #    int(np.ceil(max(self.group.X_AXISf) + x_radius + 2)),
        #    int(np.floor(min(self.group.X_AXISf) - x_radius - 1)),
        #)
        #self.max_y, self.min_y = (
        #    int(np.ceil(max(self.group.Y_AXISf) + y_radius + 2)),
        #    int(np.floor(min(self.group.Y_AXISf) - y_radius - 1)),
        #)

        peak_slices = self.data[mask]

        # must be a better way to make the meshgrid
        # starts from 1
        #x = np.arange(1, data.shape[-1] + 1)
        #y = np.arange(1, data.shape[-2] + 1)
        x = np.arange(data.shape[-1])
        y = np.arange(data.shape[-2])
        self.xy_grid = np.meshgrid(x, y)
        self.x_grid, self.y_grid = self.xy_grid

        # mask mesh data
        xy_slices = [x_grid[mask], y_grid[mask]]
        # fit data
        self.out = self.mod.fit(peak_slices, XY=xy_slices, params=self.p_guess)
        if verbose:
            print(self.out.fit_report())

    def chi_squared(self):

        # calculate chi2
        z_sim = self.mod.eval(XY=self.xy_grid, params=self.out.params)
        z_sim[~mask] = np.nan
        z_plot = data.copy()
        z_plot[~mask] = np.nan

        norm_z = (z_plot - np.nanmin(z_plot)) / (np.nanmax(z_plot) - np.nanmin(z_plot))
        norm_sim = (z_sim - np.nanmin(z_plot)) / (np.nanmax(z_plot) - np.nanmin(z_plot))
        _norm_z = norm_z[~np.isnan(norm_z)]
        _norm_sim = norm_sim[~np.isnan(norm_sim)]
        chi2 = np.sum((_norm_z - _norm_sim) ** 2.0 / _norm_sim)

        #  number of peaks in cluster
        n_peaks = len(group)
        chi2 = chi2 / n_peaks
        if chi2 < 1:
            chi_str = f"Cluster {peak.CLUSTID} containing {n_peaks} peaks - chi2={chi2:.3f}"
            print(f"Cluster {peak.CLUSTID} containing {n_peaks} peaks - chi2={chi2:.3f}")
        else:
            chi_str = f"Cluster {peak.CLUSTID} containing {n_peaks} peaks - chi2={chi2:.3f} - NEEDS CHECKING"
            print(
                f"Cluster {peak.CLUSTID} containing {n_peaks} peaks - chi2={chi2:.3f} - NEEDS CHECKING"
            )

        if log != None:
            log.write(chi_str + "\n")
        else:
            pass

    def plot_fit(self):

        if plot != None:
            plot_path = Path(plot)

            # plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            # slice out plot area
            x_plot = uc_dics["f2"].ppm(X[min_y:max_y, min_x:max_x])
            y_plot = uc_dics["f1"].ppm(Y[min_y:max_y, min_x:max_x])
            z_plot = z_plot[min_y:max_y, min_x:max_x]
            z_sim = z_sim[min_y:max_y, min_x:max_x]

            ax.set_title("$\chi^2$=" + f"{chi2:.3f}")

            # plot raw data
            ax.plot_wireframe(x_plot, y_plot, z_plot, color="k")

            ax.set_xlabel("F2 ppm")
            ax.set_ylabel("F1 ppm")
            ax.plot_wireframe(
                x_plot, y_plot, z_sim, colors="r", linestyle="--", label="fit"
            )
            ax.invert_xaxis()
            ax.invert_yaxis()
            # Annotate plots
            labs = []
            Z_lab = []
            Y_lab = []
            X_lab = []
            for k, v in out.params.valuesdict().items():
                if "amplitude" in k:
                    Z_lab.append(v)
                    # get prefix
                    labs.append(" ".join(k.split("_")[:-1]))
                elif "center_x" in k:
                    X_lab.append(uc_dics["f2"].ppm(v))
                elif "center_y" in k:
                    Y_lab.append(uc_dics["f1"].ppm(v))
            #  this is dumb as !£$@
            Z_lab = [
                data[
                    int(round(uc_dics["f1"](y, "ppm"))), int(round(uc_dics["f2"](x, "ppm")))
                ]
                for x, y in zip(X_lab, Y_lab)
            ]

            for l, x, y, z in zip(labs, X_lab, Y_lab, Z_lab):
                # print(l, x, y, z)
                ax.text(x, y, z * 1.4, l, None)

            # plt.colorbar(contf)
            plt.legend()

            name = group.CLUSTID.iloc[0]
            if show:
                plt.savefig(plot_path / f"{name}.png", dpi=300)

                def exit_program(event):
                    exit()

                axexit = plt.axes([0.81, 0.05, 0.1, 0.075])
                bnexit = Button(axexit, "Exit")
                bnexit.on_clicked(exit_program)
                plt.show()
            else:
                plt.savefig(plot_path / f"{name}.png", dpi=300)
            #    print(p_guess)
            # close plot
            plt.close()
        return out, mask
