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
import json
from datetime import datetime
from pathlib import Path
from typing import List
from enum import Enum
from dataclasses import dataclass, field

import numpy as np
import nmrglue as ng
import pandas as pd
import textwrap
from rich import print
from rich.table import Table
from rich.console import Console

from numpy import sqrt, log, pi, exp, finfo

from lmfit import Model
from scipy.special import wofz

from bokeh.palettes import Category20
from scipy import ndimage
from skimage.morphology import square, binary_closing, disk, rectangle
from skimage.filters import threshold_otsu

console = Console()
# constants
log2 = log(2)
π = pi
tiny = finfo(float).eps


class StrucEl(str, Enum):
    square = "square"
    disk = "disk"
    rectangle = "rectangle"
    mask_method = "mask_method"


class PeaklistFormat(str, Enum):
    a2 = "a2"
    a3 = "a3"
    sparky = "sparky"
    pipe = "pipe"
    peakipy = "peakipy"


class OutFmt(str, Enum):
    csv = "csv"
    pkl = "pkl"


class Lineshape(str, Enum):
    PV = "PV"
    V = "V"
    G = "G"
    L = "L"
    PV_PV = "PV_PV"
    G_L = "G_L"
    PV_G = "PV_G"
    PV_L = "PV_L"


def gaussian(x, center=0.0, sigma=1.0):
    r"""1-dimensional Gaussian function.

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
    return (1.0 / max(tiny, (sqrt(2 * π) * sigma))) * exp(
        -((1.0 * x - center) ** 2) / max(tiny, (2 * sigma**2))
    )


def lorentzian(x, center=0.0, sigma=1.0):
    r"""1-dimensional Lorentzian function.

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
    return (1.0 / (1 + ((1.0 * x - center) / max(tiny, sigma)) ** 2)) / max(
        tiny, (π * sigma)
    )


def voigt(x, center=0.0, sigma=1.0, gamma=None):
    r"""Return a 1-dimensional Voigt function.

    voigt(x, center, sigma, gamma) =
        amplitude*wofz(z).real / (sigma*sqrt(2.0 * π))

    :math:`V(x,\sigma,\gamma) = (\\frac{Re[\omega(z)]}{\sigma \sqrt{2\pi}})`

    :math:`z=\\frac{x+i\gamma}{\sigma\sqrt{2}}`

    see Voigt_ wiki

    .. _Voigt: https://en.wikipedia.org/wiki/Voigt_profile


    :param x: x values
    :type x: numpy array 1d
    :param center: center of lineshape in points
    :type center: float
    :param sigma: sigma of gaussian
    :type sigma: float
    :param gamma: gamma of lorentzian
    :type gamma: float

    :returns: Voigt lineshape
    :rtype: numpy.array

    """
    if gamma is None:
        gamma = sigma

    z = (x - center + 1j * gamma) / max(tiny, (sigma * sqrt(2.0)))
    return wofz(z).real / max(tiny, (sigma * sqrt(2.0 * π)))


def pseudo_voigt(x, center=0.0, sigma=1.0, fraction=0.5):
    r"""1-dimensional Pseudo-voigt function

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


def pvoigt2d(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    r"""2D pseudo-voigt model

    :math:`(1-fraction) G(x,center,\sigma_{gx}) + (fraction) L(x, center, \sigma_x) * (1-fraction) G(y,center,\sigma_{gy}) + (fraction) L(y, center, \sigma_y)`

    :param XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
    :type XY: numpy.array

    :param amplitude: amplitude of peak
    :type amplitude: float

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
    pv_x = pseudo_voigt(x, center_x, sigma_x, fraction)
    pv_y = pseudo_voigt(y, center_y, sigma_y, fraction)
    return amplitude * pv_x * pv_y


def pv_l(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """2D lineshape model with pseudo-voigt in x and lorentzian in y

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


def pv_g(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """2D lineshape model with pseudo-voigt in x and gaussian in y

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
    """2D lineshape model with pseudo-voigt in x and pseudo-voigt in y
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


def gaussian_lorentzian(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """2D lineshape model with gaussian in x and lorentzian in y

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


def voigt2d(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    gamma_x=1.0,
    gamma_y=1.0,
    fraction=0.5,
):
    fraction = 0.5
    gamma_x = None
    gamma_y = None
    x, y = XY
    voigt_x = voigt(x, center_x, sigma_x, gamma_x)
    voigt_y = voigt(y, center_y, sigma_y, gamma_y)
    return amplitude * voigt_x * voigt_y


def make_mask(data, c_x, c_y, r_x, r_y):
    """Create and elliptical mask

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
    mask = x**2.0 / r_x**2.0 + y**2.0 / r_y**2.0 <= 1.0
    return mask


def rmsd(residuals):
    return np.sqrt(np.sum(residuals**2.0) / len(residuals))


def fix_params(params, to_fix):
    """Set parameters to fix


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
    prefixes = []
    for k in params:
        if name in k:
            ps.append(params[k].value)
            ps_err.append(params[k].stderr)
            names.append(k)
            prefixes.append(k.split(name)[0])
    return ps, ps_err, names, prefixes


@dataclass
class PeakLimits:
    peak: pd.DataFrame
    data: np.array
    min_x: int = field(init=False)
    max_x: int = field(init=False)
    min_y: int = field(init=False)
    max_y: int = field(init=False)

    def __post_init__(self):
        self.max_y = int(self.peak.Y_AXIS + self.peak.YW) + 1
        if self.max_y > self.data.shape[0]:
            self.max_y = self.data.shape[0]
        self.max_x = int(self.peak.X_AXIS + self.peak.XW) + 1
        if self.max_x > self.data.shape[1]:
            self.max_x = self.data.shape[1]

        self.min_y = int(self.peak.Y_AXIS - self.peak.YW)
        if self.min_y < 0:
            self.min_y = 0
        self.min_x = int(self.peak.X_AXIS - self.peak.XW)
        if self.min_x < 0:
            self.min_x = 0


def estimate_amplitude(peak, data):
    assert len(data.shape) == 2
    limits = PeakLimits(peak, data)
    amplitude_est = data[limits.min_y : limits.max_y, limits.min_x : limits.max_x].sum()
    return amplitude_est


def make_param_dict(peaks, data, lineshape: Lineshape = Lineshape.PV):
    """Make dict of parameter names using prefix"""

    param_dict = {}

    for _, peak in peaks.iterrows():
        str_form = lambda x: "%s%s" % (to_prefix(peak.ASS), x)
        # using exact value of points (i.e decimal)
        param_dict[str_form("center_x")] = peak.X_AXISf
        param_dict[str_form("center_y")] = peak.Y_AXISf
        # estimate peak volume
        amplitude_est = estimate_amplitude(peak, data)
        param_dict[str_form("amplitude")] = amplitude_est
        # sigma linewidth esimate
        param_dict[str_form("sigma_x")] = peak.XW / 2.0
        param_dict[str_form("sigma_y")] = peak.YW / 2.0

        match lineshape:
            case lineshape.V:
                #  Voigt G sigma from linewidth esimate
                param_dict[str_form("sigma_x")] = peak.XW / (
                    2.0 * sqrt(2.0 * log2)
                )  # 3.6013
                param_dict[str_form("sigma_y")] = peak.YW / (
                    2.0 * sqrt(2.0 * log2)
                )  # 3.6013
                #  Voigt L gamma from linewidth esimate
                param_dict[str_form("gamma_x")] = peak.XW / 2.0
                param_dict[str_form("gamma_y")] = peak.YW / 2.0
                # height
                # add height here

            case lineshape.G:
                param_dict[str_form("fraction")] = 0.0
            case lineshape.L:
                param_dict[str_form("fraction")] = 1.0
            case lineshape.PV_PV:
                param_dict[str_form("fraction_x")] = 0.5
                param_dict[str_form("fraction_y")] = 0.5
            case _:
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
        ["@", "_at_"],
    ]
    for p in to_replace:
        prefix = prefix.replace(*p)
    return prefix + "_"


def make_models(
    model,
    peaks,
    data,
    lineshape: Lineshape = Lineshape.PV,
    xy_bounds=None,
):
    """Make composite models for multiple peaks

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
        param_dict = make_param_dict(
            peaks,
            data,
            lineshape=lineshape,
        )
        p_guess = mod.make_params(**param_dict)

    elif len(peaks) > 1:
        # make model for first peak
        first_peak, *remaining_peaks = peaks.iterrows()
        mod = Model(model, prefix="%s" % to_prefix(first_peak[1].ASS))
        for _, peak in remaining_peaks:
            mod += Model(model, prefix="%s" % to_prefix(peak.ASS))

        param_dict = make_param_dict(
            peaks,
            data,
            lineshape=lineshape,
        )
        p_guess = mod.make_params(**param_dict)
        # add Peak params to p_guess

    update_params(p_guess, param_dict, lineshape=lineshape, xy_bounds=xy_bounds)

    return mod, p_guess


def update_params(
    params, param_dict, lineshape: Lineshape = Lineshape.PV, xy_bounds=None
):
    """Update lmfit parameters with values from Peak

    :param params: lmfit parameters
    :type params: lmfit.Parameters object
    :param param_dict: parameters corresponding to each peak in fit
    :type param_dict: dict
    :param lineshape: Lineshape (PV, G, L, PV_PV etc.)
    :type lineshape: Lineshape
    :param xy_bounds: bounds on xy peak positions
    :type xy_bounds: tuple

    :returns: None
    :rtype: None

    ToDo
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
                # print(
                #    "setting limit of %s, min = %.3e, max = %.3e"
                #    % (k, params[k].min, params[k].max)
                # )
        elif "sigma" in k:
            params[k].min = 0.0
            params[k].max = 1e4

        elif "gamma" in k:
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

            #  fix fraction of G or L
            match lineshape:
                case lineshape.G | lineshape.L:
                    params[k].vary = False
                case lineshape.PV | lineshape.PV_PV:
                    params[k].vary = True
                case _:
                    pass

    # return params


def run_log(log_name="run_log.txt"):
    """Write log file containing time script was run and with which arguments"""
    with open(log_name, "a") as log:
        sys_argv = sys.argv
        sys_argv[0] = Path(sys_argv[0]).name
        run_args = " ".join(sys_argv)
        time_stamp = datetime.now()
        time_stamp = time_stamp.strftime("%A %d %B %Y at %H:%M")
        log.write(f"# Script run on {time_stamp}:\n{run_args}\n")


def df_to_rich_table(df, title: str, columns: List[str], styles: str):
    """Print dataframe using rich library

    Parameters
    ----------
    df : pandas.DataFrame
    title : str
        title of table
    columns : List[str]
        list of column names (must be in df)
    styles : List[str]
        list of styles in same order as columns
    """
    table = Table(title=title)
    for col, style in zip(columns, styles):
        table.add_column(col, style=style)
    for _, row in df.iterrows():
        row = row[columns].values
        str_row = []
        for i in row:
            match i:
                case str():
                    str_row.append(f"{i}")
                case float() if i > 1e5:
                    str_row.append(f"{i:.1e}")
                case float():
                    str_row.append(f"{i:.3f}")
                case bool():
                    str_row.append(f"{i}")
                case int():
                    str_row.append(f"{i}")
        table.add_row(*str_row)
    return table


def make_mask_from_peak_cluster(group, data):
    mask = np.zeros(data.shape, dtype=bool)
    for _, peak in group.iterrows():
        mask += make_mask(
            data, peak.X_AXISf, peak.Y_AXISf, peak.X_RADIUS, peak.Y_RADIUS
        )
    return mask, peak


def select_reference_planes_using_indices(data, indices: List[int]):
    n_planes = data.shape[0]
    if indices == []:
        return data

    max_index = max(indices)
    min_index = min(indices)

    if max_index >= n_planes:
        raise IndexError(
            f"Your data has {n_planes}. You selected plane {max_index} (allowed indices between 0 and {n_planes-1})"
        )
    elif min_index < (-1 * n_planes):
        raise IndexError(
            f"Your data has {n_planes}. You selected plane {min_index} (allowed indices between -{n_planes} and {n_planes-1})"
        )
    else:
        data = data[indices]
    return data


def select_planes_above_threshold_from_masked_data(data, threshold=None):
    """This function returns planes with data above the threshold.

    It currently uses absolute intensity values.
    Negative thresholds just result in return of the orignal data.

    """
    if threshold == None:
        selected_data = data
    else:
        selected_data = data[np.abs(data).max(axis=1) > threshold]

    if selected_data.shape[0] == 0:
        selected_data = data

    return selected_data


def get_lineshape_function(lineshape: Lineshape):
    match lineshape:
        case lineshape.PV | lineshape.G | lineshape.L:
            lineshape_function = pvoigt2d
        case lineshape.V:
            lineshape_function = voigt2d
        case lineshape.PV_PV:
            lineshape_function = pv_pv
        case lineshape.G_L:
            lineshape_function = gaussian_lorentzian
        case lineshape.PV_G:
            lineshape_function = pv_g
        case lineshape.PV_L:
            lineshape_function = pv_l
        case _:
            raise Exception("No lineshape was selected!")
    return lineshape_function


def slice_peaks_from_data_using_mask(data, mask):
    peak_slices = np.array([d[mask] for d in data])
    return peak_slices


def get_limits_for_axis_in_points(group_axis_points, mask_radius_in_points):
    max_point, min_point = (
        int(np.ceil(max(group_axis_points) + mask_radius_in_points + 1)),
        int(np.floor(min(group_axis_points) - mask_radius_in_points)),
    )
    return max_point, min_point


def deal_with_peaks_on_edge_of_spectrum(data_shape, max_x, min_x, max_y, min_y):
    if min_y < 0:
        min_y = 0

    if min_x < 0:
        min_x = 0

    if max_y > data_shape[-2]:
        max_y = data_shape[-2]

    if max_x > data_shape[-1]:
        max_x = data_shape[-1]
    return max_x, min_x, max_y, min_y


def make_meshgrid(data_shape):
    # must be a better way to make the meshgrid
    x = np.arange(data_shape[-1])
    y = np.arange(data_shape[-2])
    XY = np.meshgrid(x, y)
    return XY


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
            err = f"""[red]
            ##########################################
                NMR Data should be either 2D or 3D
            ##########################################
            [/red]"""
            # raise TypeError(err)
            sys.exit(err)

        # check that spectrum has correct number of dims
        elif self._ndim != len(dims):
            err = f"""[red]
            #################################################################
               Your spectrum has {self._ndim} dimensions with shape {data.shape}
               but you have given a dimension order of {dims}...
            #################################################################
            [/red]"""
            # raise ValueError(err)
            sys.exit(err)

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
        """Return unit conversion dict for F1"""
        return self._uc_f1

    @property
    def uc_f2(self):
        """Return unit conversion dict for F2"""
        return self._uc_f2

    @property
    def dims(self):
        """Return dimension order"""
        return self._dims

    @property
    def data(self):
        """Return array containing data"""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

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

    @property
    def planes(self):
        return self.dims[0]

    @property
    def n_planes(self):
        return self.data.shape[self.planes]

    @property
    def f1(self):
        return self.dims[1]

    @property
    def f2(self):
        return self.dims[2]

    # size of f1 and f2 in points
    @property
    def f2_size(self):
        """Return size of f2 dimension in points"""
        return self._udic[self._f2_dim]["size"]

    @property
    def f1_size(self):
        """Return size of f1 dimension in points"""
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

    # hz per point
    @property
    def hz_per_pt_f1(self):
        return 1.0 / self.pt_per_hz_f1

    @property
    def hz_per_pt_f2(self):
        return 1.0 / self.pt_per_hz_f2

    # ppm per point
    @property
    def ppm_per_pt_f1(self):
        return 1.0 / self.pt_per_ppm_f1

    @property
    def ppm_per_pt_f2(self):
        return 1.0 / self.pt_per_ppm_f2

    # get ppm limits for ppm scales
    @property
    def f2_ppm_scale(self):
        return self.uc_f2.ppm_scale()

    @property
    def f1_ppm_scale(self):
        return self.uc_f1.ppm_scale()

    @property
    def f2_ppm_limits(self):
        return self.uc_f2.ppm_limits()

    @property
    def f1_ppm_limits(self):
        return self.uc_f1.ppm_limits()

    @property
    def f1_ppm_max(self):
        return max(self.f1_ppm_limits)

    @property
    def f1_ppm_min(self):
        return min(self.f1_ppm_limits)

    @property
    def f2_ppm_max(self):
        return max(self.f2_ppm_limits)

    @property
    def f2_ppm_min(self):
        return min(self.f2_ppm_limits)

    @property
    def f2_ppm_0(self):
        return self.f2_ppm_limits[0]

    @property
    def f2_ppm_1(self):
        return self.f2_ppm_limits[1]

    @property
    def f1_ppm_0(self):
        return self.f1_ppm_limits[0]

    @property
    def f1_ppm_1(self):
        return self.f1_ppm_limits[1]


class UnknownFormat(Exception):
    pass


class Peaklist(Pseudo3D):
    """Read analysis, sparky or NMRPipe peak list and convert to NMRPipe-ish format also find peak clusters

    Parameters
    ----------
    path : path-like or str
        path to peaklist
    data_path : ndarray
        NMRPipe format data
    fmt : str
        a2|a3|sparky|pipe
    dims: list
        [planes,y,x]
    radii: list
        [x,y] Mask radii in ppm


    Methods
    -------

    clusters :
    mask_method :
    adaptive_clusters :

    Returns
    -------
    df : pandas DataFrame
        dataframe containing peaklist

    """

    def __init__(
        self,
        path,
        data_path,
        fmt: PeaklistFormat = PeaklistFormat.a2,
        dims=[0, 1, 2],
        radii=[0.04, 0.4],
        posF1="Position F2",
        posF2="Position F1",
        verbose=False,
    ):
        dic, data = ng.pipe.read(data_path)
        Pseudo3D.__init__(self, dic, data, dims)
        self.fmt = fmt
        self.peaklist_path = path
        self.data_path = data_path
        self.verbose = verbose
        self._radii = radii
        self._thres = None
        if self.verbose:
            print(
                "Points per hz f1 = %.3f, f2 = %.3f"
                % (self.pt_per_hz_f1, self.pt_per_hz_f2)
            )

        self._analysis_to_pipe_dic = {
            "#": "INDEX",
            "Position F1": "X_PPM",
            "Position F2": "Y_PPM",
            "Line Width F1 (Hz)": "XW_HZ",
            "Line Width F2 (Hz)": "YW_HZ",
            "Height": "HEIGHT",
            "Volume": "VOL",
        }
        self._assign_to_pipe_dic = {
            "#": "INDEX",
            "Pos F1": "X_PPM",
            "Pos F2": "Y_PPM",
            "LW F1 (Hz)": "XW_HZ",
            "LW F2 (Hz)": "YW_HZ",
            "Height": "HEIGHT",
            "Volume": "VOL",
        }

        self._sparky_to_pipe_dic = {
            "index": "INDEX",
            "w1": "X_PPM",
            "w2": "Y_PPM",
            "lw1 (hz)": "XW_HZ",
            "lw2 (hz)": "YW_HZ",
            "Height": "HEIGHT",
            "Volume": "VOL",
            "Assignment": "ASS",
        }

        self._analysis_to_pipe_dic[posF1] = "Y_PPM"
        self._analysis_to_pipe_dic[posF2] = "X_PPM"

        self._df = self.read_peaklist()

    def read_peaklist(self):
        match self.fmt:
            case self.fmt.a2:
                self._df = self._read_analysis()

            case self.fmt.a3:
                self._df = self._read_assign()

            case self.fmt.sparky:
                self._df = self._read_sparky()

            case self.fmt.pipe:
                self._df = self._read_pipe()

            case _:
                raise UnknownFormat("I don't know this format: {self.fmt}")

        return self._df

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._df = df
        return self._df

    @property
    def radii(self):
        return self._radii

    @property
    def f2_radius(self):
        """radius for fitting mask in f2"""
        return self.radii[0]

    @property
    def f1_radius(self):
        """radius for fitting mask in f1"""
        return self.radii[1]

    @property
    def analysis_to_pipe_dic(self):
        return self._analysis_to_pipe_dic

    @property
    def assign_to_pipe_dic(self):
        return self._assign_to_pipe_dic

    @property
    def sparky_to_pipe_dic(self):
        return self._sparky_to_pipe_dic

    @property
    def thres(self):
        if self._thres == None:
            self._thres = abs(threshold_otsu(self.data[0]))
            return self._thres
        else:
            return self._thres

    def update_df(self):
        # int point value
        self.df["X_AXIS"] = self.df.X_PPM.apply(lambda x: self.uc_f2(x, "ppm"))
        self.df["Y_AXIS"] = self.df.Y_PPM.apply(lambda x: self.uc_f1(x, "ppm"))
        # decimal point value
        self.df["X_AXISf"] = self.df.X_PPM.apply(lambda x: self.uc_f2.f(x, "ppm"))
        self.df["Y_AXISf"] = self.df.Y_PPM.apply(lambda x: self.uc_f1.f(x, "ppm"))
        # in case of missing values (should estimate though)
        self.df["XW_HZ"] = self.df.XW_HZ.replace("None", "20.0")
        self.df["YW_HZ"] = self.df.YW_HZ.replace("None", "20.0")
        self.df["XW_HZ"] = self.df.XW_HZ.replace(np.NaN, "20.0")
        self.df["YW_HZ"] = self.df.YW_HZ.replace(np.NaN, "20.0")
        # convert linewidths to float
        self.df["XW_HZ"] = self.df.XW_HZ.apply(lambda x: float(x))
        self.df["YW_HZ"] = self.df.YW_HZ.apply(lambda x: float(x))
        # convert Hz lw to points
        self.df["XW"] = self.df.XW_HZ.apply(lambda x: x * self.pt_per_hz_f2)
        self.df["YW"] = self.df.YW_HZ.apply(lambda x: x * self.pt_per_hz_f1)
        # makes an assignment column from Assign F1 and Assign F2 columns
        # in analysis2.x and ccpnmr v3 assign peak lists
        if self.fmt in [PeaklistFormat.a2, PeaklistFormat.a3]:
            self.df["ASS"] = self.df.apply(
                # lambda i: "".join([i["Assign F1"], i["Assign F2"]]), axis=1
                lambda i: f"{i['Assign F1']}_{i['Assign F2']}",
                axis=1,
            )

        # make default values for X and Y radii for fit masks
        self.df["X_RADIUS_PPM"] = np.zeros(len(self.df)) + self.f2_radius
        self.df["Y_RADIUS_PPM"] = np.zeros(len(self.df)) + self.f1_radius
        self.df["X_RADIUS"] = self.df.X_RADIUS_PPM.apply(
            lambda x: x * self.pt_per_ppm_f2
        )
        self.df["Y_RADIUS"] = self.df.Y_RADIUS_PPM.apply(
            lambda x: x * self.pt_per_ppm_f1
        )
        # add include column
        if "include" in self.df.columns:
            pass
        else:
            self.df["include"] = self.df.apply(lambda x: "yes", axis=1)

        # check assignments for duplicates
        self.check_assignments()
        # check that peaks are within the bounds of the data
        self.check_peak_bounds()

    def add_fix_bound_columns(self):
        """add columns containing parameter bounds (param_upper/param_lower)
        and whether or not parameter should be fixed (yes/no)

        For parameter bounding:

            Column names are <param_name>_upper and <param_name>_lower for upper and lower bounds respectively.
            Values are given as floating point. Value of 0.0 indicates that parameter is unbounded
            X/Y positions are given in ppm
            Linewidths are given in Hz

        For parameter fixing:

            Column names are <param_name>_fix.
            Values are given as a string 'yes' or 'no'

        """
        pass

    def _read_analysis(self):
        df = pd.read_csv(self.peaklist_path, delimiter="\t")
        new_columns = [self.analysis_to_pipe_dic.get(i, i) for i in df.columns]
        pipe_columns = dict(zip(df.columns, new_columns))
        df = df.rename(index=str, columns=pipe_columns)

        return df

    def _read_assign(self):
        df = pd.read_csv(self.peaklist_path, delimiter="\t")
        new_columns = [self.assign_to_pipe_dic.get(i, i) for i in df.columns]
        pipe_columns = dict(zip(df.columns, new_columns))
        df = df.rename(index=str, columns=pipe_columns)

        return df

    def _read_sparky(self):
        df = pd.read_csv(
            self.peaklist_path,
            skiprows=1,
            sep=r"\s+",
            names=["ASS", "Y_PPM", "X_PPM", "VOLUME", "HEIGHT", "YW_HZ", "XW_HZ"],
        )
        df["INDEX"] = df.index

        return df

    def _read_pipe(self):
        to_skip = 0
        with open(self.peaklist_path) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("VARS"):
                    columns = line.strip().split()[1:]
                elif line[:5].strip(" ").isdigit():
                    break
                else:
                    to_skip += 1
        df = pd.read_csv(
            self.peaklist_path, skiprows=to_skip, names=columns, sep=r"\s+"
        )
        return df

    def check_assignments(self):
        # self.df["ASS"] = self.df.
        self.df["ASS"] = self.df.ASS.astype(object)
        self.df.loc[self.df["ASS"].isnull(), "ASS"] = "None_dummy_0"
        self.df["ASS"] = self.df.ASS.astype(str)
        duplicates_bool = self.df.ASS.duplicated()
        duplicates = self.df.ASS[duplicates_bool]
        if len(duplicates) > 0:
            console.print(
                textwrap.dedent(
                    """
                #############################################################################
                    You have duplicated assignments in your list...
                    Currently each peak needs a unique assignment. Sorry about that buddy...
                #############################################################################
                """
                ),
                style="yellow",
            )
            self.df.loc[duplicates_bool, "ASS"] = [
                f"{i}_dummy_{num+1}" for num, i in enumerate(duplicates)
            ]
            if self.verbose:
                print("Here are the duplicates")
                print(duplicates)
                print(self.df.ASS)

            print(
                textwrap.dedent(
                    """
                    Creating dummy assignments for duplicates

                """
                )
            )

    def check_peak_bounds(self):
        columns_to_print = ["INDEX", "ASS", "X_AXIS", "Y_AXIS", "X_PPM", "Y_PPM"]
        # check that peaks are within the bounds of spectrum
        within_x = (self.df.X_PPM < self.f2_ppm_max) & (self.df.X_PPM > self.f2_ppm_min)
        within_y = (self.df.Y_PPM < self.f1_ppm_max) & (self.df.Y_PPM > self.f1_ppm_min)
        self.excluded = self.df[~(within_x & within_y)]
        self.df = self.df[within_x & within_y]
        if len(self.excluded) > 0:
            print(
                textwrap.dedent(
                    f"""[red]
                    #################################################################################

                    Excluding the following peaks as they are not within the spectrum which has shape

                    {self.data.shape}
                [/red]"""
                )
            )
            table_to_print = df_to_rich_table(
                self.excluded,
                title="Excluded",
                columns=columns_to_print,
                styles=["red" for i in columns_to_print],
            )
            print(table_to_print)
            print(
                "[red]#################################################################################[/red]"
            )

    def clusters(
        self,
        thres=None,
        struc_el: StrucEl = StrucEl.disk,
        struc_size=(3,),
        l_struc=None,
    ):
        """Find clusters of peaks

        :param thres: threshold for positive signals above which clusters are selected. If None then threshold_otsu is used
        :type thres: float

        :param struc_el: 'square'|'disk'|'rectangle'
            structuring element for binary_closing of thresholded data can be square, disc or rectangle
        :type struc_el: str

        :param struc_size: size/dimensions of structuring element
            for square and disk first element of tuple is used (for disk value corresponds to radius)
            for rectangle, tuple corresponds to (width,height).
        :type struc_size: tuple


        """
        peaks = [[y, x] for y, x in zip(self.df.Y_AXIS, self.df.X_AXIS)]

        if thres == None:
            thres = self.thres
            self._thres = abs(threshold_otsu(self.data[0]))
        else:
            self._thres = thres

        # get positive and negative
        thresh_data = np.bitwise_or(
            self.data[0] < (self._thres * -1.0), self.data[0] > self._thres
        )

        match struc_el:
            case struc_el.disk:
                radius = struc_size[0]
                if self.verbose:
                    print(f"using disk with {radius}")
                closed_data = binary_closing(thresh_data, disk(int(radius)))

            case struc_el.square:
                width = struc_size[0]
                if self.verbose:
                    print(f"using square with {width}")
                closed_data = binary_closing(thresh_data, square(int(width)))

            case struc_el.rectangle:
                width, height = struc_size
                if self.verbose:
                    print(f"using rectangle with {width} and {height}")
                closed_data = binary_closing(
                    thresh_data, rectangle(int(width), int(height))
                )

            case _:
                if self.verbose:
                    print(f"Not using any closing function")
                closed_data = thresh_data

        labeled_array, num_features = ndimage.label(closed_data, l_struc)

        self.df.loc[:, "CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

        #  renumber "0" clusters
        max_clustid = self.df["CLUSTID"].max()
        n_of_zeros = len(self.df[self.df["CLUSTID"] == 0]["CLUSTID"])
        self.df.loc[self.df[self.df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
            max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
        )

        # count how many peaks per cluster
        for ind, group in self.df.groupby("CLUSTID"):
            self.df.loc[group.index, "MEMCNT"] = len(group)

        self.df.loc[:, "color"] = self.df.apply(
            lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black",
            axis=1,
        )
        return ClustersResult(labeled_array, num_features, closed_data, peaks)

    def mask_method(self, overlap=1.0, l_struc=None):
        """connect clusters based on overlap of fitting masks

        :param overlap: fraction of mask for which overlaps are calculated
        :type overlap: float

        :returns ClusterResult: Instance of ClusterResult
        :rtype: ClustersResult
        """
        # overlap is positive
        overlap = abs(overlap)

        self._thres = threshold_otsu(self.data[0])

        mask = np.zeros(self.data[0].shape, dtype=bool)

        for ind, peak in self.df.iterrows():
            mask += make_mask(
                self.data[0],
                peak.X_AXISf,
                peak.Y_AXISf,
                peak.X_RADIUS * overlap,
                peak.Y_RADIUS * overlap,
            )

        peaks = [[y, x] for y, x in zip(self.df.Y_AXIS, self.df.X_AXIS)]
        labeled_array, num_features = ndimage.label(mask, l_struc)

        self.df.loc[:, "CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

        #  renumber "0" clusters
        max_clustid = self.df["CLUSTID"].max()
        n_of_zeros = len(self.df[self.df["CLUSTID"] == 0]["CLUSTID"])
        self.df.loc[self.df[self.df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
            max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
        )

        # count how many peaks per cluster
        for ind, group in self.df.groupby("CLUSTID"):
            self.df.loc[group.index, "MEMCNT"] = len(group)

        self.df.loc[:, "color"] = self.df.apply(
            lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black",
            axis=1,
        )

        return ClustersResult(labeled_array, num_features, mask, peaks)

    def to_fuda(self, fname="params.fuda"):
        with open("peaks.fuda", "w") as peaks_fuda:
            for ass, f1_ppm, f2_ppm in zip(self.df.ASS, self.df.Y_PPM, self.df.X_PPM):
                peaks_fuda.write(f"{ass}\t{f1_ppm:.3f}\t{f2_ppm:.3f}\n")
        groups = self.df.groupby("CLUSTID")
        fuda_params = Path(fname)
        overlap_peaks = ""

        for ind, group in groups:
            if len(group) > 1:
                overlap_peaks_str = ";".join(group.ASS)
                overlap_peaks += f"OVERLAP_PEAKS=({overlap_peaks_str})\n"

        fuda_file = textwrap.dedent(
            f"""\

# Read peaklist and spectrum info
PEAKLIST=peaks.fuda
SPECFILE={self.data_path}
PARAMETERFILE=(bruker;vclist)
ZCORR=ncyc
NOISE={self.thres} # you'll need to adjust this
BASELINE=N
VERBOSELEVEL=5
PRINTDATA=Y
LM=(MAXFEV=250;TOL=1e-5)
#Specify the default values. All values are in ppm:
DEF_LINEWIDTH_F1={self.f1_radius}
DEF_LINEWIDTH_F2={self.f2_radius}
DEF_RADIUS_F1={self.f1_radius}
DEF_RADIUS_F2={self.f2_radius}
SHAPE=GLORE
# OVERLAP PEAKS
{overlap_peaks}"""
        )
        with open(fuda_params, "w") as f:
            print(f"Writing FuDA file {fuda_file}")
            f.write(fuda_file)
        if self.verbose:
            print(overlap_peaks)


class ClustersResult:
    """Class to store results of clusters function"""

    def __init__(self, labeled_array, num_features, closed_data, peaks):
        self._labeled_array = labeled_array
        self._num_features = num_features
        self._closed_data = closed_data
        self._peaks = peaks

    @property
    def labeled_array(self):
        return self._labeled_array

    @property
    def num_features(self):
        return self._num_features

    @property
    def closed_data(self):
        return self._closed_data

    @property
    def peaks(self):
        return self._peaks


class LoadData(Peaklist):
    """Load peaklist data from peakipy .csv file output from either peakipy read or edit

    read_peaklist is redefined to just read a .csv file

    check_data_frame makes sure data frame is in good shape for setting up fits

    """

    def read_peaklist(self):
        if self.peaklist_path.suffix == ".csv":
            self.df = pd.read_csv(self.peaklist_path)  # , comment="#")

        elif self.peaklist_path.suffix == ".tab":
            self.df = pd.read_csv(self.peaklist_path, sep="\t")  # comment="#")

        else:
            self.df = pd.read_pickle(self.peaklist_path)

        self._thres = threshold_otsu(self.data[0])

        return self.df

    def check_data_frame(self):
        # make diameter columns
        if "X_DIAMETER_PPM" in self.df.columns:
            pass
        else:
            self.df["X_DIAMETER_PPM"] = self.df["X_RADIUS_PPM"] * 2.0
            self.df["Y_DIAMETER_PPM"] = self.df["Y_RADIUS_PPM"] * 2.0

        #  make a column to track edited peaks
        if "Edited" in self.df.columns:
            pass
        else:
            self.df["Edited"] = np.zeros(len(self.df), dtype=bool)

        # create include column if it doesn't exist
        if "include" in self.df.columns:
            pass
        else:
            self.df["include"] = self.df.apply(lambda _: "yes", axis=1)

        # color clusters
        self.df["color"] = self.df.apply(
            lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black",
            axis=1,
        )

        # get rid of unnamed columns
        unnamed_cols = [i for i in self.df.columns if "Unnamed:" in i]
        self.df = self.df.drop(columns=unnamed_cols)

    def update_df(self):
        """Slightly modified to retain previous configurations"""
        # int point value
        self.df["X_AXIS"] = self.df.X_PPM.apply(lambda x: self.uc_f2(x, "ppm"))
        self.df["Y_AXIS"] = self.df.Y_PPM.apply(lambda x: self.uc_f1(x, "ppm"))
        # decimal point value
        self.df["X_AXISf"] = self.df.X_PPM.apply(lambda x: self.uc_f2.f(x, "ppm"))
        self.df["Y_AXISf"] = self.df.Y_PPM.apply(lambda x: self.uc_f1.f(x, "ppm"))
        # in case of missing values (should estimate though)
        self.df["XW_HZ"] = self.df.XW_HZ.replace(np.NaN, "20.0")
        self.df["YW_HZ"] = self.df.YW_HZ.replace(np.NaN, "20.0")
        # convert linewidths to float
        self.df["XW_HZ"] = self.df.XW_HZ.apply(lambda x: float(x))
        self.df["YW_HZ"] = self.df.YW_HZ.apply(lambda x: float(x))
        # convert Hz lw to points
        self.df["XW"] = self.df.XW_HZ.apply(lambda x: x * self.pt_per_hz_f2)
        self.df["YW"] = self.df.YW_HZ.apply(lambda x: x * self.pt_per_hz_f1)
        # makes an assignment column
        if self.fmt == "a2":
            self.df["ASS"] = self.df.apply(
                lambda i: "".join([i["Assign F1"], i["Assign F2"]]), axis=1
            )

        # make default values for X and Y radii for fit masks
        # self.df["X_RADIUS_PPM"] = np.zeros(len(self.df)) + self.f2_radius
        # self.df["Y_RADIUS_PPM"] = np.zeros(len(self.df)) + self.f1_radius
        self.df["X_RADIUS"] = self.df.X_RADIUS_PPM.apply(
            lambda x: x * self.pt_per_ppm_f2
        )
        self.df["Y_RADIUS"] = self.df.Y_RADIUS_PPM.apply(
            lambda x: x * self.pt_per_ppm_f1
        )
        # add include column
        if "include" in self.df.columns:
            pass
        else:
            self.df["include"] = self.df.apply(lambda x: "yes", axis=1)

        # check assignments for duplicates
        self.check_assignments()
        # check that peaks are within the bounds of the data
        self.check_peak_bounds()


def load_config(config_path):
    if config_path.exists():
        with open(config_path) as opened_config:
            config_dic = json.load(opened_config)
            return config_dic
    else:
        return {}


def write_config(config_path, config_dic):
    with open(config_path, "w") as config:
        config.write(json.dumps(config_dic, sort_keys=True, indent=4))


def update_config_file(config_path, config_kvs):
    config_dic = load_config(config_path)
    config_dic.update(config_kvs)
    write_config(config_path, config_dic)
    return config_dic


def update_args_with_values_from_config_file(args, config_path="peakipy.config"):
    """read a peakipy config file, extract params and update args dict

    :param args: dict containing params extracted from docopt command line
    :type args: dict
    :param config_path: path to peakipy config file [default: peakipy.config]
    :type config_path: str

    :returns args: updated args dict
    :rtype args: dict
    :returns config: dict that resulted from reading config file
    :rtype config: dict

    """
    # update args with values from peakipy.config file
    config_path = Path(config_path)
    if config_path.exists():
        try:
            config = load_config(config_path)
            print(
                f"[green]Using config file with dims [yellow]{config.get('dims')}[/yellow][/green]"
            )
            args["dims"] = config.get("dims", (0, 1, 2))
            noise = config.get("noise")
            if noise:
                noise = float(noise)

            colors = config.get("colors", ["#5e3c99", "#e66101"])
        except json.decoder.JSONDecodeError:
            print(
                "[red]Your peakipy.config file is corrupted - maybe your JSON is not correct...[/red]"
            )
            print("[red]Not using[/red]")
            noise = False
            colors = args.get("colors", "#5e3c99,#e66101").strip().split(",")
    else:
        print(
            "[red]No peakipy.config found - maybe you need to generate one with peakipy read or see docs[/red]"
        )
        noise = False
        colors = args.get("colors", "#5e3c99,#e66101").strip().split(",")
        config = {}

    args["noise"] = noise
    args["colors"] = colors

    return args, config


def calculate_height_for_voigt_lineshape(df):
    df["height"] = df.apply(
        lambda x: voigt2d(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            gamma_x=x.gamma_x,
            gamma_y=x.gamma_y,
            amplitude=x.amp,
        ),
        axis=1,
    )
    df["height_err"] = df.apply(
        lambda x: x.amp_err * (x.height / x.amp) if x.amp_err != None else 0.0,
        axis=1,
    )
    return df


def calculate_fwhm_for_voigt_lineshape(df):
    df["fwhm_g_x"] = df.sigma_x.apply(
        lambda x: 2.0 * x * np.sqrt(2.0 * np.log(2.0))
    )  # fwhm of gaussian
    df["fwhm_g_y"] = df.sigma_y.apply(lambda x: 2.0 * x * np.sqrt(2.0 * np.log(2.0)))
    df["fwhm_l_x"] = df.gamma_x.apply(lambda x: 2.0 * x)  # fwhm of lorentzian
    df["fwhm_l_y"] = df.gamma_y.apply(lambda x: 2.0 * x)
    df["fwhm_x"] = df.apply(
        lambda x: 0.5346 * x.fwhm_l_x
        + np.sqrt(0.2166 * x.fwhm_l_x**2.0 + x.fwhm_g_x**2.0),
        axis=1,
    )
    df["fwhm_y"] = df.apply(
        lambda x: 0.5346 * x.fwhm_l_y
        + np.sqrt(0.2166 * x.fwhm_l_y**2.0 + x.fwhm_g_y**2.0),
        axis=1,
    )
    return df


def calculate_height_for_pseudo_voigt_lineshape(df):
    df["height"] = df.apply(
        lambda x: pvoigt2d(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            amplitude=x.amp,
            fraction=x.fraction,
        ),
        axis=1,
    )
    df["height_err"] = df.apply(lambda x: x.amp_err * (x.height / x.amp), axis=1)
    return df


def calculate_fwhm_for_pseudo_voigt_lineshape(df):
    df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
    df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)
    return df


def calculate_height_for_gaussian_lineshape(df):
    df["height"] = df.apply(
        lambda x: pvoigt2d(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            amplitude=x.amp,
            fraction=0.0,  # gaussian
        ),
        axis=1,
    )
    df["height_err"] = df.apply(lambda x: x.amp_err * (x.height / x.amp), axis=1)
    return df


def calculate_height_for_lorentzian_lineshape(df):
    df["height"] = df.apply(
        lambda x: pvoigt2d(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            amplitude=x.amp,
            fraction=1.0,  # lorentzian
        ),
        axis=1,
    )
    df["height_err"] = df.apply(lambda x: x.amp_err * (x.height / x.amp), axis=1)
    return df


def calculate_height_for_pv_pv_lineshape(df):
    df["height"] = df.apply(
        lambda x: pv_pv(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            amplitude=x.amp,
            fraction_x=x.fraction_x,
            fraction_y=x.fraction_y,
        ),
        axis=1,
    )
    df["height_err"] = df.apply(lambda x: x.amp_err * (x.height / x.amp), axis=1)
    return df


def calculate_peak_centers_in_ppm(df, peakipy_data):
    #  convert values to ppm
    df["center_x_ppm"] = df.center_x.apply(lambda x: peakipy_data.uc_f2.ppm(x))
    df["center_y_ppm"] = df.center_y.apply(lambda x: peakipy_data.uc_f1.ppm(x))
    df["init_center_x_ppm"] = df.init_center_x.apply(
        lambda x: peakipy_data.uc_f2.ppm(x)
    )
    df["init_center_y_ppm"] = df.init_center_y.apply(
        lambda x: peakipy_data.uc_f1.ppm(x)
    )
    return df


def calculate_peak_linewidths_in_hz(df, peakipy_data):
    df["sigma_x_ppm"] = df.sigma_x.apply(lambda x: x * peakipy_data.ppm_per_pt_f2)
    df["sigma_y_ppm"] = df.sigma_y.apply(lambda x: x * peakipy_data.ppm_per_pt_f1)
    df["fwhm_x_ppm"] = df.fwhm_x.apply(lambda x: x * peakipy_data.ppm_per_pt_f2)
    df["fwhm_y_ppm"] = df.fwhm_y.apply(lambda x: x * peakipy_data.ppm_per_pt_f1)
    df["fwhm_x_hz"] = df.fwhm_x.apply(lambda x: x * peakipy_data.hz_per_pt_f2)
    df["fwhm_y_hz"] = df.fwhm_y.apply(lambda x: x * peakipy_data.hz_per_pt_f1)
    return df
