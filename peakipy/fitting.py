import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from numpy import sqrt
import pandas as pd
from rich import print
from lmfit import Model, Parameters, Parameter
from lmfit.model import ModelResult
from pydantic import BaseModel

from peakipy.lineshapes import (
    Lineshape,
    pvoigt2d,
    pv_pv,
    pv_g,
    pv_l,
    voigt2d,
    gaussian_lorentzian,
    get_lineshape_function,
)
from peakipy.constants import log2


class FitDataModel(BaseModel):
    plane: int
    clustid: int
    assignment: str
    memcnt: int
    amp: float
    height: float
    center_x_ppm: float
    center_y_ppm: float
    fwhm_x_hz: float
    fwhm_y_hz: float
    lineshape: str
    x_radius: float
    y_radius: float
    center_x: float
    center_y: float
    sigma_x: float
    sigma_y: float


class FitDataModelPVGL(FitDataModel):
    fraction: float


class FitDataModelVoigt(FitDataModel):
    fraction: float
    gamma_x: float
    gamma_y: float


class FitDataModelPVPV(FitDataModel):
    fraction_x: float
    fraction_y: float


def validate_fit_data(dict):
    lineshape = dict.get("lineshape")
    if lineshape in ["PV", "G", "L"]:
        fit_data = FitDataModelPVGL(**dict)
    elif lineshape == "V":
        fit_data = FitDataModelVoigt(**dict)
    else:
        fit_data = FitDataModelPVPV(**dict)

    return fit_data.model_dump()


def validate_fit_dataframe(df):
    validated_fit_data = []
    for _, row in df.iterrows():
        fit_data = validate_fit_data(row.to_dict())
        validated_fit_data.append(fit_data)
    return pd.DataFrame(validated_fit_data)


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
    """Given a peak position and linewidth in points determine
    the limits based on the data

    Arguments
    ---------
    peak: pd.DataFrame
        peak is a row from a pandas dataframe
    data: np.array
        2D numpy array
    """

    peak: pd.DataFrame
    data: np.array
    min_x: int = field(init=False)
    max_x: int = field(init=False)
    min_y: int = field(init=False)
    max_y: int = field(init=False)

    def __post_init__(self):
        assert self.peak.Y_AXIS <= self.data.shape[0]
        assert self.peak.X_AXIS <= self.data.shape[1]
        self.max_y = int(np.ceil(self.peak.Y_AXIS + self.peak.YW)) + 1
        if self.max_y > self.data.shape[0]:
            self.max_y = self.data.shape[0]
        self.max_x = int(np.ceil(self.peak.X_AXIS + self.peak.XW)) + 1
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

    # Replace any remaining disallowed characters with underscore
    prefix = re.sub(r"[^a-z0-9_]", "_", prefix)
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
    Negative thresholds just result in return of the original data.

    """
    if threshold == None:
        selected_data = data
    else:
        selected_data = data[np.abs(data).max(axis=1) > threshold]

    if selected_data.shape[0] == 0:
        selected_data = data

    return selected_data


def validate_plane_selection(plane, pseudo3D):
    if (plane == []) or (plane == None):
        plane = list(range(pseudo3D.n_planes))

    elif max(plane) > (pseudo3D.n_planes - 1):
        raise ValueError(
            f"[red]There are {pseudo3D.n_planes} planes in your data you selected --plane {max(plane)}...[red]"
            f"plane numbering starts from 0."
        )
    elif min(plane) < 0:
        raise ValueError(
            f"[red]Plane number can not be negative; you selected --plane {min(plane)}...[/red]"
        )
    else:
        plane = sorted(plane)

    return plane


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


def unpack_xy_bounds(xy_bounds, peakipy_data):
    match xy_bounds:
        case (0, 0):
            xy_bounds = None
        case (x, y):
            # convert ppm to points
            xy_bounds = list(xy_bounds)
            xy_bounds[0] = xy_bounds[0] * peakipy_data.pt_per_ppm_f2
            xy_bounds[1] = xy_bounds[1] * peakipy_data.pt_per_ppm_f1
        case _:
            raise TypeError(
                "xy_bounds should be a tuple (<x_bounds_ppm>, <y_bounds_ppm>)"
            )
    return xy_bounds


def select_specified_planes(plane, peakipy_data):
    plane_numbers = np.arange(peakipy_data.data.shape[peakipy_data.dims[0]])
    # only fit specified planes
    if plane:
        inds = [i for i in plane]
        data_inds = [
            (i in inds) for i in range(peakipy_data.data.shape[peakipy_data.dims[0]])
        ]
        plane_numbers = np.arange(peakipy_data.data.shape[peakipy_data.dims[0]])[
            data_inds
        ]
        peakipy_data.data = peakipy_data.data[data_inds]
        print(
            "[yellow]Using only planes {plane} data now has the following shape[/yellow]",
            peakipy_data.data.shape,
        )
        if peakipy_data.data.shape[peakipy_data.dims[0]] == 0:
            print("[red]You have excluded all the data![/red]", peakipy_data.data.shape)
            exit()
    return plane_numbers, peakipy_data


def exclude_specified_planes(exclude_plane, peakipy_data):
    plane_numbers = np.arange(peakipy_data.data.shape[peakipy_data.dims[0]])
    # do not fit these planes
    if exclude_plane:
        inds = [i for i in exclude_plane]
        data_inds = [
            (i not in inds)
            for i in range(peakipy_data.data.shape[peakipy_data.dims[0]])
        ]
        plane_numbers = np.arange(peakipy_data.data.shape[peakipy_data.dims[0]])[
            data_inds
        ]
        peakipy_data.data = peakipy_data.data[data_inds]
        print(
            f"[yellow]Excluding planes {exclude_plane} data now has the following shape[/yellow]",
            peakipy_data.data.shape,
        )
        if peakipy_data.data.shape[peakipy_data.dims[0]] == 0:
            print("[red]You have excluded all the data![/red]", peakipy_data.data.shape)
            exit()
    return plane_numbers, peakipy_data


def get_fit_data_for_selected_peak_clusters(fits, clusters):
    match clusters:
        case None | []:
            pass
        case _:
            # only use these clusters
            fits = fits[fits.clustid.isin(clusters)]
            if len(fits) < 1:
                exit(f"Are you sure clusters {clusters} exist?")
    return fits


def make_masks_from_plane_data(empty_mask_array, plane_data):
    # make masks
    individual_masks = []
    for cx, cy, rx, ry, name in zip(
        plane_data.center_x,
        plane_data.center_y,
        plane_data.x_radius,
        plane_data.y_radius,
        plane_data.assignment,
    ):
        tmp_mask = make_mask(empty_mask_array, cx, cy, rx, ry)
        empty_mask_array += tmp_mask
        individual_masks.append(tmp_mask)
    filled_mask_array = empty_mask_array
    return individual_masks, filled_mask_array


def simulate_pv_pv_lineshapes_from_fitted_peak_parameters(
    peak_parameters, XY, sim_data, sim_data_singles
):
    for amp, c_x, c_y, s_x, s_y, frac_x, frac_y, ls in zip(
        peak_parameters.amp,
        peak_parameters.center_x,
        peak_parameters.center_y,
        peak_parameters.sigma_x,
        peak_parameters.sigma_y,
        peak_parameters.fraction_x,
        peak_parameters.fraction_y,
        peak_parameters.lineshape,
    ):
        sim_data_i = pv_pv(XY, amp, c_x, c_y, s_x, s_y, frac_x, frac_y).reshape(
            sim_data.shape
        )
        sim_data += sim_data_i
        sim_data_singles.append(sim_data_i)
    return sim_data, sim_data_singles


def simulate_lineshapes_from_fitted_peak_parameters(
    peak_parameters, XY, sim_data, sim_data_singles
):
    shape = sim_data.shape
    for amp, c_x, c_y, s_x, s_y, frac, lineshape in zip(
        peak_parameters.amp,
        peak_parameters.center_x,
        peak_parameters.center_y,
        peak_parameters.sigma_x,
        peak_parameters.sigma_y,
        peak_parameters.fraction,
        peak_parameters.lineshape,
    ):
        # print(amp)
        match lineshape:
            case "G" | "L" | "PV":
                sim_data_i = pvoigt2d(XY, amp, c_x, c_y, s_x, s_y, frac).reshape(shape)
            case "PV_L":
                sim_data_i = pv_l(XY, amp, c_x, c_y, s_x, s_y, frac).reshape(shape)

            case "PV_G":
                sim_data_i = pv_g(XY, amp, c_x, c_y, s_x, s_y, frac).reshape(shape)

            case "G_L":
                sim_data_i = gaussian_lorentzian(
                    XY, amp, c_x, c_y, s_x, s_y, frac
                ).reshape(shape)

            case "V":
                sim_data_i = voigt2d(XY, amp, c_x, c_y, s_x, s_y, frac).reshape(shape)
        sim_data += sim_data_i
        sim_data_singles.append(sim_data_i)
    return sim_data, sim_data_singles


@dataclass
class FitPeaksArgs:
    noise: float
    uc_dics: dict
    lineshape: Lineshape
    dims: List[int] = field(default_factory=lambda: [0, 1, 2])
    colors: Tuple[str] = ("#5e3c99", "#e66101")
    max_cluster_size: Optional[int] = None
    to_fix: List[str] = field(default_factory=lambda: ["fraction", "sigma", "center"])
    xy_bounds: Tuple[float, float] = ((0, 0),)
    vclist: Optional[Path] = (None,)
    plane: Optional[List[int]] = (None,)
    exclude_plane: Optional[List[int]] = (None,)
    reference_plane_indices: List[int] = ([],)
    initial_fit_threshold: Optional[float] = (None,)
    jack_knife_sample_errors: bool = False
    mp: bool = (True,)
    verbose: bool = (False,)
    vclist_data: Optional[np.array] = None


@dataclass
class Config:
    fit_method: str = "leastsq"


@dataclass
class FitPeaksInput:
    """input data for the fit_peaks function"""

    args: FitPeaksArgs
    data: np.array
    config: Config
    plane_numbers: list


@dataclass
class FitPeakClusterInput:
    args: FitPeaksArgs
    data: np.array
    config: Config
    plane_numbers: list
    clustid: int
    group: pd.DataFrame
    last_peak: pd.DataFrame
    mask: np.array
    mod: Model
    p_guess: Parameters
    XY: np.array
    peak_slices: np.array
    XY_slices: np.array
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    uc_dics: dict
    first_plane_data: np.array
    weights: np.array
    fit_method: str = "leastsq"
    verbose: bool = False
    masked_plane_data: np.array = field(init=False)

    def __post_init__(self):
        self.masked_plane_data = np.array([d[self.mask] for d in self.data])


@dataclass
class FitResult:
    out: ModelResult
    mask: np.array
    fit_str: str
    log: str
    group: pd.core.groupby.generic.DataFrameGroupBy
    uc_dics: dict
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    X: np.array
    Y: np.array
    Z: np.array
    Z_sim: np.array
    peak_slices: np.array
    XY_slices: np.array
    weights: np.array
    mod: Model

    def check_shifts(self):
        """Calculate difference between initial peak positions
        and check whether they moved too much from original
        position

        """
        pass


@dataclass
class FitPeaksResult:
    df: pd.DataFrame
    log: str


class FitPeaksResultDfRow(BaseModel):
    fit_prefix: str
    assignment: str
    amp: float
    amp_err: float
    center_x: float
    init_center_x: float
    center_y: float
    init_center_y: float
    sigma_x: float
    sigma_y: float
    clustid: int
    memcnt: int
    plane: int
    x_radius: float
    y_radius: float
    x_radius_ppm: float
    y_radius_ppm: float
    lineshape: str
    aic: float
    chisqr: float
    redchi: float
    residual_sum: float
    height: float
    height_err: float
    fwhm_x: float
    fwhm_y: float
    center_x_ppm: float
    center_y_ppm: float
    init_center_x_ppm: float
    init_center_y_ppm: float
    sigma_x_ppm: float
    sigma_y_ppm: float
    fwhm_x_ppm: float
    fwhm_y_ppm: float
    fwhm_x_hz: float
    fwhm_y_hz: float
    jack_knife_sample_index: Optional[int]


class FitPeaksResultRowGLPV(FitPeaksResultDfRow):
    fraction: float


class FitPeaksResultRowPVPV(FitPeaksResultDfRow):
    fraction_x: float  # for PV_PV model
    fraction_y: float  # for PV_PV model


class FitPeaksResultRowVoigt(FitPeaksResultDfRow):
    gamma_x_ppm: float  # for voigt
    gamma_y_ppm: float  # for voigt


def get_fit_peaks_result_validation_model(lineshape):
    """
    Retrieve the appropriate validation model based on the lineshape used for fitting.
    
    Parameters
    ----------
    lineshape : Lineshape
        Enum or string indicating the type of lineshape model used for fitting.
    
    Returns
    -------
    type
        The validation model class corresponding to the specified lineshape.
    """
    match lineshape:
        case lineshape.V:
            validation_model = FitPeaksResultRowVoigt
        case lineshape.PV_PV:
            validation_model = FitPeaksResultRowPVPV
        case _:
            validation_model = FitPeaksResultRowGLPV
    return validation_model


def filter_peak_clusters_by_max_cluster_size(grouped_peak_clusters, max_cluster_size):
    filtered_peak_clusters = grouped_peak_clusters.filter(
        lambda x: len(x) <= max_cluster_size
    )
    return filtered_peak_clusters


def set_parameters_to_fix_during_fit(first_plane_fit_params, to_fix):
    # fix sigma center and fraction parameters
    # could add an option to select params to fix
    match to_fix:
        case None | () | []:
            float_str = "Floating all parameters"
            parameter_set = first_plane_fit_params
        case ["None"] | ["none"]:
            float_str = "Floating all parameters"
            parameter_set = first_plane_fit_params
        case _:
            float_str = f"Fixing parameters: {to_fix}"
            parameter_set = fix_params(first_plane_fit_params, to_fix)
    return parameter_set, float_str


def get_default_lineshape_param_names(lineshape: Lineshape):
    match lineshape:
        case Lineshape.PV | Lineshape.G | Lineshape.L:
            param_names = Model(pvoigt2d).param_names
        case Lineshape.V:
            param_names = Model(voigt2d).param_names
        case Lineshape.PV_PV:
            param_names = Model(pv_pv).param_names
    return param_names


def split_parameter_sets_by_peak(
    default_param_names: List, params: List[Tuple[str, Parameter]]
):
    """params is a list of tuples where the first element of each tuple is a
    prefixed parameter name and the second element is the corresponding
    Parameter object. This is created by calling .items() on a Parameters
    object
    """
    number_of_fitted_parameters = len(params)
    number_of_default_params = len(default_param_names)
    number_of_fitted_peaks = int(number_of_fitted_parameters / number_of_default_params)
    split_param_items = [
        params[i : (i + number_of_default_params)]
        for i in range(0, number_of_fitted_parameters, number_of_default_params)
    ]
    assert len(split_param_items) == number_of_fitted_peaks
    return split_param_items


def create_parameter_dict(prefix, parameters: List[Tuple[str, Parameter]]):
    parameter_dict = dict(prefix=prefix)
    parameter_dict.update({k.replace(prefix, ""): v.value for k, v in parameters})
    parameter_dict.update(
        {f"{k.replace(prefix,'')}_stderr": v.stderr for k, v in parameters}
    )
    return parameter_dict


def get_prefix_from_parameter_names(
    default_param_names: List, parameters: List[Tuple[str, Parameter]]
):
    prefixes = [
        param_key_val[0].replace(default_param_name, "")
        for param_key_val, default_param_name in zip(parameters, default_param_names)
    ]
    assert len(set(prefixes)) == 1
    return prefixes[0]


def unpack_fitted_parameters_for_lineshape(
    lineshape: Lineshape, params: List[dict], plane_number: int
):
    default_param_names = get_default_lineshape_param_names(lineshape)
    split_parameter_names = split_parameter_sets_by_peak(default_param_names, params)
    prefixes = [
        get_prefix_from_parameter_names(default_param_names, i)
        for i in split_parameter_names
    ]
    unpacked_params = []
    for parameter_names, prefix in zip(split_parameter_names, prefixes):
        parameter_dict = create_parameter_dict(prefix, parameter_names)
        parameter_dict.update({"plane": plane_number})
        unpacked_params.append(parameter_dict)
    return unpacked_params


def perform_initial_lineshape_fit_on_cluster_of_peaks(
    fit_peak_cluster_input: FitPeakClusterInput,
) -> FitResult:
    mod = fit_peak_cluster_input.mod
    peak_slices = fit_peak_cluster_input.peak_slices
    XY_slices = fit_peak_cluster_input.XY_slices
    p_guess = fit_peak_cluster_input.p_guess
    weights = fit_peak_cluster_input.weights
    fit_method = fit_peak_cluster_input.fit_method
    mask = fit_peak_cluster_input.mask
    XY = fit_peak_cluster_input.XY
    X, Y = XY
    first_plane_data = fit_peak_cluster_input.first_plane_data
    peak = fit_peak_cluster_input.last_peak
    group = fit_peak_cluster_input.group
    min_x = fit_peak_cluster_input.min_x
    min_y = fit_peak_cluster_input.min_y
    max_x = fit_peak_cluster_input.max_x
    max_y = fit_peak_cluster_input.max_y
    verbose = fit_peak_cluster_input.verbose
    uc_dics = fit_peak_cluster_input.uc_dics

    out = mod.fit(
        peak_slices, XY=XY_slices, params=p_guess, weights=weights, method=fit_method
    )

    if verbose:
        print(out.fit_report())

    z_sim = mod.eval(XY=XY, params=out.params)
    z_sim[~mask] = np.nan
    z_plot = first_plane_data.copy()
    z_plot[~mask] = np.nan
    fit_str = ""
    log = ""

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


def refit_peak_cluster_with_constraints(
    fit_input: FitPeakClusterInput, fit_result: FitPeaksResult
):
    fit_results = []
    for num, d in enumerate(fit_input.masked_plane_data):
        plane_number = fit_input.plane_numbers[num]
        fit_result.out.fit(
            data=d,
            params=fit_result.out.params,
            weights=fit_result.weights,
        )
        fit_results.extend(
            unpack_fitted_parameters_for_lineshape(
                fit_input.args.lineshape,
                list(fit_result.out.params.items()),
                plane_number,
            )
        )
    return fit_results


def merge_unpacked_parameters_with_metadata(cluster_fit_df, group_of_peaks_df):
    """
    Combine fitted peak parameters with their associated metadata.
    
    Parameters
    ----------
    cluster_fit_df : pd.DataFrame
        DataFrame containing peak fitting results.
    group_of_peaks_df : pd.DataFrame
        DataFrame with metadata for corresponding peaks.
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with both fitting results and metadata.
    """
    group_of_peaks_df["prefix"] = group_of_peaks_df.ASS.apply(to_prefix)
    merged_cluster_fit_df = cluster_fit_df.merge(
        group_of_peaks_df, on="prefix", suffixes=["", "_init"]
    )
    return merged_cluster_fit_df


def update_cluster_df_with_fit_statistics(cluster_df, fit_result: ModelResult):
    cluster_df["chisqr"] = fit_result.chisqr
    cluster_df["redchi"] = fit_result.redchi
    cluster_df["residual_sum"] = np.sum(fit_result.residual)
    cluster_df["aic"] = fit_result.aic
    cluster_df["bic"] = fit_result.bic
    cluster_df["nfev"] = fit_result.nfev
    cluster_df["ndata"] = fit_result.ndata
    return cluster_df


def rename_columns_for_compatibility(df):
    mapping = {
        "amplitude": "amp",
        "amplitude_stderr": "amp_err",
        "X_AXIS": "init_center_x",
        "Y_AXIS": "init_center_y",
        "ASS": "assignment",
        "MEMCNT": "memcnt",
        "X_RADIUS": "x_radius",
        "Y_RADIUS": "y_radius",
    }
    df = df.rename(columns=mapping)
    return df


def add_vclist_to_df(fit_input: FitPeaksInput, df: pd.DataFrame):
    vclist_data = fit_input.args.vclist_data
    df["vclist"] = df.plane.apply(lambda x: vclist_data[x])
    return df


def prepare_group_of_peaks_for_fitting(clustid, group, fit_peaks_input: FitPeaksInput):
    lineshape_function = get_lineshape_function(fit_peaks_input.args.lineshape)

    first_plane_data = fit_peaks_input.data[0]
    mask, peak = make_mask_from_peak_cluster(group, first_plane_data)

    x_radius = group.X_RADIUS.max()
    y_radius = group.Y_RADIUS.max()

    max_x, min_x = get_limits_for_axis_in_points(
        group_axis_points=group.X_AXISf, mask_radius_in_points=x_radius
    )
    max_y, min_y = get_limits_for_axis_in_points(
        group_axis_points=group.Y_AXISf, mask_radius_in_points=y_radius
    )
    max_x, min_x, max_y, min_y = deal_with_peaks_on_edge_of_spectrum(
        fit_peaks_input.data.shape, max_x, min_x, max_y, min_y
    )
    selected_data = select_reference_planes_using_indices(
        fit_peaks_input.data, fit_peaks_input.args.reference_plane_indices
    ).sum(axis=0)
    mod, p_guess = make_models(
        lineshape_function,
        group,
        selected_data,
        lineshape=fit_peaks_input.args.lineshape,
        xy_bounds=fit_peaks_input.args.xy_bounds,
    )
    peak_slices = slice_peaks_from_data_using_mask(fit_peaks_input.data, mask)
    peak_slices = select_reference_planes_using_indices(
        peak_slices, fit_peaks_input.args.reference_plane_indices
    )
    peak_slices = select_planes_above_threshold_from_masked_data(
        peak_slices, fit_peaks_input.args.initial_fit_threshold
    )
    peak_slices = peak_slices.sum(axis=0)

    XY = make_meshgrid(fit_peaks_input.data.shape)
    X, Y = XY

    XY_slices = np.array([X.copy()[mask], Y.copy()[mask]])
    weights = 1.0 / np.array([fit_peaks_input.args.noise] * len(np.ravel(peak_slices)))
    return FitPeakClusterInput(
        args=fit_peaks_input.args,
        data=fit_peaks_input.data,
        config=fit_peaks_input.config,
        plane_numbers=fit_peaks_input.plane_numbers,
        clustid=clustid,
        group=group,
        last_peak=peak,
        mask=mask,
        mod=mod,
        p_guess=p_guess,
        XY=XY,
        peak_slices=peak_slices,
        XY_slices=XY_slices,
        weights=weights,
        fit_method=Config.fit_method,
        first_plane_data=first_plane_data,
        uc_dics=fit_peaks_input.args.uc_dics,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        verbose=fit_peaks_input.args.verbose,
    )


def fit_cluster_of_peaks(data_for_fitting: FitPeakClusterInput) -> pd.DataFrame:
    fit_result = perform_initial_lineshape_fit_on_cluster_of_peaks(data_for_fitting)
    fit_result.out.params, float_str = set_parameters_to_fix_during_fit(
        fit_result.out.params, data_for_fitting.args.to_fix
    )
    fit_results = refit_peak_cluster_with_constraints(data_for_fitting, fit_result)
    cluster_df = pd.DataFrame(fit_results)
    cluster_df = update_cluster_df_with_fit_statistics(cluster_df, fit_result.out)
    cluster_df["clustid"] = data_for_fitting.clustid
    cluster_df = merge_unpacked_parameters_with_metadata(
        cluster_df, data_for_fitting.group
    )
    return cluster_df


def fit_peak_clusters(peaks: pd.DataFrame, fit_input: FitPeaksInput) -> FitPeaksResult:
    """Fit set of peak clusters to lineshape model

    :param peaks: peaklist with generated by peakipy read or edit
    :type peaks: pd.DataFrame

    :param fit_input: Data structure containing input parameters (args, config and NMR data)
    :type fit_input: FitPeaksInput

    :returns: Data structure containing pd.DataFrame with the fitted results and a log
    :rtype: FitPeaksResult
    """
    peak_clusters = peaks.groupby("CLUSTID")
    filtered_peaks = filter_peak_clusters_by_max_cluster_size(
        peak_clusters, fit_input.args.max_cluster_size
    )
    peak_clusters = filtered_peaks.groupby("CLUSTID")
    out_str = ""
    cluster_dfs = []
    for clustid, peak_cluster in peak_clusters:
        data_for_fitting = prepare_group_of_peaks_for_fitting(
            clustid,
            peak_cluster,
            fit_input,
        )
        if fit_input.args.jack_knife_sample_errors:
            cluster_df = jack_knife_sample_errors(data_for_fitting)
        else:
            cluster_df = fit_cluster_of_peaks(data_for_fitting)
        cluster_dfs.append(cluster_df)
    df = pd.concat(cluster_dfs, ignore_index=True)

    df["lineshape"] = fit_input.args.lineshape.value

    if fit_input.args.vclist:
        df = add_vclist_to_df(fit_input, df)
    df = rename_columns_for_compatibility(df)
    return FitPeaksResult(df=df, log=out_str)


def jack_knife_sample_errors(fit_input: FitPeakClusterInput) -> pd.DataFrame:
    peak_slices = fit_input.peak_slices.copy()
    XY_slices = fit_input.XY_slices.copy()
    weights = fit_input.weights.copy()
    masked_plane_data = fit_input.masked_plane_data.copy()
    jk_results = []
    # first fit without jackknife
    jk_result = fit_cluster_of_peaks(data_for_fitting=fit_input)
    jk_result["jack_knife_sample_index"] = 0
    jk_results.append(jk_result)
    for i in np.arange(0, len(peak_slices), 10, dtype=int):
        fit_input.peak_slices = np.delete(peak_slices, i, None)
        XY_slices_0 = np.delete(XY_slices[0], i, None)
        XY_slices_1 = np.delete(XY_slices[1], i, None)
        fit_input.XY_slices = np.array([XY_slices_0, XY_slices_1])
        fit_input.weights = np.delete(weights, i, None)
        fit_input.masked_plane_data = np.delete(masked_plane_data, i, axis=1)
        jk_result = fit_cluster_of_peaks(data_for_fitting=fit_input)
        jk_result["jack_knife_sample_index"] = i + 1
        jk_results.append(jk_result)
    return pd.concat(jk_results, ignore_index=True)
