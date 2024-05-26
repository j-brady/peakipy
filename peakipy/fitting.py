from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy import sqrt
import pandas as pd
from lmfit import Model
from pydantic import BaseModel

from peakipy.lineshapes import Lineshape, pvoigt2d, pv_pv, pv_g, pv_l, voigt2d
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
    Negative thresholds just result in return of the orignal data.

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
