#!/usr/bin/env python3
"""Fit and deconvolute NMR peaks: Functions used for running peakipy fit
"""
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rich import print
from rich.console import Console
from pydantic import BaseModel
from lmfit import Model, Parameter, Parameters
from lmfit.model import ModelResult

from peakipy.core import (
    fix_params,
    Lineshape,
    pvoigt2d,
    voigt2d,
    pv_pv,
    to_prefix,
    get_limits_for_axis_in_points,
    get_lineshape_function,
    deal_with_peaks_on_edge_of_spectrum,
    select_planes_above_threshold_from_masked_data,
    select_reference_planes_using_indices,
    make_models,
    make_meshgrid,
    slice_peaks_from_data_using_mask,
    make_mask_from_peak_cluster,
)

console = Console()
# some constants
π = np.pi
sqrt2 = np.sqrt(2.0)
# temp and log paths
tmp_path = Path("tmp")
tmp_path.mkdir(exist_ok=True)
log_path = Path("log.txt")
# for printing dataframes
column_selection = ["INDEX", "ASS", "X_PPM", "Y_PPM", "CLUSTID", "MEMCNT"]


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
    mp: bool = (True,)
    verbose: bool = (False,)


@dataclass
class FirstPlaneFitInput:
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


@dataclass
class FitPeaksInput:
    """input data for the fit_peaks function"""

    args: FitPeaksArgs
    data: np.array
    config: dict
    plane_numbers: list


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


class FitPeaksResultRowGLPV(FitPeaksResultDfRow):
    fraction: float


class FitPeaksResultRowPVPV(FitPeaksResultDfRow):
    fraction_x: float  # for PV_PV model
    fraction_y: float  # for PV_PV model


class FitPeaksResultRowVoigt(FitPeaksResultDfRow):
    gamma_x_ppm: float  # for voigt
    gamma_y_ppm: float  # for voigt


def get_fit_peaks_result_validation_model(lineshape):
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
    first_plane_fit_input: FirstPlaneFitInput,
) -> FitResult:
    mod = first_plane_fit_input.mod
    peak_slices = first_plane_fit_input.peak_slices
    XY_slices = first_plane_fit_input.XY_slices
    p_guess = first_plane_fit_input.p_guess
    weights = first_plane_fit_input.weights
    fit_method = first_plane_fit_input.fit_method
    mask = first_plane_fit_input.mask
    XY = first_plane_fit_input.XY
    X, Y = XY
    first_plane_data = first_plane_fit_input.first_plane_data
    peak = first_plane_fit_input.last_peak
    group = first_plane_fit_input.group
    min_x = first_plane_fit_input.min_x
    min_y = first_plane_fit_input.min_y
    max_x = first_plane_fit_input.max_x
    max_y = first_plane_fit_input.max_y
    verbose = first_plane_fit_input.verbose
    uc_dics = first_plane_fit_input.uc_dics

    out = mod.fit(
        peak_slices, XY=XY_slices, params=p_guess, weights=weights, method=fit_method
    )

    if verbose:
        console.print(out.fit_report(), style="bold")

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


def refit_peaks_with_constraints(fit_input: FitPeaksInput, fit_result: FitPeaksResult):
    fit_results = []
    for num, d in enumerate(fit_input.data):
        plane_number = fit_input.plane_numbers[num]
        masked_data = d[fit_result.mask]
        fit_result.out.fit(
            data=masked_data,
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
        # fit_report = fit_result.out.fit_report()
        # log.write(
    return fit_results


def merge_unpacked_parameters_with_metadata(cluster_fit_df, group_of_peaks_df):
    group_of_peaks_df["prefix"] = group_of_peaks_df.ASS.apply(to_prefix)
    merged_cluster_fit_df = cluster_fit_df.merge(group_of_peaks_df, on="prefix")
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
    vclist_data = fit_input.args.get("vclist_data")
    df["vclist"] = df.plane.apply(lambda x: vclist_data[x])
    return df


def prepare_group_of_peaks_for_fitting(
    group, data, fit_peaks_input_args: FitPeaksArgs, fit_method="leastsq"
):
    lineshape_function = get_lineshape_function(fit_peaks_input_args.lineshape)

    first_plane_data = data[0]
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
        data.shape, max_x, min_x, max_y, min_y
    )
    selected_data = select_reference_planes_using_indices(
        data, fit_peaks_input_args.reference_plane_indices
    ).sum(axis=0)
    mod, p_guess = make_models(
        lineshape_function,
        group,
        selected_data,
        lineshape=fit_peaks_input_args.lineshape,
        xy_bounds=fit_peaks_input_args.xy_bounds,
    )
    peak_slices = slice_peaks_from_data_using_mask(data, mask)
    peak_slices = select_reference_planes_using_indices(
        peak_slices, fit_peaks_input_args.reference_plane_indices
    )
    peak_slices = select_planes_above_threshold_from_masked_data(
        peak_slices, fit_peaks_input_args.initial_fit_threshold
    )
    peak_slices = peak_slices.sum(axis=0)

    XY = make_meshgrid(data.shape)
    X, Y = XY

    XY_slices = np.array([X.copy()[mask], Y.copy()[mask]])
    weights = 1.0 / np.array([fit_peaks_input_args.noise] * len(np.ravel(peak_slices)))
    # weights = 1.0 / np.ravel(peak_slices)
    return FirstPlaneFitInput(
        group=group,
        last_peak=peak,
        mask=mask,
        mod=mod,
        p_guess=p_guess,
        XY=XY,
        peak_slices=peak_slices,
        XY_slices=XY_slices,
        weights=weights,
        fit_method=fit_method,
        first_plane_data=first_plane_data,
        uc_dics=fit_peaks_input_args.uc_dics,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        verbose=fit_peaks_input_args.verbose,
    )


def fit_cluster_of_peaks(
    clustid: int, peak_cluster: pd.DataFrame, fit_input: FitPeaksInput
) -> pd.DataFrame:
    data_for_fitting = prepare_group_of_peaks_for_fitting(
        peak_cluster,
        fit_input.data,
        fit_input.args,
        fit_method=fit_input.config.get("fit_method", "leastsq"),
    )
    fit_result = perform_initial_lineshape_fit_on_cluster_of_peaks(data_for_fitting)
    fit_result.out.params, float_str = set_parameters_to_fix_during_fit(
        fit_result.out.params, fit_input.args.to_fix
    )
    fit_results = refit_peaks_with_constraints(fit_input, fit_result)
    cluster_df = pd.DataFrame(fit_results)
    cluster_df = update_cluster_df_with_fit_statistics(cluster_df, fit_result.out)
    cluster_df["clustid"] = clustid
    cluster_df = merge_unpacked_parameters_with_metadata(cluster_df, peak_cluster)
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
        cluster_df = fit_cluster_of_peaks(
            clustid=clustid, peak_cluster=peak_cluster, fit_input=fit_input
        )
        cluster_dfs.append(cluster_df)
    df = pd.concat(cluster_dfs, ignore_index=True)
    df["lineshape"] = fit_input.args.lineshape.value

    if fit_input.args.vclist:
        df = add_vclist_to_df(fit_input, df)
    df = rename_columns_for_compatibility(df)
    return FitPeaksResult(df=df, log=out_str)


@dataclass
class JackKnifeResult:
    mean: float
    std: float


def jack_knife_sample_errors(
    peaks: pd.DataFrame, fit_input: FirstPlaneFitInput
) -> JackKnifeResult:
    peak_slices = fit_input.peak_slices
    XY_slices = fit_input.XY_slices
    weights = fit_input.weights
    jk_results = []
    for i in range(len(peak_slices)):
        peak_slices = np.delete(peak_slices, i, None)
        X = np.delete(XY_slices[0], i, None)
        Y = np.delete(XY_slices[1], i, None)
        weights = np.delete(weights, i, None)
        jk_results.append(
            mod.fit(peak_slices, XY=[X, Y], params=out.params, weights=weights)
        )

    # print(jk_results)
    amps = []
    sigma_xs = []
    sigma_ys = []
    names = []
    with open("test_jackknife", "w") as f:
        for i in jk_results:
            f.write(i.fit_report())
            amp, amp_err, name = get_params(i.params, "amp")
            sigma_x, sigma_x_err, name_x = get_params(i.params, "sigma_x")
            sigma_y, sigma_y_err, name_y = get_params(i.params, "sigma_y")
            f.write(f"{amp},{amp_err},{name_y}\n")
            amps.extend(amp)
            names.extend(name_y)
            sigma_xs.extend(sigma_x)
            sigma_ys.extend(sigma_y)

        df = pd.DataFrame(
            {"amp": amps, "name": names, "sigma_x": sigma_xs, "sigma_y": sigma_ys}
        )
        grouped = df.groupby("name")
        mean_amps = grouped.amp.mean()
        std_amps = grouped.amp.std()
        mean_sigma_x = grouped.sigma_x.mean()
        std_sigma_x = grouped.sigma_x.std()
        mean_sigma_y = grouped.sigma_y.mean()
        std_sigma_y = grouped.sigma_y.std()
        f.write("#####################################\n")
        f.write(
            f"{mean_amps}, {std_amps}, {mean_sigma_x}, {std_sigma_x}, {mean_sigma_y}, {std_sigma_y} "
        )
        f.write(self.out.fit_report())
        f.write("#####################################\n")
    # print(amps)
    # mean = np.mean(amps)
    # std =  np.std(amps)
    return JackKnifeResult(mean=mean_amps, std=std_amps)
