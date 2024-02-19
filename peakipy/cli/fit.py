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
from lmfit import Model, Parameter
from lmfit.model import ModelResult

from peakipy.core import (
    fix_params,
    fit_first_plane,
    LoadData,
    Lineshape,
    pvoigt2d,
    voigt2d,
    pv_pv,
    to_prefix,
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
    dims: List[int] = field(default_factory=lambda: [0, 1, 2])
    colors: Tuple[str] = ("#5e3c99", "#e66101")
    max_cluster_size: Optional[int] = None
    to_fix: List[str] = field(default_factory=lambda: ["fraction", "sigma", "center"])
    xy_bounds: Tuple[float, float] = ((0, 0),)
    vclist: Optional[Path] = (None,)
    plane: Optional[List[int]] = (None,)
    exclude_plane: Optional[List[int]] = (None,)
    reference_plane_index: List[int] = ([],)
    initial_fit_threshold: Optional[float] = (None,)
    mp: bool = (True,)
    plot: Optional[Path] = (None,)
    show: bool = (False,)
    verb: bool = (False,)


class FitPeaksInput:
    """input data for the fit_peaks function"""

    def __init__(
        self,
        args: dict,
        data: np.array,
        config: dict,
        plane_numbers: list,
        reference_planes_for_initial_fit: List[int] = [],
        use_only_planes_above_threshold: Optional[float] = None,
    ):
        self._data = data
        self._args = args
        self._config = config
        self._plane_numbers = plane_numbers
        self._planes_for_initial_fit = reference_planes_for_initial_fit
        self._use_only_planes_above_threshold = use_only_planes_above_threshold

    def check_integer_list(self):
        if hasattr(self._planes_for_initial_fit, "append"):
            pass
        else:
            return False
        if all([(type(i) == int) for i in self._planes_for_initial_fit]):
            pass
        else:
            return False
        if all([((i - 1) > self._data.shape[0]) for i in self._planes_for_initial_fit]):
            return True
        else:
            return False

    def sum_planes_for_initial_fit(self):
        if (
            self._planes_for_initial_fit
            == self._use_only_planes_above_threshold
            == None
        ):
            return self._data.sum(axis=0)

        elif self.check_integer_list():
            return self._data[self._planes_for_initial_fit].sum(axis=0)

        elif type(self._use_only_planes_above_threshold) == float:
            # very crude at the moment
            return self._data[
                self._data.max(axis=1).max(axis=1)
                > self._use_only_planes_above_threshold
            ]
        else:
            return self._data.sum(axis=0)

    @property
    def data(self):
        return self._data

    @property
    def args(self):
        return self._args

    @property
    def config(self):
        return self._config

    @property
    def plane_numbers(self):
        return self._plane_numbers

    @property
    def summed_planes_for_initial_fit(self):
        return self.sum_planes_for_initial_fit()


class FitPeaksResult:
    """Result of fitting a set of peaks"""

    def __init__(self, df: pd.DataFrame, log: str):
        self._df = df
        self._log = log

    @property
    def df(self):
        return self._df

    @property
    def log(self):
        return self._log


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
    cluster_of_peaks, fit_input: FitPeaksInput
):
    fit_result = fit_first_plane(
        cluster_of_peaks,
        fit_input.data,
        # norm(summed_planes),
        fit_input.args.get("uc_dics"),
        lineshape=fit_input.args.get("lineshape"),
        xy_bounds=fit_input.args.get("xy_bounds"),
        verbose=fit_input.args.get("verb"),
        noise=fit_input.args.get("noise"),
        fit_method=fit_input.config.get("fit_method", "leastsq"),
        reference_plane_indices=fit_input.args.get("reference_plane_indices"),
        threshold=fit_input.args.get("initial_fit_threshold"),
    )
    return fit_result


def refit_peaks_with_constraints(fit_input: FitPeaksInput, fit_result: FitPeaksResult):
    fit_results = []
    for num, d in enumerate(fit_input.data):
        plane_number = fit_input.plane_numbers[num]
        fit_result.out.fit(
            data=d[fit_result.mask],
            params=fit_result.out.params,
            weights=1.0
            / np.array(
                [fit_input.args.get("noise")] * len(np.ravel(d[fit_result.mask]))
            ),
        )
        fit_results.extend(
            unpack_fitted_parameters_for_lineshape(
                fit_input.args.get("lineshape"),
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


def fit_peaks(peaks: pd.DataFrame, fit_input: FitPeaksInput) -> FitPeaksResult:
    """Fit set of peak clusters to lineshape model

    :param peaks: peaklist with generated by peakipy read or edit
    :type peaks: pd.DataFrame

    :param fit_input: Data structure containing input parameters (args, config and NMR data)
    :type fit_input: FitPeaksInput

    :returns: Data structure containing pd.DataFrame with the fitted results and a log
    :rtype: FitPeaksResult
    """
    peak_clusters = peaks.groupby("CLUSTID")
    max_cluster_size = fit_input.args.get("max_cluster_size")
    filtered_peaks = filter_peak_clusters_by_max_cluster_size(
        peak_clusters, max_cluster_size
    )
    peak_clusters = filtered_peaks.groupby("CLUSTID")
    # setup arguments
    to_fix = fit_input.args.get("to_fix")
    lineshape = fit_input.args.get("lineshape")
    out_str = ""
    cluster_dfs = []
    for name, peak_cluster in peak_clusters:
        fit_result = perform_initial_lineshape_fit_on_cluster_of_peaks(
            peak_cluster, fit_input
        )
        fit_result.out.params, float_str = set_parameters_to_fix_during_fit(
            fit_result.out.params, to_fix
        )
        fit_results = refit_peaks_with_constraints(fit_input, fit_result)
        cluster_df = pd.DataFrame(fit_results)
        cluster_df = update_cluster_df_with_fit_statistics(cluster_df, fit_result.out)
        cluster_df["clustid"] = name
        cluster_df = merge_unpacked_parameters_with_metadata(cluster_df, peak_cluster)
        cluster_dfs.append(cluster_df)
    df = pd.concat(cluster_dfs, ignore_index=True)
    df["lineshape"] = lineshape.value
    if fit_input.args.get("vclist"):
        df = add_vclist_to_df(fit_input, df)
    df = rename_columns_for_compatibility(df)
    return FitPeaksResult(df=df, log=out_str)
