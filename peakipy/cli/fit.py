#!/usr/bin/env python3
"""Fit and deconvolute NMR peaks: Functions used for running peakipy fit
"""
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from rich import print
from rich.console import Console
from pydantic import BaseModel

from peakipy.core import (
    fix_params,
    get_params,
    fit_first_plane,
    LoadData,
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


def fit_peaks(peaks: pd.DataFrame, fit_input: FitPeaksInput) -> FitPeaksResult:
    """Fit set of peak clusters to lineshape model

    :param peaks: peaklist with generated by peakipy read or edit
    :type peaks: pd.DataFrame

    :param fit_input: Data structure containing input parameters (args, config and NMR data)
    :type fit_input: FitPeaksInput

    :returns: Data structure containing pd.DataFrame with the fitted results and a log
    :rtype: FitPeaksResult
    """
    # group peaks based on CLUSTID
    groups = peaks.groupby("CLUSTID")
    # setup arguments
    to_fix = fit_input.args.get("to_fix")
    noise = fit_input.args.get("noise")
    verb = fit_input.args.get("verb")
    initial_fit_threshold = fit_input.args.get("initial_fit_threshold")
    reference_plane_indices = fit_input.args.get("reference_plane_indices")
    lineshape = fit_input.args.get("lineshape")
    xy_bounds = fit_input.args.get("xy_bounds")
    vclist = fit_input.args.get("vclist")
    uc_dics = fit_input.args.get("uc_dics")

    # for saving data, currently not using errs for center and sigma
    amps = []
    amp_errs = []

    center_xs = []
    init_center_xs = []
    # center_x_errs = []

    center_ys = []
    init_center_ys = []
    # center_y_errs = []

    sigma_ys = []
    # sigma_y_errs = []

    sigma_xs = []
    # sigma_x_errs = []

    match lineshape:
        case lineshape.V:
            # lorentzian linewidth
            gamma_xs = []
            gamma_ys = []
            fractions = []

        case lineshape.PV_PV:
            # seperate fractions for each dim
            fractions_x = []
            fractions_y = []
        case _:
            fractions = []

    # lists for saving data
    names = []
    assign = []
    clustids = []
    memcnts = []
    planes = []
    x_radii = []
    y_radii = []
    x_radii_ppm = []
    y_radii_ppm = []
    lineshapes = []
    # errors
    chisqrs = []
    redchis = []
    aics = []
    res_sum = []

    # iterate over groups of peaks
    out_str = ""
    for name, group in groups:
        #  max cluster size
        len_group = len(group)
        if len_group <= fit_input.args.get("max_cluster_size"):
            if len_group == 1:
                peak_str = "peak"
            else:
                peak_str = "peaks"

            out_str += f"""

            ####################################
            Fitting cluster of {len_group} {peak_str}
            ####################################
            """
            # fits sum of all planes first
            fit_result = fit_first_plane(
                group,
                fit_input.data,
                # norm(summed_planes),
                uc_dics,
                lineshape=lineshape,
                xy_bounds=xy_bounds,
                verbose=verb,
                noise=noise,
                fit_method=fit_input.config.get("fit_method", "leastsq"),
                reference_plane_indices=reference_plane_indices,
                threshold=initial_fit_threshold,
            )
            fit_result.plot(
                plot_path=fit_input.args.get("plot"),
                show=fit_input.args.get("show"),
                mp=fit_input.args.get("mp"),
            )
            # jack_knife_result = fit_result.jackknife()
            # print("JackKnife", jack_knife_result.mean, jack_knife_result.std)
            first = fit_result.out
            mask = fit_result.mask
            #            log.write(
            out_str += fit_result.fit_str
            out_str += f"""
        ------------------------------------
                   Summed planes
        ------------------------------------
        {first.fit_report()}
                        """
            #            )
            # fix sigma center and fraction parameters
            # could add an option to select params to fix
            match to_fix:
                case None | () | []:
                    float_str = "Floating all parameters"
                case ["None"] | ["none"]:
                    float_str = "Floating all parameters"
                case _:
                    float_str = f"Fixing parameters: {to_fix}"
                    fix_params(first.params, to_fix)
            if verb:
                console.print(float_str, style="magenta")

            out_str += float_str + "\n"

            for num, d in enumerate(fit_input.data):
                plane_number = fit_input.plane_numbers[num]
                first.fit(
                    data=d[mask],
                    params=first.params,
                    weights=1.0 / np.array([noise] * len(np.ravel(d[mask]))),
                )
                fit_report = first.fit_report()
                # log.write(
                out_str += f"""
        ------------------------------------
                     Plane = {num+1}
        ------------------------------------
        {fit_report}
                        """
                #               )
                if verb:
                    console.print(fit_report, style="bold")

                amp, amp_err, name = get_params(first.params, "amplitude")
                cen_x, cen_x_err, cx_name = get_params(first.params, "center_x")
                cen_y, cen_y_err, cy_name = get_params(first.params, "center_y")
                sig_x, sig_x_err, sx_name = get_params(first.params, "sigma_x")
                sig_y, sig_y_err, sy_name = get_params(first.params, "sigma_y")
                # currently chi square is calculated for all peaks in cluster (not individual peaks)
                # chi2 - residual sum of squares
                chisqrs.extend([first.chisqr for _ in sy_name])
                # reduced chi2
                redchis.extend([first.redchi for _ in sy_name])
                # Akaike Information criterion
                aics.extend([first.aic for _ in sy_name])
                # residual sum of squares
                res_sum.extend([np.sum(first.residual) for _ in sy_name])

                # deal with lineshape specific parameters
                match lineshape:
                    case lineshape.PV_PV:
                        frac_x, frac_err_x, name = get_params(
                            first.params, "fraction_x"
                        )
                        frac_y, frac_err_y, name = get_params(
                            first.params, "fraction_y"
                        )
                        fractions_x.extend(frac_x)
                        fractions_y.extend(frac_y)
                    case lineshape.V:
                        frac, frac_err, name = get_params(first.params, "fraction")
                        gam_x, gam_x_err, gx_name = get_params(first.params, "gamma_x")
                        gam_y, gam_y_err, gy_name = get_params(first.params, "gamma_y")
                        gamma_xs.extend(gam_x)
                        gamma_ys.extend(gam_y)
                        fractions.extend(frac)
                    case _:
                        frac, frac_err, name = get_params(first.params, "fraction")
                        fractions.extend(frac)

                # extend lists with fit data
                amps.extend(amp)
                amp_errs.extend(amp_err)
                center_xs.extend(cen_x)
                init_center_xs.extend(group.X_AXISf)
                # center_x_errs.extend(cen_x_err)
                center_ys.extend(cen_y)
                init_center_ys.extend(group.Y_AXISf)
                # center_y_errs.extend(cen_y_err)
                sigma_xs.extend(sig_x)
                # sigma_x_errs.extend(sig_x_err)
                sigma_ys.extend(sig_y)
                # sigma_y_errs.extend(sig_y_err)
                # add plane number, this should map to vclist
                planes.extend([plane_number for _ in amp])
                lineshapes.extend([lineshape.value for _ in amp])
                #  get prefix for fit
                names.extend([first.model.prefix] * len(name))
                assign.extend(group["ASS"])
                clustids.extend(group["CLUSTID"])
                memcnts.extend(group["MEMCNT"])
                x_radii.extend(group["X_RADIUS"])
                y_radii.extend(group["Y_RADIUS"])
                x_radii_ppm.extend(group["X_RADIUS_PPM"])
                y_radii_ppm.extend(group["Y_RADIUS_PPM"])

    df_dic = {
        "fit_prefix": names,
        "assignment": assign,
        "amp": amps,
        "amp_err": amp_errs,
        # "height": heights,
        # "height_err": height_errs,
        "center_x": center_xs,
        "init_center_x": init_center_xs,
        # "center_x_err": center_x_errs,
        "center_y": center_ys,
        "init_center_y": init_center_ys,
        # "center_y_err": center_y_errs,
        "sigma_x": sigma_xs,
        # "sigma_x_err": sigma_x_errs,
        "sigma_y": sigma_ys,
        # "sigma_y_err": sigma_y_errs,
        "clustid": clustids,
        "memcnt": memcnts,
        "plane": planes,
        "x_radius": x_radii,
        "y_radius": y_radii,
        "x_radius_ppm": x_radii_ppm,
        "y_radius_ppm": y_radii_ppm,
        "lineshape": lineshapes,
        "aic": aics,
        "chisqr": chisqrs,
        "redchi": redchis,
        "residual_sum": res_sum,
        # "slope": slopes,
        # "intercept": intercepts
    }

    # lineshape specific
    match lineshape:
        case lineshape.PV_PV:
            df_dic["fraction_x"] = fractions_x
            df_dic["fraction_y"] = fractions_y
        case lineshape.V:
            df_dic["gamma_x"] = gamma_xs
            df_dic["gamma_y"] = gamma_ys
            df_dic["fraction"] = fractions
        case _:
            df_dic["fraction"] = fractions

    #  make dataframe
    df = pd.DataFrame(df_dic)
    # Fill nan values
    df.fillna(value=np.nan, inplace=True)
    # vclist
    if vclist:
        vclist_data = fit_input.args.get("vclist_data")
        df["vclist"] = df.plane.apply(lambda x: vclist_data[x])
    #  output data
    return FitPeaksResult(df=df, log=out_str)
