#!/usr/bin/env python3
"""Fit and deconvolute NMR peaks

    Usage:
        fit <peaklist> <data> <output> [options]

    Arguments:
        <peaklist>                                  peaklist output from read_peaklist.py
        <data>                                      2D or pseudo3D NMRPipe data (single file)
        <output>                                    output peaklist "<output>.csv" will output CSV
                                                    format file, "<output>.tab" will give a tab delimited output
                                                    while "<output>.pkl" results in Pandas pickle of DataFrame

    Options:
        -h --help                                   Show this page
        -v --version                                Show version

        --dims=<ID,F1,F2>                           Dimension order [default: 0,1,2]

        --max_cluster_size=<max_cluster_size>       Maximum size of cluster to fit (i.e exclude large clusters) [default: 999]

        --lineshape=<PV/V/G/L/PV_PV>                Lineshape to fit [default: PV]

        --fix=<fraction,sigma,center>               Parameters to fix after initial fit on summed planes [default: fraction,sigma,center]

        --xy_bounds=<x_ppm,y_ppm>                   Bound X and Y peak centers during fit [default: None]
                                                    This can be set like so --xy_bounds=0.1,0.5

        --vclist=<fname>                            Bruker style vclist [default: None]

        --plane=<int>                               Specific plane(s) to fit [default: -1]
                                                    eg. --plane=1 or --plane=1,4,5

        --exclude_plane=<int>                       Specific plane(s) to fit [default: -1]
                                                    eg. --plane=1 or --plane=1,4,5

        --nomp                                      Do not use multiprocessing

        --plot=<dir>                                Whether to plot wireframe fits for each peak
                                                    (saved into <dir>) [default: None]

        --show                                      Whether to show (using plt.show()) wireframe
                                                    fits for each peak. Only works if --plot is also selected

        --verb                                      Print what's going on




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
import os
from pathlib import Path
from multiprocessing import cpu_count, Pool

import nmrglue as ng
import numpy as np
import pandas as pd

from docopt import docopt
from colorama import Fore, init

from tabulate import tabulate
from skimage.filters import threshold_otsu
from schema import Schema, And, Or, Use, SchemaError

from peakipy.core import (
    fix_params,
    get_params,
    fit_first_plane,
    LoadData,
    run_log,
    read_config,
    voigt2d,
    pvoigt2d,
    pv_pv,
)

# colorama
init(autoreset=True)
# some constants
π = np.pi
sqrt2 = np.sqrt(2.0)
# temp and log paths
tmp_path = Path("tmp")
tmp_path.mkdir(exist_ok=True)
log_path = Path("log.txt")
# for printing dataframes
column_selection = ["INDEX", "ASS", "X_PPM", "Y_PPM", "CLUSTID", "MEMCNT"]


def check_xybounds(x):
    x = x.split(",")
    if len(x) == 2:
        # xy_bounds = float(x[0]), float(x[1])
        xy_bounds = [float(i) for i in x]
        return xy_bounds
    else:
        print(Fore.RED + "🤔 xy_bounds must be pair of floats e.g. --xy_bounds=0.05,0.5")
        exit()


# prepare data for multiprocessing


def chunks(l, n):
    """ split list into n chunks

        will return empty lists if n > len(l)

        :param l: list of values you wish to split
        :type l: list
        :param n: number of sub lists you want to generate
        :type n: int

        :returns sub_lists: list of lists
        :rtype sub_lists: list
    """
    # create n empty lists
    sub_lists = [[] for _ in range(n)]
    # append into n lists
    for num, i in enumerate(l):
        sub_lists[num % n].append(i)
    return sub_lists


def split_peaklist(peaklist, n_cpu, tmp_path=tmp_path):
    """ split peaklist into smaller files based on number of cpus

        :param peaklist: Peaklist data generated by peakipy read or edit scripts
        :type peaklist: pandas.DataFrame

        :returns tmp_path: Temporary directory path
        :rtype tmp_path: pathlib.Path
    """
    # clustid numbers
    clustids = peaklist.CLUSTID.unique()
    # make n_cpu lists of clusters
    clustids = chunks(clustids, n_cpu)
    for i in range(n_cpu):
        # get sub dataframe containing ith clustid list
        split_peaks = peaklist[peaklist.CLUSTID.isin(clustids[i])]
        # save sub dataframe
        split_peaks.to_csv(tmp_path / f"peaks_{i}.csv", index=False)
    return tmp_path


class FitPeaksInput:
    """ input data for the fit_peaks function """

    def __init__(self, args: dict, data: np.array, config: dict, plane_numbers: list):

        self._data = data
        self._args = args
        self._config = config
        self._plane_numbers = plane_numbers

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


class FitPeaksResult:
    """ Result of fitting a set of peaks """

    def __init__(self, df: pd.DataFrame, log: str):

        self._df = df
        self._log = log

    @property
    def df(self):
        return self._df

    @property
    def log(self):
        return self._log


def fit_peaks(peaks: pd.DataFrame, fit_input: FitPeaksInput):
    """ Fit set of peak clusters to lineshape model

        :param peaks: peaklist with generated by peakipy read or edit
        :type peaks: pd.DataFrame

        :param fit_input: Data structure containing input parameters (args, config and NMR data)
        :type fit_input: FitPeaksInput

        :returns: Data structure containing pd.DataFrame with the fitted results and a log
        :rtype: FitPeaksResult
    """
    # sum planes for initial fit
    summed_planes = fit_input.data.sum(axis=0)

    # group peaks based on CLUSTID
    groups = peaks.groupby("CLUSTID")
    # setup arguments
    to_fix = fit_input.args.get("to_fix")
    noise = fit_input.args.get("noise")
    verb = fit_input.args.get("verb")
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

    if lineshape == "V":
        # lorentzian linewidth
        gamma_xs = []
        gamma_ys = []

    if lineshape == "PV_PV":
        # seperate fractions for each dim
        fractions_x = []
        fractions_y = []
    else:
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
                summed_planes,
                # norm(summed_planes),
                uc_dics,
                lineshape=lineshape,
                xy_bounds=xy_bounds,
                verbose=verb,
                noise=noise,
                fit_method=fit_input.config.get("fit_method", "leastsq"),
            )
            fit_result.plot(
                plot_path=fit_input.args.get("plot"),
                show=fit_input.args.get("--show"),
                nomp=fit_input.args.get("--nomp"),
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
            if len(to_fix) == 0 or to_fix == "None":
                float_str = "Floating all parameters"
                if verb:
                    print(float_str)
                pass
            else:
                to_fix = to_fix
                float_str = f"Fixing parameters: {to_fix}"
                if verb:
                    print(float_str)
                fix_params(first.params, to_fix)
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
                    print(fit_report)

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
                if lineshape == "PV_PV":
                    frac_x, frac_err_x, name = get_params(first.params, "fraction_x")
                    frac_y, frac_err_y, name = get_params(first.params, "fraction_y")
                    fractions_x.extend(frac_x)
                    fractions_y.extend(frac_y)
                elif lineshape == "V":
                    frac, frac_err, name = get_params(first.params, "fraction")
                    gam_x, gam_x_err, gx_name = get_params(first.params, "gamma_x")
                    gam_y, gam_y_err, gy_name = get_params(first.params, "gamma_y")
                    gamma_xs.extend(gam_x)
                    gamma_ys.extend(gam_y)
                    fractions.extend(frac)
                else:
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
                lineshapes.extend([lineshape for _ in amp])
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
    if lineshape == "PV_PV":
        df_dic["fraction_x"] = fractions_x
        df_dic["fraction_y"] = fractions_y
    else:
        df_dic["fraction"] = fractions

    if lineshape == "V":
        df_dic["gamma_x"] = gamma_xs
        df_dic["gamma_y"] = gamma_ys
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


def check_input(args):
    """ Validate commandline input

        :param args: docopt argument dictionary
        :type args: dict

    """
    schema = Schema(
        {
            "<peaklist>": And(
                os.path.exists,
                open,
                error=Fore.RED + f"🤔 {args['<peaklist>']} should exist and be readable",
            ),
            "<data>": And(
                os.path.exists,
                # Use(
                ng.pipe.read,
                error=Fore.RED
                + f"🤔 {args['<data>']} should be NMRPipe format 2D or 3D cube",
                # ),
                # error=f"🤔 {args['<data>']} either does not exist or is not an NMRPipe format 2D or 3D",
            ),
            "<output>": Use(str),
            "--max_cluster_size": And(
                Use(int),
                lambda n: 0 < n,
                error=Fore.RED + "Max cluster size must be greater than 0",
            ),
            "--lineshape": Or(
                "PV",
                "L",
                "G",
                "PV_PV",
                "PV_G",
                "PV_L",
                "G_L",
                "V",
                error=Fore.RED
                + "🤔 --lineshape must be either PV, L, G, PV_PV, PV_G, PV_L, G_L or V",
            ),
            "--fix": Or(
                Use(
                    lambda x: [
                        i
                        for i in x.split(",")
                        if i
                        in [
                            "fraction",
                            "center",
                            "sigma",
                            "gamma",
                            "fraction_x",
                            "center_x",
                            "sigma_x",
                            "gamma_x",
                            "fraction_y",
                            "center_y",
                            "sigma_y",
                            "gamma_y",
                        ]
                    ]
                )
            ),
            "--dims": Use(
                lambda n: [int(i) for i in eval(n)],
                error=Fore.RED
                + "🤔 --dims should be list of integers e.g. --dims=0,1,2",
            ),
            "--vclist": Or(
                "None",
                And(
                    os.path.exists,
                    Use(
                        np.genfromtxt,
                        error=Fore.RED + f"🤔 cannot open {args.get('--vclist')}",
                    ),
                ),
            ),
            "--plot": Or("None", Use(lambda f: Path(f))),
            "--xy_bounds": Or(
                "None",
                Use(
                    check_xybounds,
                    error=Fore.RED
                    + "🤔 xy_bounds must be pair of floats e.g. --xy_bounds=0.05,0.5",
                ),
            ),
            "--plane": Or(
                0,
                Use(
                    lambda n: [int(i) for i in n.split(",")],
                    error=Fore.RED
                    + "🤔 plane(s) to fit should be an integer or list of integers e.g. --plane=1,2,3,4",
                ),
            ),
            "--exclude_plane": Or(
                0,
                Use(
                    lambda n: [int(i) for i in n.split(",")],
                    error=Fore.RED
                    + "🤔 plane(s) to exclude should be an integer or list of integers e.g. --exclude_plane=1,2,3,4",
                ),
            ),
            object: object,
        },
        # ignore_extra_keys=True,
    )

    # validate arguments
    try:
        args = schema.validate(args)
        return args
    except SchemaError as e:
        exit(e)


def main(arguments):
    # number of CPUs
    n_cpu = cpu_count()

    args = check_input(docopt(__doc__, argv=arguments))

    lineshape = args.get("--lineshape")
    args["lineshape"] = lineshape

    # params to fix
    to_fix = args.get("--fix")
    args["to_fix"] = to_fix
    # print(to_fix)
    verb = args.get("--verb")
    if verb:
        print("Using ", args)
    args["verb"] = verb

    # path to peaklist
    peaklist = Path(args.get("<peaklist>"))
    # get dims from command line input
    # read NMR data
    args, config = read_config(args)
    dims = args.get("--dims")
    data = args.get("<data>")
    peakipy_data = LoadData(peaklist, data, dims=dims)

    # only include peaks with 'include'
    if "include" in peakipy_data.df.columns:
        pass
    else:
        # for compatibility
        peakipy_data.df["include"] = peakipy_data.df.apply(lambda _: "yes", axis=1)

    if len(peakipy_data.df[peakipy_data.df.include != "yes"]) > 0:
        excluded = peakipy_data.df[peakipy_data.df.include != "yes"][column_selection]
        print(
            Fore.YELLOW + f"The following peaks have been exluded:\n",
            tabulate(excluded, headers="keys", tablefmt="fancy_grid"),
        )
        peakipy_data.df = peakipy_data.df[peakipy_data.df.include == "yes"]

    # filter list based on cluster size
    max_cluster_size = args.get("--max_cluster_size")
    if max_cluster_size == 999:
        max_cluster_size = peakipy_data.df.MEMCNT.max()
        if peakipy_data.df.MEMCNT.max() > 10:
            print(
                Fore.RED
                + f"""
                ##################################################################
                You have some clusters of as many as {max_cluster_size} peaks.
                You may want to consider reducing the size of your clusters as the
                fits will struggle.

                Otherwise you can use the --max_cluster_size flag to exclude large
                clusters
                ##################################################################
            """
            )
    else:
        max_cluster_size = max_cluster_size
    args["max_cluster_size"] = max_cluster_size

    # read vclist
    vclist = args.get("--vclist")
    if type(vclist) == np.ndarray:
        vclist_data = vclist
        args["vclist_data"] = vclist_data
        vclist = True
    else:
        vclist = False
    args["vclist"] = vclist

    # plot results or not
    plot = args.get("--plot")
    if plot == "None":
        plot = None
        log_file = open(tmp_path / log_path, "w")
    else:
        log_file = open(tmp_path / log_path, "w")
        plot.mkdir(parents=True, exist_ok=True)

    args["plot"] = plot

    uc_dics = {"f1": peakipy_data.uc_f1, "f2": peakipy_data.uc_f2}
    args["uc_dics"] = uc_dics

    # check data shape is consistent with dims
    if len(peakipy_data.dims) != len(peakipy_data.data.shape):
        print(
            f"Dims are {peakipy_data.dims} while data shape is {peakipy_data.data.shape}?"
        )
        exit()

    plane_numbers = np.arange(peakipy_data.data.shape[peakipy_data.dims[0]])
    # only fit specified planes
    if args.get("--plane", [-1]) != [-1]:
        _inds = args.get("--plane")
        inds = [i for i in _inds]
        data_inds = [
            (i in inds) for i in range(peakipy_data.data.shape[peakipy_data.dims[0]])
        ]
        plane_numbers = np.arange(peakipy_data.data.shape[peakipy_data.dims[0]])[data_inds]
        peakipy_data.data = peakipy_data.data[data_inds]
        print(
            Fore.YELLOW + f"Using only planes {_inds} data now has the following shape",
            peakipy_data.data.shape,
        )
        if peakipy_data.data.shape[peakipy_data.dims[0]] == 0:
            print(Fore.RED + "You have excluded all the data!", peakipy_data.data.shape)
            exit()

    # do not fit these planes
    if args.get("--exclude_plane", [-1]) != [-1]:
        _inds = args.get("--exclude_plane")
        inds = [i for i in _inds]
        data_inds = [
            (i not in inds)
            for i in range(peakipy_data.data.shape[peakipy_data.dims[0]])
        ]
        plane_numbers = np.arange(peakipy_data.data.shape[peakipy_data.dims[0]])[data_inds]
        peakipy_data.data = peakipy_data.data[data_inds]
        print(
            Fore.YELLOW + f"Excluding planes {_inds} data now has the following shape",
            peakipy_data.data.shape,
        )
        if peakipy_data.data.shape[peakipy_data.dims[0]] == 0:
            print(Fore.RED + "You have excluded all the data!", peakipy_data.data.shape)
            exit()

    # setting noise for calculation of chi square
    if not args.get("noise"):
        noise = abs(threshold_otsu(peakipy_data.data))

    args["noise"] = noise
    # print(noise)

    xy_bounds = args.get("--xy_bounds")

    if xy_bounds == "None":
        xy_bounds = None
    else:
        # convert ppm to points
        xy_bounds[0] = xy_bounds[0] * peakipy_data.pt_per_ppm_f2
        xy_bounds[1] = xy_bounds[1] * peakipy_data.pt_per_ppm_f1

    args["xy_bounds"] = xy_bounds

    # convert linewidths from Hz to points in case they were adjusted when running edit.py
    peakipy_data.df["XW"] = peakipy_data.df.XW_HZ * peakipy_data.pt_per_hz_f2
    peakipy_data.df["YW"] = peakipy_data.df.YW_HZ * peakipy_data.pt_per_hz_f1

    # convert peak positions from ppm to points in case they were adjusted running edit.py
    peakipy_data.df["X_AXIS"] = peakipy_data.df.X_PPM.apply(
        lambda x: peakipy_data.uc_f2(x, "PPM")
    )
    peakipy_data.df["Y_AXIS"] = peakipy_data.df.Y_PPM.apply(
        lambda x: peakipy_data.uc_f1(x, "PPM")
    )
    peakipy_data.df["X_AXISf"] = peakipy_data.df.X_PPM.apply(
        lambda x: peakipy_data.uc_f2.f(x, "PPM")
    )
    peakipy_data.df["Y_AXISf"] = peakipy_data.df.Y_PPM.apply(
        lambda x: peakipy_data.uc_f1.f(x, "PPM")
    )
    # start fitting data
    # prepare data for multiprocessing
    if (peakipy_data.df.CLUSTID.nunique() >= n_cpu) and not args.get("--nomp"):
        print(Fore.GREEN + "Using multiprocessing")
        # split peak lists
        tmp_dir = split_peaklist(peakipy_data.df, n_cpu)
        peaklists = [
            pd.read_csv(tmp_dir / Path(f"peaks_{i}.csv")) for i in range(n_cpu)
        ]
        args_list = [
            FitPeaksInput(args, peakipy_data.data, config, plane_numbers) for _ in range(n_cpu)
        ]
        with Pool(processes=n_cpu) as pool:
            # result = pool.map(fit_peaks, peaklists)
            result = pool.starmap(fit_peaks, zip(peaklists, args_list))
            df = pd.concat([i.df for i in result], ignore_index=True)
            for num, i in enumerate(result):
                i.df.to_csv(tmp_dir / Path(f"peaks_{num}_fit.csv"), index=False)
                log_file.write(i.log + "\n")
    else:
        print(Fore.GREEN + "Not using multiprocessing")
        result = fit_peaks(
            peakipy_data.df, FitPeaksInput(args, peakipy_data.data, config, plane_numbers)
        )
        df = result.df
        log_file.write(result.log)

    # finished fitting

    # close log file
    log_file.close()
    output = Path(args["<output>"])
    suffix = output.suffix
    #  convert sigmas to fwhm
    if args["lineshape"] == "V":
        # calculate peak height
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
        df["height_err"] = df.apply(lambda x: x.amp_err * (x.height / x.amp), axis=1)
        df["fwhm_g_x"] = df.sigma_x.apply(
            lambda x: 2.0 * x * np.sqrt(2.0 * np.log(2.0))
        )  # fwhm of gaussian
        df["fwhm_g_y"] = df.sigma_y.apply(
            lambda x: 2.0 * x * np.sqrt(2.0 * np.log(2.0))
        )
        df["fwhm_l_x"] = df.gamma_x.apply(lambda x: 2.0 * x)  # fwhm of lorentzian
        df["fwhm_l_y"] = df.gamma_y.apply(lambda x: 2.0 * x)
        df["fwhm_x"] = df.apply(
            lambda x: 1.0692 * x.gamma_x
            + np.sqrt(0.8664 * x.gamma_x ** 2.0 + 5.545177*x.sigma_x ** 2.0),
            axis=1,
        )
        df["fwhm_y"] = df.apply(
            lambda x: 1.0692 * x.gamma_y
            + np.sqrt(0.8664 * x.gamma_y ** 2.0 + 5.545177*x.sigma_y ** 2.0),
            axis=1,
        )
        # df["fwhm_y"] = df.apply(lambda x: x.gamma_y + np.sqrt(x.gamma_y**2.0 + 4 * x.sigma_y**2.0 * 2.0 * np.log(2.)), axis=1)
        # df["fwhm_x"] = df.apply(lambda x: x.gamma_x + np.sqrt(x.gamma_x**2.0 + 4 * x.sigma_x**2.0 * 2.0 * np.log(2.)), axis=1)
        # df["fwhm_y"] = df.apply(lambda x: x.gamma_y + np.sqrt(x.gamma_y**2.0 + 4 * x.sigma_y**2.0 * 2.0 * np.log(2.)), axis=1)

    if args["lineshape"] == "PV":
        # calculate peak height
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
        df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
        df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)

    elif args["lineshape"] == "G":
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
        df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
        df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)

    elif args["lineshape"] == "L":
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
        df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
        df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)

    elif args["lineshape"] == "PV_PV":
        # calculate peak height
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
        df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
        df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)

    else:
        df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
        df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)
    #  convert values to ppm
    df["center_x_ppm"] = df.center_x.apply(lambda x: peakipy_data.uc_f2.ppm(x))
    df["center_y_ppm"] = df.center_y.apply(lambda x: peakipy_data.uc_f1.ppm(x))
    df["init_center_x_ppm"] = df.init_center_x.apply(
        lambda x: peakipy_data.uc_f2.ppm(x)
    )
    df["init_center_y_ppm"] = df.init_center_y.apply(
        lambda x: peakipy_data.uc_f1.ppm(x)
    )
    df["sigma_x_ppm"] = df.sigma_x.apply(lambda x: x * peakipy_data.ppm_per_pt_f2)
    df["sigma_y_ppm"] = df.sigma_y.apply(lambda x: x * peakipy_data.ppm_per_pt_f1)
    df["fwhm_x_ppm"] = df.fwhm_x.apply(lambda x: x * peakipy_data.ppm_per_pt_f2)
    df["fwhm_y_ppm"] = df.fwhm_y.apply(lambda x: x * peakipy_data.ppm_per_pt_f1)
    df["fwhm_x_hz"] = df.fwhm_x.apply(lambda x: x * peakipy_data.hz_per_pt_f2)
    df["fwhm_y_hz"] = df.fwhm_y.apply(lambda x: x * peakipy_data.hz_per_pt_f1)

    # save data
    if suffix == ".csv":
        df.to_csv(output, float_format="%.4f", index=False)

    elif suffix == ".tab":
        df.to_csv(output, sep="\t", float_format="%.4f", index=False)

    else:
        df.to_pickle(output)

    print(
        """
           🍾 ✨ Finished! ✨ 🍾       
             
        """
    )
    run_log()


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(arguments=argv)
