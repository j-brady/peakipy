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

        --lineshape=<G/L/PV/PV_PV/PV_G/PV_L/G_L>    lineshape to fit [default: PV]

        --fix=<fraction,sigma,center>               Parameters to fix after initial fit on summed planes [default: fraction,sigma,center]

        --xy_bounds=<x_ppm,y_ppm>                   Bound X and Y peak centers during fit [default: None]
                                                    This can be set like so --xy_bounds=0.1,0.5

        --vclist=<fname>                            Bruker style vclist [default: None]

        --plane=<int>                               Specific plane(s) to fit [default: 0]
                                                    eg. --plane=1 or --plane=1,4,5

        --exclude_plane=<int>                       Specific plane(s) to fit [default: 0]
                                                    eg. --plane=1 or --plane=1,4,5

        --nomp                                      Do not use multiprocessing

        --plot=<dir>                                Whether to plot wireframe fits for each peak
                                                    (saved into <dir>) [default: None]

        --show                                      Whether to show (using plt.show()) wireframe
                                                    fits for each peak. Only works if --plot is also selected

        --verb                                      Print what's going on



    ToDo: change outputs (print/log.txt) so that they do not attempt to write at the same time during multiprocess

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
import json
from pathlib import Path
from multiprocessing import cpu_count, Pool

import nmrglue as ng
import numpy as np
import pandas as pd

from docopt import docopt
from skimage.filters import threshold_otsu
from schema import Schema, And, Or, Use, SchemaError

from peakipy.core import fix_params, get_params, fit_first_plane, Pseudo3D, run_log


def check_xybounds(x):
    x = x.split(",")
    if len(x) == 2:
        xy_bounds = float(x[0]), float(x[1])
        return xy_bounds
    else:
        print("🤔 xy_bounds must be pair of floats e.g. --xy_bounds=0.05,0.5")
        exit()


def split_peaklist(peaklist, n_cpu):
    """ split peaklist into smaller files based on number of cpus"""
    tmp_path = Path("tmp")
    tmp_path.mkdir(exist_ok=True)
    clustids = peaklist.CLUSTID.unique()
    window = int(np.ceil(len(clustids) / n_cpu))
    clustids = [clustids[i : i + window] for i in range(0, len(clustids), window)]
    for i in range(n_cpu):
        split_peaks = peaklist[peaklist.CLUSTID.isin(clustids[i])]
        split_peaks.to_csv(tmp_path / f"peaks_{i}.csv", index=False)
    return tmp_path


class FitPeaksInput:
    def __init__(self, args, data):

        self.data = data
        self.args = args


class FitPeaksResult:
    def __init__(self, df: pd.DataFrame, log: str):

        self.df = df
        self.log = log


def fit_peaks(peaks, fit_input):
    # sum planes for initial fit
    summed_planes = fit_input.data.sum(axis=0)

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

    names = []
    assign = []
    clustids = []
    planes = []
    x_radii = []
    y_radii = []
    x_radii_ppm = []
    y_radii_ppm = []
    lineshapes = []

    if fit_input.args.get("lineshape") == "PV_PV":
        fractions_x = []
        fractions_y = []
    else:
        fractions = []
    # group peaks based on CLUSTID
    groups = peaks.groupby("CLUSTID")
    to_fix = fit_input.args.get("to_fix")
    noise = fit_input.args.get("noise")
    verb = fit_input.args.get("verb")
    lineshape = fit_input.args.get("lineshape")
    xy_bounds = fit_input.args.get("xy_bounds")
    vclist = fit_input.args.get("vclist")
    uc_dics = fit_input.args.get("uc_dics")
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
            )
            fit_result.plot(
                plot_path=fit_input.args.get("plot"), show=fit_input.args.get("--show"), nomp=fit_input.args.get("--nomp")
            )
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

                if lineshape == "PV_PV":
                    frac_x, frac_err_x, name = get_params(first.params, "fraction_x")
                    frac_y, frac_err_y, name = get_params(first.params, "fraction_y")
                    fractions_x.extend(frac_x)
                    fractions_y.extend(frac_y)
                else:
                    frac, frac_err, name = get_params(first.params, "fraction")
                    fractions.extend(frac)

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
                planes.extend([num for _ in amp])
                lineshapes.extend([lineshape for _ in amp])
                #  get prefix for fit
                names.extend([i.replace("fraction", "") for i in name])
                assign.extend(group["ASS"])
                clustids.extend(group["CLUSTID"])
                x_radii.extend(group["X_RADIUS"])
                y_radii.extend(group["Y_RADIUS"])
                x_radii_ppm.extend(group["X_RADIUS_PPM"])
                y_radii_ppm.extend(group["Y_RADIUS_PPM"])

    df_dic = {
        "fit_prefix": names,
        "assignment": assign,
        "amp": amps,
        "amp_err": amp_errs,
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
        "plane": planes,
        "x_radius": x_radii,
        "y_radius": y_radii,
        "x_radius_ppm": x_radii_ppm,
        "y_radius_ppm": y_radii_ppm,
        "lineshape": lineshapes,
    }

    if lineshape == "PV_PV":
        df_dic["fraction_x"] = fractions_x
        df_dic["fraction_y"] = fractions_y
    else:
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


def main(argv):
    # number of CPUs
    n_cpu = cpu_count()

    args = docopt(__doc__, argv=argv)

    schema = Schema(
        {
            "<peaklist>": And(
                os.path.exists,
                open,
                error=f"🤔 {args['<peaklist>']} should exist and be readable",
            ),
            "<data>": And(
                os.path.exists,
                Use(
                    ng.pipe.read,
                    error=f"🤔 {args['<data>']} should be NMRPipe format 2D or 3D cube",
                ),
                error=f"🤔 {args['<data>']} either does not exist or is not an NMRPipe format 2D or 3D",
            ),
            "<output>": Use(str),
            "--max_cluster_size": And(Use(int), lambda n: 0 < n),
            "--lineshape": Or(
                "PV",
                "L",
                "G",
                "PV_PV",
                "PV_G",
                "PV_L",
                "G_L",
                error="🤔 --lineshape must be either PV, L, G, PV_PV, PV_G, PV_L, G_L",
            ),
            "--fix": Or(
                Use(
                    lambda x: [
                        i
                        for i in x.split(",")
                        if (i == "fraction") or (i == "center") or (i == "sigma")
                    ]
                )
            ),
            "--dims": Use(
                lambda n: [int(i) for i in eval(n)],
                error="🤔 --dims should be list of integers e.g. --dims=0,1,2",
            ),
            "--vclist": Or(
                "None",
                And(
                    os.path.exists,
                    Use(np.genfromtxt, error=f"🤔 cannot open {args.get('--vclist')}"),
                ),
            ),
            "--plot": Or("None", Use(lambda f: Path(f))),
            "--xy_bounds": Or(
                "None",
                Use(
                    check_xybounds,
                    error="🤔 xy_bounds must be pair of floats e.g. --xy_bounds=0.05,0.5",
                ),
            ),
            "--plane": Or(
                0,
                Use(
                    lambda n: [int(i) for i in n.split(",")],
                    error="🤔 plane(s) to fit should be an integer or list of integers e.g. --plane=1,2,3,4",
                ),
            ),
            "--exclude_plane": Or(
                0,
                Use(
                    lambda n: [int(i) for i in n.split(",")],
                    error="🤔 plane(s) to exclude should be an integer or list of integers e.g. --exclude_plane=1,2,3,4",
                ),
            ),
            object: object,
        },
        # ignore_extra_keys=True,
    )


    try:
        args = schema.validate(args)
    except SchemaError as e:
        exit(e)

    config_path = Path("peakipy.config")
    if config_path.exists():
        config = json.load(open(config_path))
        print(f"Using config file with --dims={config.get('--dims')}")
        args["--dims"] = config.get("--dims", [0, 1, 2])
        noise = config.get("noise")
        if noise:
            noise = float(noise)
    else:
        noise = False
    args["noise"] = noise

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

    # determine file type
    if peaklist.suffix == ".csv":
        peaks = pd.read_csv(peaklist)  # , comment="#")
    else:
        # assume that file is a pickle
        peaks = pd.read_pickle(peaklist)

    # only include peaks with 'include'
    if "include" in peaks.columns:
        pass
    else:
        # for compatibility
        peaks["include"] = peaks.apply(lambda _: "yes", axis=1)

    if len(peaks[peaks.include != "yes"]) > 0:
        print(f"The following peaks have been exluded:\n{peaks[peaks.include != 'yes']}")
        peaks = peaks[peaks.include == "yes"]

    # filter list based on cluster size
    max_cluster_size = args.get("--max_cluster_size")
    if max_cluster_size == 999:
        max_cluster_size = peaks.MEMCNT.max()
        if peaks.MEMCNT.max() > 10:
            print(
                f"""
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
        log_file = open("log.txt", "w")
    else:
        log_file = open("~log.txt", "w")
        plot.mkdir(parents=True, exist_ok=True)

    args["plot"] = plot

    # get dims from command line input
    dims = args.get("--dims")

    # read NMR data
    dic, data = args["<data>"]

    pseudo3D = Pseudo3D(dic, data, dims)
    uc_f1 = pseudo3D.uc_f1
    uc_f2 = pseudo3D.uc_f2
    uc_dics = {"f1": uc_f1, "f2": uc_f2}
    args["uc_dics"] = uc_dics

    dims = pseudo3D.dims
    data = pseudo3D.data
    if len(dims) != len(data.shape):
        print(f"Dims are {dims} while data shape is {data.shape}?")
        exit()

    if args.get("--plane", [0]) != [0]:
        _inds = args.get("--plane")
        inds = [i - 1 for i in _inds]
        data_inds = [(i in inds) for i in range(data.shape[dims[0]])]
        data = data[data_inds]
        print(f"Using only planes {_inds} data now has the following shape", data.shape)
        if data.shape[dims[0]] == 0:
            print("You have excluded all the data!", data.shape)
            exit()

    if args.get("--exclude_plane", [0]) != [0]:
        _inds = args.get("--exclude_plane")
        inds = [i - 1 for i in _inds]
        data_inds = [(i not in inds) for i in range(data.shape[dims[0]])]
        data = data[data_inds]
        print(f"Excluding planes {_inds} data now has the following shape", data.shape)
        if data.shape[dims[0]] == 0:
            print("You have excluded all the data!", data.shape)
            exit()

    if not noise:
        noise = threshold_otsu(data)

    args["noise"] = noise
    # print(noise)

    # point per Hz
    pt_per_hz_f2 = pseudo3D.pt_per_hz_f2
    pt_per_hz_f1 = pseudo3D.pt_per_hz_f1

    # point per Hz
    hz_per_pt_f2 = 1.0 / pt_per_hz_f2
    hz_per_pt_f1 = 1.0 / pt_per_hz_f1

    # ppm per point
    ppm_per_pt_f2 = pseudo3D.ppm_per_pt_f2
    ppm_per_pt_f1 = pseudo3D.ppm_per_pt_f1

    # point per ppm
    pt_per_ppm_f2 = pseudo3D.pt_per_ppm_f2
    pt_per_ppm_f1 = pseudo3D.pt_per_ppm_f1

    xy_bounds = args.get("--xy_bounds")

    if xy_bounds == "None":
        xy_bounds = None
    else:
        # convert ppm to points
        xy_bounds[0] = xy_bounds[0] * pt_per_ppm_f2
        xy_bounds[1] = xy_bounds[1] * pt_per_ppm_f1

    args["xy_bounds"] = xy_bounds

    # convert linewidths from Hz to points in case they were adjusted when running run_check_fits.py
    peaks["XW"] = peaks.XW_HZ * pt_per_hz_f2
    peaks["YW"] = peaks.YW_HZ * pt_per_hz_f1

    # convert peak positions from ppm to points in case they were adjusted running run_check_fits.py
    peaks["X_AXIS"] = peaks.X_PPM.apply(lambda x: uc_f2(x, "PPM"))
    peaks["Y_AXIS"] = peaks.Y_PPM.apply(lambda x: uc_f1(x, "PPM"))
    peaks["X_AXISf"] = peaks.X_PPM.apply(lambda x: uc_f2.f(x, "PPM"))
    peaks["Y_AXISf"] = peaks.Y_PPM.apply(lambda x: uc_f1.f(x, "PPM"))

    if (peaks.CLUSTID.nunique() >= n_cpu) and not args.get("--nomp"):
        print("Using multiprocessing")
        # split peak lists
        tmp_dir = split_peaklist(peaks, n_cpu)
        peaklists = [pd.read_csv(tmp_dir / f"peaks_{i}.csv") for i in range(n_cpu)]
        args_list = [FitPeaksInput(args, data) for _ in range(n_cpu)]
        with Pool(processes=n_cpu) as pool:
            # result = pool.map(fit_peaks, peaklists)
            result = pool.starmap(fit_peaks, zip(peaklists, args_list))
            df = pd.concat([i.df for i in result], ignore_index=True)
            for num, i in enumerate(result):
                i.df.to_csv(tmp_dir / f"peaks_{num}_fit.csv", index=False)
                log_file.write(i.log + "\n")
    else:
        print("Not using multiprocessing")
        result = fit_peaks(peaks, FitPeaksInput(args, data))
        df = result.df
        log_file.write(result.log)

    # close log file
    log_file.close()
    output = Path(args["<output>"])
    suffix = output.suffix
    #  convert sigmas to fwhm
    df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
    df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)
    #  convert values to ppm
    df["center_x_ppm"] = df.center_x.apply(lambda x: uc_f2.ppm(x))
    df["center_y_ppm"] = df.center_y.apply(lambda x: uc_f1.ppm(x))
    df["init_center_x_ppm"] = df.init_center_x.apply(lambda x: uc_f2.ppm(x))
    df["init_center_y_ppm"] = df.init_center_y.apply(lambda x: uc_f1.ppm(x))
    df["sigma_x_ppm"] = df.sigma_x.apply(lambda x: x * ppm_per_pt_f2)
    df["sigma_y_ppm"] = df.sigma_y.apply(lambda x: x * ppm_per_pt_f1)
    df["fwhm_x_ppm"] = df.fwhm_x.apply(lambda x: x * ppm_per_pt_f2)
    df["fwhm_y_ppm"] = df.fwhm_y.apply(lambda x: x * ppm_per_pt_f1)
    df["fwhm_x_hz"] = df.fwhm_x.apply(lambda x: x * hz_per_pt_f2)
    df["fwhm_y_hz"] = df.fwhm_y.apply(lambda x: x * hz_per_pt_f1)

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
    main(argv)
