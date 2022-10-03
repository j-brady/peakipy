#!/usr/bin/env python3
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
from distutils.command.config import config
import os
import json
import shutil
from pathlib import Path
from enum import Enum
from tabnanny import verbose
from typing import Optional, Tuple, List
from multiprocessing import Pool

import typer
import numpy as np
import nmrglue as ng
import pandas as pd

from rich import print
from rich.table import Table
from skimage.filters import threshold_otsu

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Button

from peakipy.core import (
    Peaklist,
    run_log,
    LoadData,
    read_config,
    pv_pv,
    pvoigt2d,
    voigt2d,
    make_mask,
    pv_g,
    pv_l,
    gaussian_lorentzian,
    Pseudo3D,
)
from .fit import (
    cpu_count,
    fit_peaks,
    FitPeaksInput,
    split_peaklist,
)
from .edit import BokehScript

app = typer.Typer()
tmp_path = Path("tmp")
tmp_path.mkdir(exist_ok=True)
log_path = Path("log.txt")
# for printing dataframes
column_selection = ["INDEX", "ASS", "X_PPM", "Y_PPM", "CLUSTID", "MEMCNT"]
bad_column_selection = [
    "clustid",
    "amp",
    "center_x_ppm",
    "center_y_ppm",
    "fwhm_x_hz",
    "fwhm_y_hz",
    "lineshape",
]
bad_color_selection = [
    "green",
    "blue",
    "yellow",
    "orange",
    "yellow",
    "orange",
    "pink",
]


class PeaklistFormat(Enum):
    a2 = "a2"
    a3 = "a3"
    sparky = "sparky"
    pipe = "pipe"
    peakipy = "peakipy"


class StrucEl(Enum):
    square = "square"
    disk = "disk"
    rectangle = "rectangle"


class OutFmt(Enum):
    csv = "csv"
    pkl = "pkl"


class Lineshape(Enum):
    PV = "PV"
    V = "V"
    G = "G"
    L = "L"
    PV_PV = "PV_PV"


@app.command()
def read(
    peaklist_path: Path,
    data_path: Path,
    peaklist_format: PeaklistFormat,
    thres: Optional[float] = None,
    struc_el: StrucEl = "disk",
    struc_size: Tuple[int, int] = (3, None),  # Tuple[int, Optional[int]] = (3, None),
    x_radius_ppm: float = 0.04,
    y_radius_ppm: float = 0.4,
    x_ppm_column_name: str = "Position F1",
    y_ppm_column_name: str = "Position F2",
    dims: Tuple[int, int, int] = (0, 1, 2),
    outfmt: OutFmt = "csv",
    show: bool = False,
    fuda: bool = False,
):
    """Read NMRPipe/Analysis peaklist into pandas dataframe

    Usage:
        read <peaklist_path> <data_path> (--a2|--a3|--sparky|--pipe|--peakipy) [options]

    Arguments:
        <peaklist_path>                Analysis2/CCPNMRv3(assign)/Sparky/NMRPipe peak list (see below)
        <data_path>                    2D or pseudo3D NMRPipe data

        --a2                      Analysis peaklist as input (tab delimited)
        --a3                      CCPNMR v3 peaklist as input (tab delimited)
        --sparky                  Sparky peaklist as input
        --pipe                    NMRPipe peaklist as input
        --peakipy                 peakipy peaklist.csv or .tab (originally output from peakipy read or edit)

    Options:
        -h --help                 Show this screen
        -v --verb                 Verbose mode
        --version                 Show version

        --thres=<thres>           Threshold for making binary mask that is used for peak clustering [default: None]
                                  If set to None then threshold_otsu from scikit-image is used to determine threshold

        --struc_el=<str>          Structuring element for binary_closing [default: disk]
                                  'square'|'disk'|'rectangle'

        --struc_size=<int,>       Size/dimensions of structuring element [default: 3,]
                                  For square and disk first element of tuple is used (for disk value corresponds to radius).
                                  For rectangle, tuple corresponds to (width,height).

        --f1radius=<float>        F1 radius in ppm for fit mask [default: 0.4]
        --f2radius=<float>        F2 radius in ppm for fit mask [default: 0.04]

        --dims=<planes,F1,F2>     Order of dimensions [default: 0,1,2]

        --posF2=<column_name>     Name of column in Analysis2 peak list containing F2 (i.e. X_PPM)
                                  peak positions [default: "Position F1"]

        --posF1=<column_name>     Name of column in Analysis2 peak list containing F1 (i.e. Y_PPM)
                                  peak positions [default: "Position F2"]

        --outfmt=<csv/pkl>        Format of output peaklist [default: csv]

        --show                    Show the clusters on the spectrum color coded using matplotlib

        --fuda                    Create a parameter file for running fuda (params.fuda)


    Examples:
        peakipy read test.tab test.ft2 --pipe --dims0,1
        peakipy read test.a2 test.ft2 --a2 --thres=1e5  --dims=0,2,1
        peakipy read ccpnTable.tsv test.ft2 --a3 --f1radius=0.3 --f2radius=0.03
        peakipy read test.csv test.ft2 --peakipy --dims=0,1,2

    Description:

       NMRPipe column headers:

           INDEX X_AXIS Y_AXIS DX DY X_PPM Y_PPM X_HZ Y_HZ XW YW XW_HZ YW_HZ X1 X3 Y1 Y3 HEIGHT DHEIGHT VOL PCHI2 TYPE ASS CLUSTID MEMCNT

       Are mapped onto analysis peak list

           'Number', '#', 'Position F1', 'Position F2', 'Sampled None',
           'Assign F1', 'Assign F2', 'Assign F3', 'Height', 'Volume',
           'Line Width F1 (Hz)', 'Line Width F2 (Hz)', 'Line Width F3 (Hz)',
            'Merit', 'Details', 'Fit Method', 'Vol. Method'

       Or sparky peaklist

             Assignment         w1         w2        Volume   Data Height   lw1 (hz)   lw2 (hz)

       Clusters of peaks are selected

    """

    # verbose_mode = args.get("--verb")
    # if verbose_mode:
    #    print("Using arguments:", args)

    clust_args = {
        "struc_el": struc_el,
        "struc_size": struc_size,
    }
    # name of output peaklist
    outname = peaklist_path.stem
    cluster = True

    match peaklist_format.value:
        case "a2":
            # set X and Y ppm column names if not default (i.e. "Position F1" = "X_PPM"
            # "Position F2" = "Y_PPM" ) this is due to Analysis2 often having the
            #  dimension order flipped relative to convention
            peaks = Peaklist(
                peaklist_path,
                data_path,
                fmt="a2",
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
                posF1=y_ppm_column_name,
                posF2=x_ppm_column_name,
            )
            # peaks.adaptive_clusters(block_size=151,offset=0)

        case "a3":
            peaks = Peaklist(
                peaklist_path,
                data_path,
                fmt="a3",
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
            )

        case "sparky":

            peaks = Peaklist(
                peaklist_path,
                data_path,
                fmt="sparky",
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
            )

        case "pipe":
            peaks = Peaklist(
                peaklist_path,
                data_path,
                fmt="pipe",
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
            )

        case "peakipy":
            # read in a peakipy .csv file
            peaks = LoadData(peaklist_path, data_path, fmt="peakipy", dims=dims)
            cluster = False
            # don't overwrite the old .csv file
            outname = outname + "_new"

    peaks.update_df()

    data = peaks.df
    thres = peaks.thres

    if cluster:
        peaks.clusters(thres=thres, **clust_args, l_struc=None)
    else:
        pass

    if fuda:
        peaks.to_fuda()

    # if verbose_mode:
    #    print(data.head())

    match outfmt.value:
        case "csv":
            outname = outname + ".csv"
            data.to_csv(outname, float_format="%.4f", index=False)
        case "pkl":
            outname = outname + ".pkl"
            data.to_pickle(outname)

    # write config file
    config_path = Path("peakipy.config")
    config_kvs = [
        ("dims", dims),
        ("data_path", str(data_path)),
        ("thres", float(thres)),
        ("y_radius_ppm", y_radius_ppm),
        ("x_radius_ppm", x_radius_ppm),
        ("fit_method", "leastsq"),
    ]
    try:
        if config_path.exists():
            with open(config_path) as opened_config:
                config_dic = json.load(opened_config)
                # update values in dict
                config_dic.update(dict(config_kvs))

        else:
            # make a new config
            config_dic = dict(config_kvs)

    except json.decoder.JSONDecodeError:

        print(
            f"Your {config_path} may be corrupted. Making new one (old one moved to {config_path}.bak)"
        )
        shutil.copy(f"{config_path}", f"{config_path}.bak")
        config_dic = dict(config_kvs)

    with open(config_path, "w") as config:
        # write json
        print(config_dic)
        config.write(json.dumps(config_dic, sort_keys=True, indent=4))
        # json.dump(config_dic, fp=config, sort_keys=True, indent=4)

    run_log()

    yaml = f"""
    ##########################################################################################################
    #  This first block is global parameters which can be overridden by adding the desired argument          #
    #  to your list of spectra. One exception is "colors" which if set in global params overrides the        #
    #  color option set for individual spectra as the colors will now cycle through the chosen matplotlib    #
    #  colormap                                                                                              #
    ##########################################################################################################

    cs: {thres}                     # contour start
    contour_num: 10                 # number of contours
    contour_factor: 1.2             # contour factor
    colors: tab20                   # must be matplotlib.cm colormap
    show_cs: True

    outname: ["clusters.pdf","clusters.png"] # either single value or list of output names
    ncol: 1 #  tells matplotlib how many columns to give the figure legend - if not set defaults to 2
    clusters: {outname}
    dims: {dims}

    # Here is where your list of spectra to plot goes
    spectra:

            - fname: {data}
              label: ""
              contour_num: 20
              linewidths: 0.1
    """

    if show:
        with open("show_clusters.yml", "w") as out:
            out.write(yaml)
        os.system("peakipy spec show_clusters.yml")

    print(f"[green]Finished! Use {outname} to run peakipy edit or fit.[/green]")


@app.command()
def fit(
    peaklist_path: Path,
    data_path: Path,
    output_path: Path,
    dims: Tuple[int, int, int] = (0, 1, 2),
    max_cluster_size: int = 999,
    lineshape: Lineshape = "PV",
    fix: List[str] = ["fraction", "sigma", "center"],
    xy_bounds: Tuple[float, float] = (0, 0),
    vclist: Optional[Path] = None,
    plane: Optional[List[int]] = None,
    exclude_plane: Optional[List[int]] = None,
    nomp: bool = False,
    plot: Optional[Path] = None,
    show: bool = False,
    verb: bool = False,
):
    """ Fit NMR data to lineshape models and deconvolute overlapping peaks

    Arguments:
        <peaklist_path>                                  peaklist output from read_peaklist.py
        <data_path>                                      2D or pseudo3D NMRPipe data (single file)
        <output_path>                                    output peaklist "<output>.csv" will output CSV
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
    """
    # number of CPUs
    n_cpu = cpu_count()

    # read NMR data
    args = {}
    config = {}
    peakipy_data = LoadData(peaklist_path, data_path, dims=dims)

    # only include peaks with 'include'
    if "include" in peakipy_data.df.columns:
        pass
    else:
        # for compatibility
        peakipy_data.df["include"] = peakipy_data.df.apply(lambda _: "yes", axis=1)

    if len(peakipy_data.df[peakipy_data.df.include != "yes"]) > 0:
        excluded = peakipy_data.df[peakipy_data.df.include != "yes"][column_selection]
        table = Table("[yellow] Excluded peaks [/yellow]")
        for i in column_selection:
            table.add_column(i)
            for row in excluded[i].values:
                table.add_row(row)

        print(table)

        peakipy_data.df = peakipy_data.df[peakipy_data.df.include == "yes"]

    # filter list based on cluster size
    if max_cluster_size == 999:
        max_cluster_size = peakipy_data.df.MEMCNT.max()
        if peakipy_data.df.MEMCNT.max() > 10:
            print(
                f"""[red]
                ##################################################################
                You have some clusters of as many as {max_cluster_size} peaks.
                You may want to consider reducing the size of your clusters as the
                fits will struggle.

                Otherwise you can use the --max_cluster_size flag to exclude large
                clusters
                ##################################################################
            [/red]"""
            )
    else:
        max_cluster_size = max_cluster_size
    args["max_cluster_size"] = max_cluster_size
    args["to_fix"] = fix
    args["verb"] = verb

    # read vclist
    if vclist is None:
        vclist = False
    elif vclist.exists():
        vclist_data = np.genfromtxt(vclist)
        args["vclist_data"] = vclist_data
        vclist = True
    else:
        raise Exception("vclist not found...")

    args["vclist"] = vclist

    # plot results or not
    log_file = open(tmp_path / log_path, "w")
    if plot:
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

    # setting noise for calculation of chi square
    # if noise is None:
    noise = abs(threshold_otsu(peakipy_data.data))
    args["noise"] = noise
    args["lineshape"] = lineshape.value

    match xy_bounds:
        case (0, 0):
            xy_bounds = None
        case (x, y):
            # convert ppm to points
            xy_bounds[0] = xy_bounds[0] * peakipy_data.pt_per_ppm_f2
            xy_bounds[1] = xy_bounds[1] * peakipy_data.pt_per_ppm_f1

    args["xy_bounds"] = xy_bounds
    # args, config = read_config(args)
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
    if (peakipy_data.df.CLUSTID.nunique() >= n_cpu) and not nomp:
        print("[green]Using multiprocessing[/green]")
        # split peak lists
        tmp_dir = split_peaklist(peakipy_data.df, n_cpu)
        peaklists = [
            pd.read_csv(tmp_dir / Path(f"peaks_{i}.csv")) for i in range(n_cpu)
        ]
        args_list = [
            FitPeaksInput(args, peakipy_data.data, config, plane_numbers)
            for _ in range(n_cpu)
        ]
        with Pool(processes=n_cpu) as pool:
            # result = pool.map(fit_peaks, peaklists)
            result = pool.starmap(fit_peaks, zip(peaklists, args_list))
            df = pd.concat([i.df for i in result], ignore_index=True)
            for num, i in enumerate(result):
                i.df.to_csv(tmp_dir / Path(f"peaks_{num}_fit.csv"), index=False)
                log_file.write(i.log + "\n")
    else:
        print("[green]Not using multiprocessing[green]")
        result = fit_peaks(
            peakipy_data.df,
            FitPeaksInput(args, peakipy_data.data, config, plane_numbers),
        )
        df = result.df
        log_file.write(result.log)

    # finished fitting

    # close log file
    log_file.close()
    output = Path(output_path)
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
            lambda x: 0.5346 * x.fwhm_l_x
            + np.sqrt(0.2166 * x.fwhm_l_x**2.0 + x.fwhm_g_x**2.0),
            axis=1,
        )
        df["fwhm_y"] = df.apply(
            lambda x: 0.5346 * x.fwhm_l_y
            + np.sqrt(0.2166 * x.fwhm_l_y**2.0 + x.fwhm_g_y**2.0),
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
        """[green]
           🍾 ✨ Finished! ✨ 🍾
           [/green]       
        """
    )
    run_log()


def df_to_table(df, title: str, columns: List[str], styles: str):
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
    for ind, row in df.iterrows():
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


@app.command()
def check(
    fits: Path,
    data_path: Path,
    dims: Tuple[int, int, int] = (0, 1, 2),
    clusters: Optional[List[int]] = None,
    plane: int = 0,
    outname: Path = "plots.pdf",
    first: bool = False,
    show: bool = False,
    label: bool = False,
    individual: bool = False,
    ccpn: bool = False,
    rcount: int = 50,
    ccount: int = 50,
    colors: Tuple[str, str] = ("#5e3c99", "#e66101"),
    verb: bool = False,
):
    """
    Usage:
         check <fits> <nmrdata> [options]

     Options:
         --dims=<id,f1,f2>         Dimension order [default: 0,1,2]

         --clusters=<id1,id2,etc>  Plot selected cluster based on clustid [default: None]
                                   e.g. --clusters=1 or --clusters=2,4,6,7
         --plane=<int>             Plot selected plane [default: 0]
                                   e.g. --plane=2 will plot second plane only

         --outname=<plotname>      Plot name [default: plots.pdf]

         --first, -f               Only plot first plane (overrides --plane option)
         --show, -s                Invoke plt.show() for interactive plot
         --individual, -i          Plot individual fitted peaks as surfaces with different colors
         --label, -l               Label individual peaks
         --ccpn


         --rcount=<int>            row count setting for wireplot [default: 50]
         --ccount=<int>            column count setting for wireplot [default: 50]
         --colors=<data,fit>       plot colors [default: #5e3c99,#e66101]"""
    columns_to_print = [
        "assignment",
        "clustid",
        "memcnt",
        "plane",
        "amp",
        "height",
        "center_x_ppm",
        "center_y_ppm",
        "fwhm_x_hz",
        "fwhm_y_hz",
        "lineshape",
    ]
    fits = pd.read_csv(fits)
    args = {}
    # get dims from config file
    config_path = Path("peakipy.config")
    args, config = read_config(args, config_path)

    ccpn_flag = args.get("--ccpn")
    if ccpn_flag:
        from ccpn.ui.gui.widgets.PlotterWidget import PlotterWidget
    else:
        pass
    dic, data = ng.pipe.read(data_path)
    pseudo3D = Pseudo3D(dic, data, dims)

    # first only overrides plane option
    if first:
        plane = 0
    else:
        plane = plane

    if plane > pseudo3D.n_planes:
        raise ValueError(
            f"[red]There are {pseudo3D.n_planes} planes in your data you selected --plane {plane}...[red]"
            f"plane numbering starts from 0."
        )
    elif plane < 0:
        raise ValueError(
            f"[red]Plane number can not be negative; you selected --plane {plane}...[/red]"
        )
    # in case first plane is chosen
    elif plane == 0:
        selected_plane = plane
    # plane numbers start from 1 so adjust for indexing
    else:
        selected_plane = plane
        # fits = fits[fits["plane"] == plane]
        # print(fits)

    if type(ccount) == int:
        ccount = ccount
    else:
        raise TypeError("ccount should be an integer")

    if type(rcount) == int:
        rcount = rcount
    else:
        raise TypeError("rcount should be an integer")

    match colors:
        case (data_color, fit_color):
            data_color, fit_color = colors
        case _:
            data_color, fit_color = "green", "blue"

        # raise TypeError(
        # "colors should be valid pair for matplotlib. i.e. g,b or green,blue"
        # )

    match clusters:
        case None | []:
            pass
        case _:
            # only use these clusters
            fits = fits[fits.clustid.isin(clusters)]
            if len(fits) < 1:
                exit(f"Are you sure clusters {clusters} exist?")

    groups = fits.groupby("clustid")

    # make plotting meshes
    x = np.arange(pseudo3D.f2_size)
    y = np.arange(pseudo3D.f1_size)
    XY = np.meshgrid(x, y)
    X, Y = XY

    with PdfPages(outname) as pdf:

        for name, group in groups:
            table = df_to_table(
                group,
                title="",
                columns=columns_to_print,
                styles=["blue" for _ in columns_to_print],
            )
            print(table)
            # print(
            #    Fore.BLE
            #    + tabulate(
            #        group[columns_to_print],
            #        showindex=False,
            #        tablefmt="fancy_grid",
            #        headers="keys",
            #        floatfmt=".3f",
            #    )
            # )

            mask = np.zeros((pseudo3D.f1_size, pseudo3D.f2_size), dtype=bool)

            first_plane = group[group.plane == selected_plane]

            x_radius = group.x_radius.max()
            y_radius = group.y_radius.max()
            max_x, min_x = (
                int(np.ceil(max(group.center_x) + x_radius + 1)),
                int(np.floor(min(group.center_x) - x_radius)),
            )
            max_y, min_y = (
                int(np.ceil(max(group.center_y) + y_radius + 1)),
                int(np.floor(min(group.center_y) - y_radius)),
            )

            #  deal with peaks on the edge of spectrum
            if min_y < 0:
                min_y = 0

            if min_x < 0:
                min_x = 0

            if max_y > pseudo3D.f1_size:
                max_y = pseudo3D.f1_size

            if max_x > pseudo3D.f2_size:
                max_x = pseudo3D.f2_size

            masks = []
            # make masks
            for cx, cy, rx, ry, name in zip(
                first_plane.center_x,
                first_plane.center_y,
                first_plane.x_radius,
                first_plane.y_radius,
                first_plane.assignment,
            ):

                tmp_mask = make_mask(mask, cx, cy, rx, ry)
                mask += tmp_mask
                masks.append(tmp_mask)

            # generate simulated data
            for plane_id, plane in group.groupby("plane"):
                sim_data_singles = []
                sim_data = np.zeros((pseudo3D.f1_size, pseudo3D.f2_size))
                shape = sim_data.shape
                try:
                    for amp, c_x, c_y, s_x, s_y, frac_x, frac_y, ls in zip(
                        plane.amp,
                        plane.center_x,
                        plane.center_y,
                        plane.sigma_x,
                        plane.sigma_y,
                        plane.fraction_x,
                        plane.fraction_y,
                        plane.lineshape,
                    ):

                        sim_data_i = pv_pv(
                            XY, amp, c_x, c_y, s_x, s_y, frac_x, frac_y
                        ).reshape(shape)
                        sim_data += sim_data_i
                        sim_data_singles.append(sim_data_i)
                except:
                    for amp, c_x, c_y, s_x, s_y, frac, ls in zip(
                        plane.amp,
                        plane.center_x,
                        plane.center_y,
                        plane.sigma_x,
                        plane.sigma_y,
                        plane.fraction,
                        plane.lineshape,
                    ):
                        # print(amp)
                        match ls:
                            case "G" | "L" | "PV":
                                sim_data_i = pvoigt2d(
                                    XY, amp, c_x, c_y, s_x, s_y, frac
                                ).reshape(shape)
                            case "PV_L":
                                sim_data_i = pv_l(
                                    XY, amp, c_x, c_y, s_x, s_y, frac
                                ).reshape(shape)

                            case "PV_G":
                                sim_data_i = pv_g(
                                    XY, amp, c_x, c_y, s_x, s_y, frac
                                ).reshape(shape)

                            case "G_L":
                                sim_data_i = gaussian_lorentzian(
                                    XY, amp, c_x, c_y, s_x, s_y, frac
                                ).reshape(shape)

                            case "V":
                                sim_data_i = voigt2d(
                                    XY, amp, c_x, c_y, s_x, s_y, frac
                                ).reshape(shape)
                        sim_data += sim_data_i
                        sim_data_singles.append(sim_data_i)

                masked_data = pseudo3D.data[plane_id].copy()
                masked_sim_data = sim_data.copy()
                masked_data[~mask] = np.nan
                masked_sim_data[~mask] = np.nan

                if ccpn_flag:
                    plt = PlotterWidget()
                else:
                    plt = matplotlib.pyplot

                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection="3d")
                # slice out plot area
                x_plot = pseudo3D.uc_f2.ppm(X[min_y:max_y, min_x:max_x])
                y_plot = pseudo3D.uc_f1.ppm(Y[min_y:max_y, min_x:max_x])
                masked_data = masked_data[min_y:max_y, min_x:max_x]
                sim_plot = masked_sim_data[min_y:max_y, min_x:max_x]
                # or len(masked_data)<1 or len(sim_plot)<1

                if len(x_plot) < 1 or len(y_plot) < 1:
                    print(
                        f"[red]Nothing to plot for cluster {int(plane.clustid)}[/red]"
                    )
                    print(f"[red]x={x_plot},y={y_plot}[/red]")
                    print(print_bad(plane))
                    plt.close()
                    # print(Fore.RED + "Maybe your F1/F2 radii for fitting were too small...")
                elif masked_data.shape[0] == 0 or masked_data.shape[1] == 0:
                    print(
                        f"[red]Nothing to plot for cluster {int(plane.clustid)}[/red]"
                    )
                    print(
                        df_to_table(
                            plane,
                            title="Bad plane",
                            columns=bad_column_selection,
                            styles=bad_color_selection,
                        )
                    )
                    spec_lim_f1 = " - ".join(
                        ["%8.3f" % i for i in pseudo3D.f1_ppm_limits]
                    )
                    spec_lim_f2 = " - ".join(
                        ["%8.3f" % i for i in pseudo3D.f2_ppm_limits]
                    )
                    print(
                        f"Spectrum limits are {pseudo3D.f2_label:4s}:{spec_lim_f2} ppm"
                    )
                    print(
                        f"                    {pseudo3D.f1_label:4s}:{spec_lim_f1} ppm"
                    )
                    plt.close()
                else:

                    residual = masked_data - sim_plot
                    cset = ax.contourf(
                        x_plot,
                        y_plot,
                        residual,
                        zdir="z",
                        offset=np.nanmin(masked_data) * 1.1,
                        alpha=0.5,
                        cmap=cm.coolwarm,
                    )
                    fig.colorbar(cset, ax=ax, shrink=0.5, format="%.2e")

                    if individual:
                        # for making colored masks
                        for single_mask, single in zip(masks, sim_data_singles):
                            single[~single_mask] = np.nan
                        sim_data_singles = [
                            sim_data_single[min_y:max_y, min_x:max_x]
                            for sim_data_single in sim_data_singles
                        ]
                        #  for plotting single fit surfaces
                        single_colors = [
                            cm.viridis(i)
                            for i in np.linspace(0, 1, len(sim_data_singles))
                        ]
                        [
                            ax.plot_surface(
                                x_plot, y_plot, z_single, color=c, alpha=0.5
                            )
                            for c, z_single in zip(single_colors, sim_data_singles)
                        ]

                    ax.plot_wireframe(
                        x_plot,
                        y_plot,
                        sim_plot,
                        # colors=[cm.coolwarm(i) for i in np.ravel(residual)],
                        colors=fit_color,
                        linestyle="--",
                        label="fit",
                        rcount=rcount,
                        ccount=ccount,
                    )
                    ax.plot_wireframe(
                        x_plot,
                        y_plot,
                        masked_data,
                        colors=data_color,
                        linestyle="-",
                        label="data",
                        rcount=rcount,
                        ccount=ccount,
                    )
                    ax.set_ylabel(pseudo3D.f1_label)
                    ax.set_xlabel(pseudo3D.f2_label)

                    # axes will appear inverted
                    ax.view_init(30, 120)

                    # names = ",".join(plane.assignment)
                    title = f"Plane={plane_id},Cluster={plane.clustid.iloc[0]}"
                    plt.title(title)
                    print(f"[green]Plotting: {title}[/green]")
                    out_str = "Volumes (Heights)\n----------------\n"
                    # chi2s = []
                    for ind, row in plane.iterrows():

                        out_str += (
                            f"{row.assignment} = {row.amp:.3e} ({row.height:.3e})\n"
                        )
                        if label:
                            ax.text(
                                row.center_x_ppm,
                                row.center_y_ppm,
                                row.height * 1.2,
                                row.assignment,
                            )

                    ax.text2D(
                        -0.15,
                        1.0,
                        out_str,
                        transform=ax.transAxes,
                        fontsize=10,
                        va="top",
                        bbox=dict(boxstyle="round", ec="k", fc="none"),
                    )

                    ax.legend()
                    pdf.savefig()

                    if show:

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
                        if ccpn_flag:
                            plt.show(windowTitle="", size=(1000, 500))
                        else:
                            plt.show()

                    plt.close()

                    if first:
                        break
    run_log()


@app.command()
def edit(
    peaklist_path: Path,
    data_path: Path,
    dims: Tuple[int, int, int] = (0, 1, 2),
):
    from bokeh.util.browser import view
    from bokeh.server.server import Server

    run_log()
    bs = BokehScript(peaklist_path=peaklist_path, data_path=data_path, dims=dims)
    server = Server({"/edit": bs.init})
    server.start()
    print("[green]Opening peakipy: Edit fits on http://localhost:5006/edit[/green]")
    server.io_loop.add_callback(server.show, "/edit")
    server.io_loop.start()


if __name__ == "__main__":
    app()
