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
import os
import json
import shutil
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple, List
from multiprocessing import Pool

import typer
import numpy as np
import nmrglue as ng
import pandas as pd

from rich import print
from skimage.filters import threshold_otsu

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Button
import yaml

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
    df_to_rich_table,
    StrucEl,
    PeaklistFormat,
    Lineshape,
    OutFmt,
)
from .fit import (
    cpu_count,
    fit_peaks,
    FitPeaksInput,
    split_peaklist,
)
from .edit import BokehScript
from .spec import yaml_file

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
    "red",
    "yellow",
    "red",
    "magenta",
]




@app.command(help="Read NMRPipe/Analysis peaklist into pandas dataframe")
def read(
    peaklist_path: Path,
    data_path: Path,
    peaklist_format: PeaklistFormat,
    thres: Optional[float] = None,
    struc_el: StrucEl = StrucEl.disk,
    struc_size: Tuple[int, int] = (3, None),  # Tuple[int, Optional[int]] = (3, None),
    x_radius_ppm: float = 0.04,
    y_radius_ppm: float = 0.4,
    x_ppm_column_name: str = "Position F1",
    y_ppm_column_name: str = "Position F2",
    dims: List[int] = [0, 1, 2],
    outfmt: OutFmt = OutFmt.csv,
    show: bool = False,
    fuda: bool = False,
):
    """Read NMRPipe/Analysis peaklist into pandas dataframe


    Parameters
    ----------
    peaklist_path : Path
        Analysis2/CCPNMRv3(assign)/Sparky/NMRPipe peak list (see below)
    data_path : Path
        2D or pseudo3D NMRPipe data
    peaklist_format : PeaklistFormat
        a2 - Analysis peaklist as input (tab delimited)
        a3 - CCPNMR v3 peaklist as input (tab delimited)
        sparky - Sparky peaklist as input
        pipe - NMRPipe peaklist as input
        peakipy - peakipy peaklist.csv or .tab (originally output from peakipy read or edit)

    thres : Optional[float]
        Threshold for making binary mask that is used for peak clustering [default: None]
        If set to None then threshold_otsu from scikit-image is used to determine threshold
    struc_el : StrucEl
        Structuring element for binary_closing [default: disk]
        'square'|'disk'|'rectangle'
    struc_size : Tuple[int, int]
        Size/dimensions of structuring element [default: 3, None]
        For square and disk first element of tuple is used (for disk value corresponds to radius).
        For rectangle, tuple corresponds to (width,height).
    x_radius_ppm : float
        F2 radius in ppm for fit mask [default: 0.04]
    y_radius_ppm : float
        F1 radius in ppm for fit mask [default: 0.4]
    dims : Tuple[int]
        <planes,y,x>
        Order of dimensions [default: 0,1,2]
    posF2 : str
        Name of column in Analysis2 peak list containing F2 (i.e. X_PPM)
        peak positions [default: "Position F1"]
    posF1 : str
        Name of column in Analysis2 peak list containing F1 (i.e. Y_PPM)
        peak positions [default: "Position F2"]
    outfmt : OutFmt
        Format of output peaklist [default: csv]
    show : bool
        Show the clusters on the spectrum color coded using matplotlib
    fuda : bool
        Create a parameter file for running fuda (params.fuda)


    Examples
    --------
        peakipy read test.tab test.ft2 pipe --dims 0 --dims 1
        peakipy read test.a2 test.ft2 a2 --thres 1e5  --dims 0 --dims 2 --dims 1
        peakipy read ccpnTable.tsv test.ft2 a3 --y_radius 0.3 --x_radius 0.03
        peakipy read test.csv test.ft2 peakipy --dims 0 1 2

    Description
    -----------

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

    match peaklist_format:
        case peaklist_format.a2:
            # set X and Y ppm column names if not default (i.e. "Position F1" = "X_PPM"
            # "Position F2" = "Y_PPM" ) this is due to Analysis2 often having the
            #  dimension order flipped relative to convention
            peaks = Peaklist(
                peaklist_path,
                data_path,
                fmt=PeaklistFormat.a2,
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
                posF1=y_ppm_column_name,
                posF2=x_ppm_column_name,
            )
            # peaks.adaptive_clusters(block_size=151,offset=0)

        case peaklist_format.a3:
            peaks = Peaklist(
                peaklist_path,
                data_path,
                fmt=PeaklistFormat.a3,
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
            )

        case peaklist_format.sparky:

            peaks = Peaklist(
                peaklist_path,
                data_path,
                fmt=PeaklistFormat.sparky,
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
            )

        case peaklist_format.pipe:
            peaks = Peaklist(
                peaklist_path,
                data_path,
                fmt=PeaklistFormat.pipe,
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
            )

        case peaklist_format.peakipy:
            # read in a peakipy .csv file
            peaks = LoadData(peaklist_path, data_path, fmt=PeaklistFormat.peakipy, dims=dims)
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

            - fname: {data_path}
              label: ""
              contour_num: 20
              linewidths: 0.5
    """

    if show:
        with open("show_clusters.yml", "w") as out:
            out.write(yaml)
        os.system("peakipy spec show_clusters.yml")

    print(f"[green]Finished! Use {outname} to run peakipy edit or fit.[/green]")


@app.command(help="Fit NMR data to lineshape models and deconvolute overlapping peaks")
def fit(
    peaklist_path: Path,
    data_path: Path,
    output_path: Path,
    max_cluster_size: Optional[int] = None,
    lineshape: Lineshape = Lineshape.PV,
    fix: List[str] = ["fraction", "sigma", "center"],
    xy_bounds: Tuple[float, float] = (0, 0),
    vclist: Optional[Path] = None,
    plane: Optional[List[int]] = None,
    exclude_plane: Optional[List[int]] = None,
    mp: bool = True,
    plot: Optional[Path] = None,
    show: bool = False,
    verb: bool = False,
):
    """Fit NMR data to lineshape models and deconvolute overlapping peaks

    Parameters
    ----------
    peaklist_path : Path
        peaklist output from read_peaklist.py
    data_path : Path
        2D or pseudo3D NMRPipe data (single file)
    output_path : Path
        output peaklist "<output>.csv" will output CSV
        format file, "<output>.tab" will give a tab delimited output
        while "<output>.pkl" results in Pandas pickle of DataFrame
    max_cluster_size : int
        Maximum size of cluster to fit (i.e exclude large clusters) [default: None]
    lineshape : Lineshape
        Lineshape to fit [default: Lineshape.PV]
    fix : List[str] 
        <fraction,sigma,center>
        Parameters to fix after initial fit on summed planes [default: fraction,sigma,center]
    xy_bounds : Tuple[float,float]
        <x_ppm,y_ppm>
        Bound X and Y peak centers during fit [default: (0,0) which means no bounding]
        This can be set like so --xy_bounds=0.1,0.5
    vclist : Optional[Path]
        Bruker style vclist [default: None]
    plane : Optional[List[int]]
        Specific plane(s) to fit [default: None]
        eg. [1,4,5] will use only planes 1, 4 and 5
    exclude_plane : Optional[List[int]]
        Specific plane(s) to fit [default: None]
        eg. [1,4,5] will exclude planes 1, 4 and 5
    mp : bool
        Use multiprocessing [default: True]
    plot : Optional[Path]
        Whether to plot wireframe fits for each peak
        (saved into Path provided) [default: None]
    show : bool
        Whether to show (using plt.show()) wireframe
        fits for each peak. Only works if Path is provided to the plot
        argument
    verb : bool
        Print what's going on
    """
    # number of CPUs
    n_cpu = cpu_count()

    # read NMR data
    args = {}
    config = {}
    args, config = read_config(args)
    dims = config.get("dims", [0, 1, 2])
    peakipy_data = LoadData(peaklist_path, data_path, dims=dims)

    # only include peaks with 'include'
    if "include" in peakipy_data.df.columns:
        pass
    else:
        # for compatibility
        peakipy_data.df["include"] = peakipy_data.df.apply(lambda _: "yes", axis=1)

    if len(peakipy_data.df[peakipy_data.df.include != "yes"]) > 0:
        excluded = peakipy_data.df[peakipy_data.df.include != "yes"][column_selection]
        table = df_to_rich_table(
            excluded,
            title="[yellow] Excluded peaks [/yellow]",
            columns=excluded.columns,
            styles=["yellow" for i in excluded.columns],
        )
        print(table)
        peakipy_data.df = peakipy_data.df[peakipy_data.df.include == "yes"]

    # filter list based on cluster size
    if max_cluster_size is None:
        max_cluster_size = peakipy_data.df.MEMCNT.max()
        if peakipy_data.df.MEMCNT.max() > 10:
            print(
                f"""[red]
                ##################################################################
                You have some clusters of as many as {max_cluster_size} peaks.
                You may want to consider reducing the size of your clusters as the
                fits will struggle.

                Otherwise you can use the --max-cluster-size flag to exclude large
                clusters
                ##################################################################
            [/red]"""
            )
    else:
        max_cluster_size = max_cluster_size
    args["max_cluster_size"] = max_cluster_size
    args["to_fix"] = fix
    args["verb"] = verb
    args["show"] = show
    args["mp"] = mp

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
    args["lineshape"] = lineshape

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
    if (peakipy_data.df.CLUSTID.nunique() >= n_cpu) and mp:
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
    match lineshape:
        case lineshape.V:
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

        case lineshape.PV:
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

        case lineshape.G:
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

        case lineshape.L:
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

        case lineshape.PV_PV:
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

        case _:
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


@app.command(help="Interactive plots for checking fits")
def check(
    fits: Path,
    data_path: Path,
    clusters: Optional[List[int]] = None,
    plane: int = 0,
    outname: Path = Path("plots.pdf"),
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
    """Interactive plots for checking fits

    Parameters
    ----------
    fits : Path
    data_path : Path
    clusters : Optional[List[int]]
        <id1,id2,etc> 
        Plot selected cluster based on clustid [default: None]
        e.g. clusters=[2,4,6,7] 
    plane : int
        Plot selected plane [default: 0]
        e.g. plane=2 will plot second plane only
    outname : Path
        Plot name [default: Path("plots.pdf")]
    first : bool
        Only plot first plane (overrides --plane option)
    show : bool 
        Invoke plt.show() for interactive plot
    individual : bool
        Plot individual fitted peaks as surfaces with different colors
    label : bool
        Label individual peaks
    ccpn : bool
        for use in ccpnmr
    rcount : int
        row count setting for wireplot [default: 50]
    ccount : int
        column count setting for wireplot [default: 50]
    colors : Tuple[str,str]
        <data,fit> 
        plot colors [default: #5e3c99,#e66101]
    verb : bool
        verbose mode 
    """
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
    dims = config.get("dims", (1, 2, 3))

    ccpn_flag = ccpn
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
            table = df_to_rich_table(
                group,
                title="",
                columns=columns_to_print,
                styles=["blue" for _ in columns_to_print],
            )
            print(table)

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
                    print(
                        df_to_rich_table(
                            plane,
                            title="",
                            columns=bad_column_selection,
                            styles=bad_color_selection,
                        )
                    )
                    plt.close()
                    # print(Fore.RED + "Maybe your F1/F2 radii for fitting were too small...")
                elif masked_data.shape[0] == 0 or masked_data.shape[1] == 0:
                    print(
                        f"[red]Nothing to plot for cluster {int(plane.clustid)}[/red]"
                    )
                    print(
                        df_to_rich_table(
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
                    cbl = fig.colorbar(cset, ax=ax, shrink=0.5, format="%.2e")
                    cbl.ax.set_title("Residual", pad=20)

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
                    out_str = "Volumes (Heights)\n===========\n"
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
                                (1,1,1),
                            )

                    ax.text2D(
                        -0.5,
                        1.0,
                        out_str,
                        transform=ax.transAxes,
                        fontsize=10,
                        fontfamily="sans-serif",
                        va="top",
                        bbox=dict(boxstyle="round", ec="k", fc="k",alpha=0.5),
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


@app.command(help="Interactive Bokeh dashboard for configuring fitting parameters")
def edit(
    peaklist_path: Path,
    data_path: Path,
):
    from bokeh.util.browser import view
    from bokeh.server.server import Server

    run_log()
    bs = BokehScript(peaklist_path=peaklist_path, data_path=data_path)
    server = Server({"/edit": bs.init})
    server.start()
    print("[green]Opening peakipy: Edit fits on http://localhost:5006/edit[/green]")
    server.io_loop.add_callback(server.show, "/edit")
    server.io_loop.start()


def make_yaml_file(name, yaml_file=yaml_file):

    if os.path.exists(name):
        print(f"Copying {name} to {name}.bak")
        shutil.copy(name, f"{name}.bak")

    print(f"Making yaml file ... {name}")
    with open(name, "w") as new_yaml_file:
        new_yaml_file.write(yaml_file)


@app.command(help="Show first plane with clusters")
def spec(yaml_file: Path, new: bool = False):
    if new:
        make_yaml_file(name=yaml_file)
        exit()

    if yaml_file.exists():
        params = yaml.load(open(yaml_file, "r"), Loader=yaml.FullLoader)
    else:
        print(
            f"[red]{yaml_file} does not exist. Use 'peakipy spec <yaml_file> --new' to create one[/red]"
        )
        exit()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cs_g = float(params["cs"])
    spectra = params["spectra"]
    contour_num_g = params.get("contour_num", 10)
    contour_factor_g = params.get("contour_factor", 1.2)
    nspec = len(spectra)
    notes = []
    legends = 0
    for num, spec in enumerate(spectra):

        # unpack spec specific parameters
        fname = spec["fname"]

        if params.get("colors"):
            # currently overrides color option
            color = np.linspace(0, 1, nspec)[num]
            colors = cm.get_cmap(params.get("colors"))(color)
            # print("Colors set to cycle though %s from Matplotlib"%params.get("colors"))
            # print(colors)
            colors = colors[:-1]

        else:
            colors = spec["colors"]

        neg_colors = spec.get("neg_colors")
        label = spec.get("label")
        cs = float(spec.get("cs", cs_g))
        contour_num = spec.get("contour_num", contour_num_g)
        contour_factor = spec.get("contour_factor", contour_factor_g)
        #  append cs and colors to notes
        notes.append((cs, colors))

        # read spectra
        dic, data = ng.pipe.read(fname)
        udic = ng.pipe.guess_udic(dic, data)

        ndim = udic["ndim"]

        if ndim == 1:
            uc_f1 = ng.pipe.make_uc(dic, data, dim=0)

        elif ndim == 2:
            f1, f2 = params.get("dims", [0, 1])
            uc_f1 = ng.pipe.make_uc(dic, data, dim=f1)
            uc_f2 = ng.pipe.make_uc(dic, data, dim=f2)

            ppm_f1 = uc_f1.ppm_scale()
            ppm_f2 = uc_f2.ppm_scale()

            ppm_f1_0, ppm_f1_1 = uc_f1.ppm_limits()  # max,min
            ppm_f2_0, ppm_f2_1 = uc_f2.ppm_limits()  # max,min

        elif ndim == 3:
            dims = params.get("dims", [0, 1, 2])
            f1, f2, f3 = dims
            uc_f1 = ng.pipe.make_uc(dic, data, dim=f1)
            uc_f2 = ng.pipe.make_uc(dic, data, dim=f2)
            uc_f3 = ng.pipe.make_uc(dic, data, dim=f3)
            #  need to make more robust
            ppm_f1 = uc_f2.ppm_scale()
            ppm_f2 = uc_f3.ppm_scale()

            ppm_f1_0, ppm_f1_1 = uc_f2.ppm_limits()  # max,min
            ppm_f2_0, ppm_f2_1 = uc_f3.ppm_limits()  # max,min

            # if f1 == 0:
            #    data = data[f1]
            if dims != [1, 2, 3]:
                data = np.transpose(data, dims)
            data = data[0]
            # x and y are set to f2 and f1
            f1, f2 = f2, f3
            # elif f1 == 1:
            #    data = data[:,0,:]
            # else:
            #    data = data[:,:,0]

        # plot parameters
        contour_start = cs  # contour level start value
        contour_num = contour_num  # number of contour levels
        contour_factor = contour_factor  # scaling factor between contour levels

        # calculate contour levels
        cl = contour_start * contour_factor ** np.arange(contour_num)
        if len(cl) > 1 and np.min(np.diff(cl)) <= 0.0:
            print(f"Setting contour levels to np.abs({cl})")
            cl = np.abs(cl)

        ax.contour(
            data,
            cl,
            colors=[colors for _ in cl],
            linewidths=spec.get("linewidths", 0.5),
            extent=(ppm_f2_0, ppm_f2_1, ppm_f1_0, ppm_f1_1),
        )

        if neg_colors:
            ax.contour(
                data * -1,
                cl,
                colors=[neg_colors for _ in cl],
                linewidths=spec.get("linewidths", 0.5),
                extent=(ppm_f2_0, ppm_f2_1, ppm_f1_0, ppm_f1_1),
            )

        else:  # if no neg color given then plot with 0.5 alpha
            ax.contour(
                data * -1,
                cl,
                colors=[colors for _ in cl],
                linewidths=spec.get("linewidths", 0.5),
                extent=(ppm_f2_0, ppm_f2_1, ppm_f1_0, ppm_f1_1),
                alpha=0.5,
            )

        # make legend
        if label:
            legends += 1
            # hack for legend
            ax.plot([], [], c=colors, label=label)

    # plt.xlim(ppm_f2_0, ppm_f2_1)
    ax.invert_xaxis()
    ax.set_xlabel(udic[f2]["label"] + " ppm")
    if params.get("xlim"):
        ax.set_xlim(*params.get("xlim"))

    # plt.ylim(ppm_f1_0, ppm_f1_1)
    ax.invert_yaxis()
    ax.set_ylabel(udic[f1]["label"] + " ppm")

    if legends > 0:
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=params.get("ncol", 2)
        )

    plt.tight_layout()

    #  add a list of outfiles
    y = 0.025
    # only write cs levels if show_cs: True in yaml file
    if params.get("show_cs"):
        for num, j in enumerate(notes):
            col = j[1]
            con_strt = j[0]
            ax.text(0.025, y, "cs=%.2e" % con_strt, color=col, transform=ax.transAxes)
            y += 0.05

    if params.get("clusters"):

        peaklist = params.get("clusters")
        if os.path.splitext(peaklist)[-1] == ".csv":
            clusters = pd.read_csv(peaklist)
        else:
            clusters = pd.read_pickle(peaklist)
        groups = clusters.groupby("CLUSTID")
        for ind, group in groups:
            if len(group) == 1:
                ax.plot(group.X_PPM, group.Y_PPM, "ko", markersize=1)  # , picker=5)
            else:
                ax.plot(group.X_PPM, group.Y_PPM, "o", markersize=1)  # , picker=5)

    if params.get("outname") and (type(params.get("outname")) == list):
        for i in params.get("outname"):
            plt.savefig(i, bbox_inches="tight", dpi=300)
    else:
        plt.savefig(params.get("outname", "test.pdf"), bbox_inches="tight")

    # fig.canvas.mpl_connect("pick_event", onpick)
    # line, = ax.plot(np.random.rand(100), 'o', picker=5)  # 5 points tolerance
    plt.show()


if __name__ == "__main__":
    app()
