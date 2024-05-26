#!/usr/bin/env python3
import os
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Annotated
from multiprocessing import Pool, cpu_count

import typer
import numpy as np
import nmrglue as ng
import pandas as pd

from tqdm import tqdm
from rich import print
from skimage.filters import threshold_otsu
from pydantic import BaseModel

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

import yaml
import plotly.io as pio

pio.templates.default = "plotly_dark"

from peakipy.io import (
    Peaklist,
    LoadData,
    Pseudo3D,
    StrucEl,
    PeaklistFormat,
    OutFmt,
    get_vclist,
)
from peakipy.utils import (
    run_log,
    df_to_rich_table,
    write_config,
    update_config_file,
    update_args_with_values_from_config_file,
    update_linewidths_from_hz_to_points,
    update_peak_positions_from_ppm_to_points,
    check_data_shape_is_consistent_with_dims,
    check_for_include_column_and_add_if_missing,
    remove_excluded_peaks,
    warn_if_trying_to_fit_large_clusters,
    save_data,
)

from peakipy.lineshapes import (
    Lineshape,
    calculate_lineshape_specific_height_and_fwhm,
    calculate_peak_centers_in_ppm,
    calculate_peak_linewidths_in_hz,
)
from peakipy.fitting import (
    get_limits_for_axis_in_points,
    deal_with_peaks_on_edge_of_spectrum,
    select_specified_planes,
    exclude_specified_planes,
    unpack_xy_bounds,
    validate_plane_selection,
    get_fit_data_for_selected_peak_clusters,
    make_masks_from_plane_data,
    simulate_lineshapes_from_fitted_peak_parameters,
    simulate_pv_pv_lineshapes_from_fitted_peak_parameters,
    validate_fit_dataframe,
)

from .fit import (
    fit_peak_clusters,
    FitPeaksInput,
    FitPeaksArgs,
)
from peakipy.plotting import (
    PlottingDataForPlane,
    validate_sample_count,
    unpack_plotting_colors,
    create_plotly_figure,
    create_residual_figure,
    create_matplotlib_figure,
)
from .spec import yaml_file

app = typer.Typer()
tmp_path = Path("tmp")
tmp_path.mkdir(exist_ok=True)
log_path = Path("log.txt")


peaklist_path_help = "Path to peaklist"
data_path_help = "Path to 2D or pseudo3D processed NMRPipe data (e.g. .ft2 or .ft3)"
peaklist_format_help = "The format of your peaklist. This can be a2 for CCPN Analysis version 2 style, a3 for CCPN Analysis version 3, sparky, pipe for NMRPipe, or peakipy if you want to use a previously .csv peaklist from peakipy"
thres_help = "Threshold for making binary mask that is used for peak clustering. If set to None then threshold_otsu from scikit-image is used to determine threshold"
x_radius_ppm_help = "X radius in ppm of the elliptical fitting mask for each peak"
y_radius_ppm_help = "Y radius in ppm of the elliptical fitting mask for each peak"
dims_help = "Dimension order of your data"


@app.command(help="Read NMRPipe/Analysis peaklist into pandas dataframe")
def read(
    peaklist_path: Annotated[Path, typer.Argument(help=peaklist_path_help)],
    data_path: Annotated[Path, typer.Argument(help=data_path_help)],
    peaklist_format: Annotated[
        PeaklistFormat, typer.Argument(help=peaklist_format_help)
    ],
    thres: Annotated[Optional[float], typer.Option(help=thres_help)] = None,
    struc_el: StrucEl = StrucEl.disk,
    struc_size: Tuple[int, int] = (3, None),  # Tuple[int, Optional[int]] = (3, None),
    x_radius_ppm: Annotated[float, typer.Option(help=x_radius_ppm_help)] = 0.04,
    y_radius_ppm: Annotated[float, typer.Option(help=y_radius_ppm_help)] = 0.4,
    x_ppm_column_name: str = "Position F1",
    y_ppm_column_name: str = "Position F2",
    dims: Annotated[List[int], typer.Option(help=dims_help)] = [0, 1, 2],
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

    clust_args = {
        "struc_el": struc_el,
        "struc_size": struc_size,
    }
    # name of output peaklist
    outname = peaklist_path.parent / peaklist_path.stem
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
            peaks = LoadData(
                peaklist_path, data_path, fmt=PeaklistFormat.peakipy, dims=dims
            )
            cluster = False
            # don't overwrite the old .csv file
            outname = outname.parent / (outname.stem + "_new")

    peaks.update_df()

    data = peaks.df
    thres = peaks.thres

    if cluster:
        peaks.clusters(thres=thres, **clust_args, l_struc=None)
    else:
        pass

    if fuda:
        peaks.to_fuda()

    match outfmt.value:
        case "csv":
            outname = outname.with_suffix(".csv")
            data.to_csv(outname, float_format="%.4f", index=False)
        case "pkl":
            outname = outname.with_suffix(".pkl")
            data.to_pickle(outname)

    # write config file
    config_path = peaklist_path.parent / Path("peakipy.config")
    config_kvs = [
        ("dims", dims),
        ("data_path", str(data_path)),
        ("thres", float(thres)),
        ("y_radius_ppm", y_radius_ppm),
        ("x_radius_ppm", x_radius_ppm),
        ("fit_method", "leastsq"),
    ]
    try:
        update_config_file(config_path, config_kvs)

    except json.decoder.JSONDecodeError:
        print(
            "\n"
            + f"[yellow]Your {config_path} may be corrupted. Making new one (old one moved to {config_path}.bak)[/yellow]"
        )
        shutil.copy(f"{config_path}", f"{config_path}.bak")
        config_dic = dict(config_kvs)
        write_config(config_path, config_dic)

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

    print(
        f"""[green]

          ✨✨ Finished reading and clustering peaks! ✨✨

             Use {outname} to run peakipy edit or fit.[/green]

          """
    )


fix_help = "Set parameters to fix after initial lineshape fit (see docs)"
xy_bounds_help = (
    "Restrict fitted peak centre within +/- x and y from initial picked position"
)
reference_plane_index_help = (
    "Select plane(s) to use for initial estimation of lineshape parameters"
)
mp_help = "Use multiprocessing"
vclist_help = "Provide a vclist style file"
plane_help = "Select individual planes for fitting"
exclude_plane_help = "Exclude individual planes from fitting"


@app.command(help="Fit NMR data to lineshape models and deconvolute overlapping peaks")
def fit(
    peaklist_path: Annotated[Path, typer.Argument(help=peaklist_path_help)],
    data_path: Annotated[Path, typer.Argument(help=data_path_help)],
    output_path: Path,
    max_cluster_size: Optional[int] = None,
    lineshape: Lineshape = Lineshape.PV,
    fix: Annotated[List[str], typer.Option(help=fix_help)] = [
        "fraction",
        "sigma",
        "center",
    ],
    xy_bounds: Annotated[Tuple[float, float], typer.Option(help=xy_bounds_help)] = (
        0,
        0,
    ),
    vclist: Annotated[Optional[Path], typer.Option(help=vclist_help)] = None,
    plane: Annotated[Optional[List[int]], typer.Option(help=plane_help)] = None,
    exclude_plane: Annotated[
        Optional[List[int]], typer.Option(help=exclude_plane_help)
    ] = None,
    reference_plane_index: Annotated[
        List[int], typer.Option(help=reference_plane_index_help)
    ] = [],
    initial_fit_threshold: Optional[float] = None,
    jack_knife_sample_errors: bool = False,
    mp: Annotated[bool, typer.Option(help=mp_help)] = True,
    verbose: bool = False,
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
        This can be set like so --xy-bounds 0.1 0.5
    vclist : Optional[Path]
        Bruker style vclist [default: None]
    plane : Optional[List[int]]
        Specific plane(s) to fit [default: None]
        eg. [1,4,5] will use only planes 1, 4 and 5
    exclude_plane : Optional[List[int]]
        Specific plane(s) to fit [default: None]
        eg. [1,4,5] will exclude planes 1, 4 and 5
    initial_fit_threshold: Optional[float]
        threshold used to select planes for fitting of initial lineshape parameters. Only planes with
        intensities above this threshold will be included in the intial fit of summed planes.
    mp : bool
        Use multiprocessing [default: True]
    verb : bool
        Print what's going on
    """
    # number of CPUs
    n_cpu = cpu_count()

    # read NMR data
    args = {}
    config = {}
    data_dir = peaklist_path.parent
    args, config = update_args_with_values_from_config_file(
        args, config_path=data_dir / "peakipy.config"
    )
    dims = config.get("dims", [0, 1, 2])
    peakipy_data = LoadData(peaklist_path, data_path, dims=dims)
    peakipy_data = check_for_include_column_and_add_if_missing(peakipy_data)
    peakipy_data = remove_excluded_peaks(peakipy_data)
    max_cluster_size = warn_if_trying_to_fit_large_clusters(
        max_cluster_size, peakipy_data
    )
    # remove peak clusters larger than max_cluster_size
    peakipy_data.df = peakipy_data.df[peakipy_data.df.MEMCNT <= max_cluster_size]

    args["max_cluster_size"] = max_cluster_size
    args["to_fix"] = fix
    args["verbose"] = verbose
    args["mp"] = mp
    args["initial_fit_threshold"] = initial_fit_threshold
    args["reference_plane_indices"] = reference_plane_index
    args["jack_knife_sample_errors"] = jack_knife_sample_errors

    args = get_vclist(vclist, args)
    # plot results or not
    log_file = open(tmp_path / log_path, "w")

    uc_dics = {"f1": peakipy_data.uc_f1, "f2": peakipy_data.uc_f2}
    args["uc_dics"] = uc_dics

    check_data_shape_is_consistent_with_dims(peakipy_data)
    plane_numbers, peakipy_data = select_specified_planes(plane, peakipy_data)
    plane_numbers, peakipy_data = exclude_specified_planes(exclude_plane, peakipy_data)
    noise = abs(threshold_otsu(peakipy_data.data))
    args["noise"] = noise
    args["lineshape"] = lineshape
    xy_bounds = unpack_xy_bounds(xy_bounds, peakipy_data)
    args["xy_bounds"] = xy_bounds
    peakipy_data = update_linewidths_from_hz_to_points(peakipy_data)
    peakipy_data = update_peak_positions_from_ppm_to_points(peakipy_data)
    # prepare data for multiprocessing
    nclusters = peakipy_data.df.CLUSTID.nunique()
    npeaks = peakipy_data.df.shape[0]
    if (nclusters >= n_cpu) and mp:
        print(
            f"[green]Using multiprocessing to fit {npeaks} peaks in {nclusters} clusters [/green]"
            + "\n"
        )
        fit_peaks_args = FitPeaksInput(
            FitPeaksArgs(**args), peakipy_data.data, config, plane_numbers
        )
        with (
            Pool(processes=n_cpu) as pool,
            tqdm(
                total=len(peakipy_data.df.CLUSTID.unique()),
                ascii="▱▰",
                colour="green",
            ) as pbar,
        ):
            result = [
                pool.apply_async(
                    fit_peak_clusters,
                    args=(
                        peaklist,
                        fit_peaks_args,
                    ),
                    callback=lambda _: pbar.update(1),
                ).get()
                for _, peaklist in peakipy_data.df.groupby("CLUSTID")
            ]
            df = pd.concat([i.df for i in result], ignore_index=True)
            for num, i in enumerate(result):
                log_file.write(i.log + "\n")
    else:
        print("[green]Not using multiprocessing[green]")
        result = fit_peak_clusters(
            peakipy_data.df,
            FitPeaksInput(
                FitPeaksArgs(**args), peakipy_data.data, config, plane_numbers
            ),
        )
        df = result.df
        log_file.write(result.log)

    # finished fitting

    # close log file
    log_file.close()
    output = Path(output_path)
    df = calculate_lineshape_specific_height_and_fwhm(lineshape, df)
    df = calculate_peak_centers_in_ppm(df, peakipy_data)
    df = calculate_peak_linewidths_in_hz(df, peakipy_data)

    save_data(df, output)

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
    plane: Optional[List[int]] = None,
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
    plotly: bool = False,
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
        e.g. --plane 2 will plot second plane only
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
    fits = validate_fit_dataframe(pd.read_csv(fits))
    args = {}
    # get dims from config file
    config_path = data_path.parent / "peakipy.config"
    args, config = update_args_with_values_from_config_file(args, config_path)
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
        selected_planes = [0]
    else:
        selected_planes = validate_plane_selection(plane, pseudo3D)
    ccount = validate_sample_count(ccount)
    rcount = validate_sample_count(rcount)
    data_color, fit_color = unpack_plotting_colors(colors)
    fits = get_fit_data_for_selected_peak_clusters(fits, clusters)

    peak_clusters = fits.query(f"plane in @selected_planes").groupby("clustid")

    # make plotting meshes
    x = np.arange(pseudo3D.f2_size)
    y = np.arange(pseudo3D.f1_size)
    XY = np.meshgrid(x, y)
    X, Y = XY

    with PdfPages(outname) as pdf:
        for _, peak_cluster in peak_clusters:
            table = df_to_rich_table(
                peak_cluster,
                title="",
                columns=columns_to_print,
                styles=["blue" for _ in columns_to_print],
            )
            print(table)

            x_radius = peak_cluster.x_radius.max()
            y_radius = peak_cluster.y_radius.max()
            max_x, min_x = get_limits_for_axis_in_points(
                group_axis_points=peak_cluster.center_x, mask_radius_in_points=x_radius
            )
            max_y, min_y = get_limits_for_axis_in_points(
                group_axis_points=peak_cluster.center_y, mask_radius_in_points=y_radius
            )
            max_x, min_x, max_y, min_y = deal_with_peaks_on_edge_of_spectrum(
                pseudo3D.data.shape, max_x, min_x, max_y, min_y
            )

            empty_mask_array = np.zeros(
                (pseudo3D.f1_size, pseudo3D.f2_size), dtype=bool
            )
            first_plane = peak_cluster[peak_cluster.plane == selected_planes[0]]
            individual_masks, mask = make_masks_from_plane_data(
                empty_mask_array, first_plane
            )

            # generate simulated data
            for plane_id, plane in peak_cluster.groupby("plane"):
                sim_data_singles = []
                sim_data = np.zeros((pseudo3D.f1_size, pseudo3D.f2_size))
                try:
                    (
                        sim_data,
                        sim_data_singles,
                    ) = simulate_pv_pv_lineshapes_from_fitted_peak_parameters(
                        plane, XY, sim_data, sim_data_singles
                    )
                except:
                    (
                        sim_data,
                        sim_data_singles,
                    ) = simulate_lineshapes_from_fitted_peak_parameters(
                        plane, XY, sim_data, sim_data_singles
                    )

                plot_data = PlottingDataForPlane(
                    pseudo3D,
                    plane_id,
                    plane,
                    X,
                    Y,
                    mask,
                    individual_masks,
                    sim_data,
                    sim_data_singles,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    fit_color,
                    data_color,
                    rcount,
                    ccount,
                )

                if ccpn_flag:
                    plt = PlotterWidget()
                # fig = create_plotly_figure(plot_data)
                if plotly:
                    fig = create_plotly_figure(plot_data)
                    residual_fig = create_residual_figure(plot_data)
                    return fig, residual_fig
                else:
                    plt = matplotlib.pyplot
                    create_matplotlib_figure(
                        plot_data, pdf, individual, label, ccpn_flag, show
                    )
                if first:
                    break

    run_log()


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
