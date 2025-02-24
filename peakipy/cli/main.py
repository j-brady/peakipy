#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Annotated
from multiprocessing import Pool, cpu_count

import typer
import numpy as np
import nmrglue as ng
import pandas as pd

from tqdm import tqdm
from rich import print
from skimage.filters import threshold_otsu

from mpl_toolkits.mplot3d import axes3d
from matplotlib.backends.backend_pdf import PdfPages
from bokeh.models.widgets.tables import ScientificFormatter

import plotly.io as pio
import panel as pn

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
    mkdir_tmp_dir,
    create_log_path,
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
    check_for_existing_output_file_and_backup
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
from peakipy.cli.edit import BokehScript

pn.extension("plotly")
pn.config.theme = "dark"


@dataclass
class PlotContainer:
    main_figure: pn.pane.Plotly
    residual_figure: pn.pane.Plotly


@lru_cache(maxsize=1)
def data_singleton_edit():
    return EditData()


@lru_cache(maxsize=1)
def data_singleton_check():
    return CheckData()


@dataclass
class EditData:
    peaklist_path: Path = Path("./test.csv")
    data_path: Path = Path("./test.ft2")
    _bs: BokehScript = field(init=False)

    def load_data(self):
        self._bs = BokehScript(self.peaklist_path, self.data_path)

    @property
    def bs(self):
        return self._bs


@dataclass
class CheckData:
    fits_path: Path = Path("./fits.csv")
    data_path: Path = Path("./test.ft2")
    config_path: Path = Path("./peakipy.config")
    _df: pd.DataFrame = field(init=False)

    def load_dataframe(self):
        self._df = validate_fit_dataframe(pd.read_csv(self.fits_path))

    @property
    def df(self):
        return self._df


app = typer.Typer()


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
        Create a parameter file for running fuda (params.fuda)


    Examples
    --------
        peakipy read test.tab test.ft2 pipe --dims 0 --dims 1
        peakipy read test.a2 test.ft2 a2 --thres 1e5  --dims 0 --dims 2 --dims 1
        peakipy read ccpnTable.tsv test.ft2 a3 --y-radius-ppm 0.3 --x_radius-ppm 0.03
        peakipy read test.csv test.ft2 peakipy --dims 0 --dims 1 --dims 2

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
    mkdir_tmp_dir(peaklist_path.parent)
    log_path = create_log_path(peaklist_path.parent)

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
        
        case peaklist_format.csv:
            peaks = Peaklist(
                peaklist_path,
                data_path,
                fmt=PeaklistFormat.csv,
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
            )

    peaks.update_df()

    data = peaks.df
    thres = peaks.thres

    if cluster:
        if struc_el == StrucEl.mask_method:
            peaks.mask_method(overlap=struc_size[0])
        else:
            peaks.clusters(thres=thres, **clust_args, l_struc=None)
    else:
        pass

    if fuda:
        peaks.to_fuda()

    match outfmt.value:
        case "csv":
            outname = outname.with_suffix(".csv")
            data.to_csv(check_for_existing_output_file_and_backup(outname), float_format="%.4f", index=False)
        case "pkl":
            outname = outname.with_suffix(".pkl")
            data.to_pickle(check_for_existing_output_file_and_backup(outname))

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

    run_log(log_path)

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
    tmp_path = mkdir_tmp_dir(peaklist_path.parent)
    log_path = create_log_path(peaklist_path.parent)
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
    log_file = open(log_path, "w")

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
        print("[green]Not using multiprocessing[/green]")
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
    run_log(log_path)


@app.command()
def edit(peaklist_path: Path, data_path: Path, test: bool = False):
    data = data_singleton_edit()
    data.peaklist_path = peaklist_path
    data.data_path = data_path
    data.load_data()
    panel_app(test=test)


fits_help = "CSV file containing peakipy fits"
panel_help = "Open fits in browser with an interactive panel app"
individual_help = "Show individual peak fits as surfaces"
label_help = "Add peak assignment labels"
first_help = "Show only first plane"
plane_help = "Select planes to plot"
clusters_help = "Select clusters to plot"
colors_help = "Customize colors for data and fit lines respectively"
show_help = "Open interactive matplotlib window"
outname_help = "Name of output multipage pdf"


@app.command(help="Interactive plots for checking fits")
def check(
    fits_path: Annotated[Path, typer.Argument(help=fits_help)],
    data_path: Annotated[Path, typer.Argument(help=data_path_help)],
    panel: Annotated[bool, typer.Option(help=panel_help)] = False,
    clusters: Annotated[Optional[List[int]], typer.Option(help=clusters_help)] = None,
    plane: Annotated[Optional[List[int]], typer.Option(help=plane_help)] = None,
    first: Annotated[bool, typer.Option(help=first_help)] = False,
    show: Annotated[bool, typer.Option(help=show_help)] = False,
    label: Annotated[bool, typer.Option(help=label_help)] = False,
    individual: Annotated[bool, typer.Option(help=individual_help)] = False,
    outname: Annotated[Path, typer.Option(help=outname_help)] = Path("plots.pdf"),
    colors: Annotated[Tuple[str, str], typer.Option(help=colors_help)] = (
        "#5e3c99",
        "#e66101",
    ),
    rcount: int = 50,
    ccount: int = 50,
    ccpn: bool = False,
    plotly: bool = False,
    test: bool = False,
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
    log_path = create_log_path(fits_path.parent)
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
    fits = validate_fit_dataframe(pd.read_csv(fits_path))
    args = {}
    # get dims from config file
    config_path = data_path.parent / "peakipy.config"
    args, config = update_args_with_values_from_config_file(args, config_path)
    dims = config.get("dims", (1, 2, 3))

    if panel:
        create_check_panel(
            fits_path=fits_path, data_path=data_path, config_path=config_path, test=test
        )
        return

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

    all_plot_data = []
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

        empty_mask_array = np.zeros((pseudo3D.f1_size, pseudo3D.f2_size), dtype=bool)
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
            all_plot_data.append(plot_data)
            if plotly:
                fig = create_plotly_figure(plot_data)
                residual_fig = create_residual_figure(plot_data)
                return fig, residual_fig
            if first:
                break

    with PdfPages(data_path.parent / outname) as pdf:
        for plot_data in all_plot_data:
            create_matplotlib_figure(
                plot_data, pdf, individual, label, ccpn_flag, show, test
            )

    run_log(log_path)


def create_plotly_pane(cluster, plane):
    data = data_singleton_check()
    fig, residual_fig = check(
        fits_path=data.fits_path,
        data_path=data.data_path,
        clusters=[cluster],
        plane=[plane],
        # config_path=data.config_path,
        plotly=True,
    )
    fig["layout"].update(height=800, width=800)
    residual_fig["layout"].update(width=400)
    fig = fig.to_dict()
    residual_fig = residual_fig.to_dict()
    return pn.Row(pn.pane.Plotly(fig), pn.pane.Plotly(residual_fig))


def get_cluster(cluster):
    tabulator_stylesheet = """
    .tabulator-cell {
        font-size: 12px;
    }
    .tabulator-headers {
        font-size: 12px;
    }
    """
    table_formatters = {
        "amp": ScientificFormatter(precision=3),
        "height": ScientificFormatter(precision=3),
    }
    data = data_singleton_check()
    cluster_groups = data.df.groupby("clustid")
    cluster_group = cluster_groups.get_group(cluster)
    df_pane = pn.widgets.Tabulator(
        cluster_group[
            [
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
        ],
        selectable=False,
        disabled=True,
        width=800,
        show_index=False,
        frozen_columns=["assignment","clustid","plane"],
        stylesheets=[tabulator_stylesheet],
        formatters=table_formatters,
    )
    return df_pane


def update_peakipy_data_on_edit_of_table(event):
    data = data_singleton_edit()
    column = event.column
    row = event.row
    value = event.value
    data.bs.peakipy_data.df.loc[row, column] = value
    data.bs.update_memcnt()


def panel_app(test=False):
    data = data_singleton_edit()
    bs = data.bs
    bokeh_pane = pn.pane.Bokeh(bs.p)
    spectrum_view_settings = pn.WidgetBox(
        "# Contour settings", bs.pos_neg_contour_radiobutton, bs.contour_start
    )
    save_peaklist_box = pn.WidgetBox(
        "# Save your peaklist",
        bs.savefilename,
        bs.button,
        pn.layout.Divider(),
        bs.exit_button,
    )
    recluster_settings = pn.WidgetBox(
        "# Re-cluster your peaks",
        bs.clust_div,
        bs.struct_el,
        bs.struct_el_size,
        pn.layout.Divider(),
        bs.recluster_warning,
        bs.recluster,
        sizing_mode="stretch_width",
    )
    button = pn.widgets.Button(name="Fit selected cluster(s)", button_type="primary")
    fit_controls = pn.WidgetBox(
        "# Fit controls",
        button,
        pn.layout.Divider(),
        bs.select_plane,
        bs.checkbox_group,
        pn.layout.Divider(),
        bs.select_reference_planes_help,
        bs.select_reference_planes,
        pn.layout.Divider(),
        bs.set_initial_fit_threshold_help,
        bs.set_initial_fit_threshold,
        pn.layout.Divider(),
        bs.select_fixed_parameters_help,
        bs.select_fixed_parameters,
        pn.layout.Divider(),
        bs.select_lineshape_radiobuttons_help,
        bs.select_lineshape_radiobuttons,
    )

    mask_adjustment_controls = pn.WidgetBox(
        "# Fitting mask adjustment", bs.slider_X_RADIUS, bs.slider_Y_RADIUS
    )

    # bs.source.on_change()
    def fit_peaks_button_click(event):
        check_app.loading = True
        bs.fit_selected(None)
        check_panel = create_check_panel(bs.TEMP_OUT_CSV, bs.data_path, edit_panel=True)
        check_app.objects = check_panel.objects
        check_app.loading = False

    button.on_click(fit_peaks_button_click)

    def update_source_selected_indices(event):
        bs.source.selected.indices = bs.tabulator_widget.selection

    # Use on_selection_changed to immediately capture the updated selection
    bs.tabulator_widget.param.watch(update_source_selected_indices, 'selection')
    bs.tabulator_widget.on_edit(update_peakipy_data_on_edit_of_table)

    template = pn.template.BootstrapTemplate(
        title="Peakipy",
        sidebar=[mask_adjustment_controls, fit_controls],
    )
    spectrum = pn.Card(
        pn.Column(
            pn.Row(
                bokeh_pane,
                bs.tabulator_widget,),
            pn.Row(
                spectrum_view_settings,recluster_settings, save_peaklist_box,
            ),
        ),
        title="Peakipy fit",
    )
    check_app = pn.Card(title="Peakipy check")
    template.main.append(pn.Column(check_app, spectrum))
    if test:
        return
    else:
        template.show()


def create_check_panel(
    fits_path: Path,
    data_path: Path,
    config_path: Path = Path("./peakipy.config"),
    edit_panel: bool = False,
    test: bool = False,
):
    data = data_singleton_check()
    data.fits_path = fits_path
    data.data_path = data_path
    data.config_path = config_path
    data.load_dataframe()

    clusters = [(row.clustid, row.memcnt) for _, row in data.df.iterrows()]

    select_cluster = pn.widgets.Select(
        name="Cluster (number of peaks)", options={f"{c} ({m})": c for c, m in clusters}
    )
    select_plane = pn.widgets.Select(
        name="Plane", options={f"{plane}": plane for plane in data.df.plane.unique()}
    )
    result_table_pane = pn.bind(get_cluster, select_cluster)
    interactive_plotly_pane = pn.bind(
        create_plotly_pane, cluster=select_cluster, plane=select_plane
    )
    check_pane = pn.Card(
        # info_pane,
        # pn.Row(select_cluster, select_plane),
        pn.Row(
            pn.Column(
                pn.Row(pn.Card(result_table_pane, title="Fitted parameters for cluster"),
                       pn.Card(select_cluster, select_plane, title="Select cluster and plane")),
                pn.Card(interactive_plotly_pane, title="Fitted cluster"),
            ),
        ),
        title="Peakipy check",
    )
    if edit_panel:
        return check_pane
    elif test:
        return
    else:
        check_pane.show()


if __name__ == "__main__":
    app()
