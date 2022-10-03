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
from typing import Optional, Tuple

import typer
from docopt import docopt
from schema import And, Or, Use, Schema, SchemaError
import nmrglue as ng
from colorama import Fore, init

from peakipy.core import Peaklist, run_log, LoadData


app = typer.Typer()
# colorama
init(autoreset=True)


def check_args(args):

    schema = Schema(
        {
            "<peaklist>": And(
                os.path.exists,
                open,
                error=f"🤔 {args['<peaklist>']} should exist and be readable",
            ),
            "<data>": And(
                os.path.exists,
                ng.pipe.read,
                error=f"🤔 {args['<data>']} should be NMRPipe format 2D or 3D cube",
            ),
            "--thres": Or("None", Use(float)),
            "--struc_el": Use(str),
            # "--struc_size": Use(str),
            "--f1radius": Use(
                float,
                error=f"F1 radius must be a float - you gave {args['--f1radius']}",
            ),
            "--f2radius": Use(
                float,
                error=f"F2 radius must be a float - you gave {args['--f2radius']}",
            ),
            "--dims": Use(
                lambda n: [int(i) for i in eval(n)],
                error="🤔 --dims should be list of integers e.g. --dims=0,1,2",
            ),
            "--posF1": Use(str),  # check whether in dic
            "--posF2": Use(str),  # check whether in dic
            "--outfmt": Or("csv", error="Currently must be csv"),
            object: object,
        },
        # ignore_extra_keys=True,
    )

    # validate arguments
    try:
        args = schema.validate(args)
    except SchemaError as e:
        exit(e)

    return args


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

@app.command()
def read(
    peaklist: Path,
    data: Path,
    peaklist_format: PeaklistFormat,
    thres: Optional[float] = None,
    struc_el: StrucEl = "disk",
    struc_size: Tuple = (3, None),#Tuple[int, Optional[int]] = (3, None),
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
        read <peaklist> <data> (--a2|--a3|--sparky|--pipe|--peakipy) [options]

    Arguments:
        <peaklist>                Analysis2/CCPNMRv3(assign)/Sparky/NMRPipe peak list (see below)
        <data>                    2D or pseudo3D NMRPipe data

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

    args = docopt(__doc__, argv=argv)
    args = check_args(args)

    #verbose_mode = args.get("--verb")
    #if verbose_mode:
    #    print("Using arguments:", args)

    clust_args = {
        "struc_el": struc_el,
        "struc_size": struc_size,
    }
    # name of output peaklist
    outname = peaklist.stem
    cluster = True

    match peaklist_format:
        case "a2":
            # set X and Y ppm column names if not default (i.e. "Position F1" = "X_PPM"
            # "Position F2" = "Y_PPM" ) this is due to Analysis2 often having the
            #  dimension order flipped relative to convention
            peaks = Peaklist(
                peaklist,
                data,
                fmt="a2",
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
                posF1=y_ppm_column_name,
                posF2=x_ppm_column_name,
            )
            # peaks.adaptive_clusters(block_size=151,offset=0)

        case "a3":
            peaks = Peaklist(
                peaklist,
                data,
                fmt="a3",
                dims=dims,
                radii=[x_radius_ppm, y_radius_ppm],
            )

        case "sparky":

            peaks = Peaklist(
                peaklist, data, fmt="sparky", dims=dims, radii=[x_radius_ppm, y_radius_ppm]
            )

        case "pipe":
            peaks = Peaklist(
                peaklist, data, fmt="pipe", dims=dims, radii=[x_radius_ppm, y_radius_ppm]
            )

        case "peakipy":
            # read in a peakipy .csv file
            peaks = LoadData(peaklist, data, fmt="peakipy", dims=dims)
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

    #if verbose_mode:
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
        ("--dims", dims),
        ("<data>", data),
        ("--thres", float(thres)),
        ("--y_radius_ppm", y_radius_ppm),
        ("--x_radius_ppm", x_radius_ppm),
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

    print(Fore.GREEN + f"Finished! Use {outname} to run peakipy edit or fit.")


if __name__ == "__main__":
    app()

