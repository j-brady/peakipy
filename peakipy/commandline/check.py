#!/usr/bin/env python3
""" Plot peakipy fits

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


        --rcount=<int>            row count setting for wireplot [default: 50]
        --ccount=<int>            column count setting for wireplot [default: 50]
        --colors=<data,fit>       plot colors [default: #5e3c99,#e66101]
        
        --help

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
from sys import exit
from pathlib import Path

import pandas as pd
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
from colorama import Fore, init

init(autoreset=True)
from tabulate import tabulate
from docopt import docopt
from schema import SchemaError, Schema, And, Use
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Button

from peakipy.core import (
    make_mask,
    pvoigt2d,
    voigt2d,
    pv_pv,
    pv_g,
    pv_l,
    gaussian_lorentzian,
    Pseudo3D,
    run_log,
    read_config,
)

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


def check_input(args):
    """ validate commandline input """
    schema = Schema(
        {
            "<fits>": And(
                os.path.exists,
                open,
                error=Fore.RED + f"{args['<fits>']} should exist and be readable",
            ),
            "<nmrdata>": And(
                os.path.exists,
                ng.pipe.read,
                error=Fore.RED
                + f"{args['<nmrdata>']} either does not exist or is not an NMRPipe format 2D or 3D",
            ),
            "--dims": Use(
                lambda n: [int(i) for i in n.split(",")],
                error=Fore.RED + "--dims should be list of integers e.g. --dims=0,1,2",
            ),
            "--plane": Use(
                lambda n: int(n),
                error=Fore.RED + "--plane should be integer e.g. --plane=2",
            ),
            object: object,
        }
    )

    try:
        args = schema.validate(args)
        return args
    except SchemaError as e:
        exit(e)


def print_bad(plane):
    tab = tabulate(
        plane[
            [
                "clustid",
                "amp",
                "center_x_ppm",
                "center_y_ppm",
                "fwhm_x_hz",
                "fwhm_y_hz",
                "lineshape",
            ]
        ],
        headers="keys",
        tablefmt="fancy_grid",
    )
    return tab


def main(argv):

    args = docopt(__doc__, argv=argv)
    args = check_input(args)
    fits = Path(args.get("<fits>"))
    fits = pd.read_csv(fits)

    # get dims from config file
    config_path = Path("peakipy.config")
    args, config = read_config(args, config_path)

    dims = args.get("--dims")
    colors = args.get("colors")
    data_path = args.get("<nmrdata>")
    dic, data = ng.pipe.read(data_path)
    pseudo3D = Pseudo3D(dic, data, dims)

    outname = args.get("--outname")
    first_only = args.get("--first")
    show = args.get("--show")
    clusters = args.get("--clusters")
    ccount = eval(args.get("--ccount"))
    rcount = eval(args.get("--rcount"))

    # first only overrides plane option
    if first_only:
        plane = 0
    else:
        plane = args.get("--plane")

    if plane > pseudo3D.n_planes:
        raise ValueError(
            Fore.RED
            + f"There are {pseudo3D.n_planes} planes in your data you selected --plane={plane}..."
            f"plane numbering starts from 0."
        )
    elif plane < 0:
        raise ValueError(
            Fore.RED
            + f"Plane number can not be negative; you selected --plane={plane}..."
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

    if (type(colors) == list) and len(colors) == 2:
        data_color, fit_color = colors
    else:
        raise TypeError(
            "colors should be valid pair for matplotlib. i.e. g,b or green,blue"
        )

    if clusters == "None":
        pass
    else:
        clusters = [int(i) for i in clusters.split(",")]
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

        for ind, group in groups:

            print(
                Fore.BLUE
                + tabulate(
                    group[columns_to_print],
                    showindex=False,
                    tablefmt="fancy_grid",
                    headers="keys",
                    floatfmt=".3f",
                )
            )

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
                        if (ls == "G") or (ls == "L") or (ls == "PV"):
                            sim_data_i = pvoigt2d(
                                XY, amp, c_x, c_y, s_x, s_y, frac
                            ).reshape(shape)
                        elif ls == "PV_L":
                            sim_data_i = pv_l(
                                XY, amp, c_x, c_y, s_x, s_y, frac
                            ).reshape(shape)

                        elif ls == "PV_G":
                            sim_data_i = pv_g(
                                XY, amp, c_x, c_y, s_x, s_y, frac
                            ).reshape(shape)

                        elif ls == "G_L":
                            sim_data_i = gaussian_lorentzian(
                                XY, amp, c_x, c_y, s_x, s_y, frac
                            ).reshape(shape)

                        elif ls == "V":
                            sim_data_i = voigt2d(
                                XY, amp, c_x, c_y, s_x, s_y, frac
                            ).reshape(shape)
                        sim_data += sim_data_i
                        sim_data_singles.append(sim_data_i)

                masked_data = pseudo3D.data[plane_id].copy()
                masked_sim_data = sim_data.copy()
                masked_data[~mask] = np.nan
                masked_sim_data[~mask] = np.nan

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
                        Fore.RED + f"Nothing to plot for cluster {int(plane.clustid)}"
                    )
                    print(Fore.RED + f"x={x_plot},y={y_plot}")
                    print(Fore.RED + print_bad(plane))
                    plt.close()
                    # print(Fore.RED + "Maybe your F1/F2 radii for fitting were too small...")
                elif masked_data.shape[0] == 0 or masked_data.shape[1] == 0:
                    print(
                        Fore.RED + f"Nothing to plot for cluster {int(plane.clustid)}"
                    )
                    print(Fore.RED + print_bad(plane))
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

                    if args.get("--individual"):
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
                    print(Fore.GREEN + f"Plotting: {title}")
                    out_str = "Volumes (Heights)\n----------------\n"
                    # chi2s = []
                    label_peaks = args.get("--label")
                    for ind, row in plane.iterrows():

                        out_str += (
                            f"{row.assignment} = {row.amp:.3e} ({row.height:.3e})\n"
                        )
                        if label_peaks:
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
                        plt.show()

                    plt.close()

                    if first_only:
                        break
    run_log()


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
