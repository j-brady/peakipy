#!/usr/bin/env python3
"""Fit and deconvolute NMR peaks

    Usage:
        fit_peaks.py <peaklist> <data> <output> [options]

    Arguments:
        <peaklist>                             peaklist output from read_peaklist.py
        <data>                                 2D or pseudo3D NMRPipe data (single file)
        <output>                               output peaklist "<output>.csv" will output CSV
                                               format file, "<output>.tab" will give a tab delimited output
                                               while "<output>.pkl" results in Pandas pickle of DataFrame

    Options:
        -h --help                              Show this page
        -v --version                           Show version

        --dims=<ID,F1,F2>                      Dimension order [default: 0,1,2]
        --max_cluster_size=<max_cluster_size>  Maximum size of cluster to fit (i.e exclude large clusters) [default: None]
        --lineshape=<G/L/PV>                   lineshape to fit [default: PV]
        --fix=<fraction,sigma,center>          Parameters to fix after initial fit on summed planes [default: fraction,sigma,center]
        --vclist=<fname>                       Bruker style vclist [default: None]

        --plot=<dir>                           Whether to plot wireframe fits for each peak
                                               (saved into <dir>) [default: None]

        --show                                 Whether to show (using plt.show()) wireframe
                                               fits for each peak. Only works if --plot is also selected

        --verb                                 Print what's going on


"""
from pathlib import Path

import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lmfit import Model
from mpl_toolkits.mplot3d import Axes3D
from docopt import docopt

from peakipy.core import fix_params, get_params, fit_first_plane, Pseudo3D


def norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


args = docopt(__doc__)

max_cluster_size = args.get("--max_cluster_size")
if max_cluster_size == "None":
    max_cluster_size = 1000
else:
    max_cluster_size = int(max_cluster_size)

lineshape = args.get("--lineshape")
# params to fix
to_fix = args.get("--fix")
to_fix = to_fix.split(",")
verb = args.get("--verb")
if verb:
    print("Using ", args)
log = open("log.txt", "w")

# path to peaklist
peaklist = Path(args.get("<peaklist>"))

# determine filetype
if peaklist.suffix == ".csv":
    peaks = pd.read_csv(peaklist)
else:
    # assume that file is a pickle
    peaks = pd.read_pickle(peaklist)

# read vclist
vclist = args.get("--vclist")
if vclist == "None":
    vclist = False
else:
    vclist_data = np.genfromtxt(vclist)
    vclist = True

# plot results or not
plot = args.get("--plot")
if plot == "None":
    plot = None
else:
    plot = Path(plot)
    plot.mkdir(parents=True, exist_ok=True)

# get dims from command line input
dims = args.get("--dims")
dims = [int(i) for i in dims.split(",")]

# read NMR data
dic, data = ng.pipe.read(args["<data>"])

pseudo3D = Pseudo3D(dic, data, dims)
uc_f1 = pseudo3D.uc_f1
uc_f2 = pseudo3D.uc_f2
uc_dics = {"f1": uc_f1, "f2": uc_f2}

dims = pseudo3D.dims
data = pseudo3D.data

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

# convert linewidths from Hz to points in case they were adjusted when running run_check_fits.py
peaks["XW"] = peaks.XW_HZ * pt_per_hz_f2
peaks["YW"] = peaks.YW_HZ * pt_per_hz_f1

# convert peak positions from ppm to points in case they were adjusted running run_check_fits.py
peaks["X_AXIS"] = peaks.X_PPM.apply(lambda x: uc_f2(x, "PPM"))
peaks["Y_AXIS"] = peaks.Y_PPM.apply(lambda x: uc_f1(x, "PPM"))
peaks["X_AXISf"] = peaks.X_PPM.apply(lambda x: uc_f2.f(x, "PPM"))
peaks["Y_AXISf"] = peaks.Y_PPM.apply(lambda x: uc_f1.f(x, "PPM"))

# sum planes for initial fit
summed_planes = data.sum(axis=0)

# for saving data, currently not using errs for center and sigma
amps = []
amp_errs = []

center_xs = []
# center_x_errs = []

center_ys = []
# center_y_errs = []

sigma_ys = []
# sigma_y_errs = []

sigma_xs = []
# sigma_x_errs = []

fractions = []
names = []
indices = []
assign = []
clustids = []
planes = []
x_radii = []
y_radii = []
x_radii_ppm = []
y_radii_ppm = []
lineshapes = []

# group peaks based on CLUSTID
groups = peaks.groupby("CLUSTID")

# iterate over groups of peaks
for name, group in groups:
    #  max cluster size
    if len(group) <= max_cluster_size:
        # fits sum of all planes first
        first, mask = fit_first_plane(
            group,
            summed_planes,
            # norm(summed_planes),
            uc_dics,
            lineshape=lineshape,
            plot=plot,
            show=args.get("--show"),
            verbose=verb,
            log=log,
        )

        # fix sigma center and fraction parameters
        # could add an option to select params to fix
        if len(to_fix) == 0 or to_fix == "None":
            if verb:
                print("Floating all parameters")
            pass
        else:
            to_fix = to_fix
            if verb:
                print("Fixing parameters:", to_fix)
            fix_params(first.params, to_fix)

        for num, d in enumerate(data):

            first.fit(data=d[mask], params=first.params)
            if verb:
                print(first.fit_report())

            amp, amp_err, name = get_params(first.params, "amplitude")
            cen_x, cen_x_err, name = get_params(first.params, "center_x")
            cen_y, cen_y_err, name = get_params(first.params, "center_y")
            sig_x, sig_x_err, name = get_params(first.params, "sigma_x")
            sig_y, sig_y_err, name = get_params(first.params, "sigma_y")
            frac, frac_err, name = get_params(first.params, "fraction")

            amps.extend(amp)
            amp_errs.extend(amp_err)
            center_xs.extend(cen_x)
            # center_x_errs.extend(cen_x_err)
            center_ys.extend(cen_y)
            # center_y_errs.extend(cen_y_err)
            sigma_xs.extend(sig_x)
            # sigma_x_errs.extend(sig_x_err)
            sigma_ys.extend(sig_y)
            # sigma_y_errs.extend(sig_y_err)
            fractions.extend(frac)
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
    # "center_x_err": center_x_errs,
    "center_y": center_ys,
    # "center_y_err": center_y_errs,
    "sigma_x": sigma_xs,
    # "sigma_x_err": sigma_x_errs,
    "sigma_y": sigma_ys,
    # "sigma_y_err": sigma_y_errs,
    "fraction": fractions,
    "clustid": clustids,
    "plane": planes,
    "x_radius": x_radii,
    "y_radius": y_radii,
    "x_radius_ppm": x_radii_ppm,
    "y_radius_ppm": y_radii_ppm,
    "lineshape": lineshapes,
}

#  make dataframe
df = pd.DataFrame(df_dic)
#  convert sigmas to fwhm
df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)
#  convert values to ppm
df["center_x_ppm"] = df.center_x.apply(lambda x: uc_f2.ppm(x))
df["center_y_ppm"] = df.center_y.apply(lambda x: uc_f1.ppm(x))
df["sigma_x_ppm"] = df.sigma_x.apply(lambda x: x * ppm_per_pt_f2)
df["sigma_y_ppm"] = df.sigma_y.apply(lambda x: x * ppm_per_pt_f1)
df["fwhm_x_ppm"] = df.fwhm_x.apply(lambda x: x * ppm_per_pt_f2)
df["fwhm_y_ppm"] = df.fwhm_y.apply(lambda x: x * ppm_per_pt_f1)
df["fwhm_x_hz"] = df.fwhm_x.apply(lambda x: x * hz_per_pt_f2)
df["fwhm_y_hz"] = df.fwhm_y.apply(lambda x: x * hz_per_pt_f1)
# Fill nan values
df.fillna(value=np.nan, inplace=True)
# vclist
if vclist:
    df["vclist"] = df.plane.apply(lambda x: vclist_data[x])
#  output data
output = Path(args["<output>"])
suffix = output.suffix
if suffix == ".csv":
    df.to_csv(output, float_format="%.4f")

elif suffix == ".tab":
    df.to_csv(output, sep="\t", float_format="%.4f")

else:
    df.to_pickle(output)
