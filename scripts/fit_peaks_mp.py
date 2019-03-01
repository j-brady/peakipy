#!/usr/bin/env python3
"""Fit and deconvolute NMR peaks

    Usage:
        fit_pipe_peaks.py <peaklist> <data> <output> [options]

    Arguments:
        <peaklist>  peaklist output from read_peaklist.py
        <data>      2D or pseudo3D NMRPipe data (single file)  
        <output>    output peaklist "<output>.csv" will output CSV
                    format file, "<output>.tab" will give a tab delimited output
                    while "<output>.pkl" results in Pandas pickle of DataFrame

    Options:
        -h --help  Show this page
        -v --version Show version

        --dims=<ID,F1,F2>                      Dimension order [default: 0,1,2]
        --max_cluster_size=<max_cluster_size>  Maximum size of cluster to fit (i.e exclude large clusters) [default: None]
        --x_radius=<ppm>                       x_radius in ppm for fit mask [default: 0.05]
        --y_radius=<ppm>                       y_radius in ppm for fit mask [default: 0.5]
        --min_rsq=<float>                      minimum R2 required to accept fit [default: 0.85]
        --lineshape=<G/L/PV>                   lineshape to fit [default: PV]

        --plot=<dir>                           Whether to plot wireframe fits for each peak 
                                               (saved into <dir>) [default: None]

        --show                                 Whether to show (using plt.show()) wireframe
                                               fits for each peak

        --vclist=<path>                        vclist-like file containing delays will be incorporated into final dataframe [default: None]

    ToDo: 
        1. per peak R^2, fit first summed spec (may need to adjust start params for this)
        2. decide clusters based on lw?
        3. add vclist data to output?
        4. add threshold to R2 so that you just give an error for the fit and suggest reselecting the group.

"""
import multiprocessing
from multiprocessing import Pool
from pathlib import Path

import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lmfit import Model
from mpl_toolkits.mplot3d import Axes3D
from docopt import docopt

from peak_deconvolution.core import fix_params, get_params, r_square, fit_first_plane


args = docopt(__doc__)

# set up multiprocessing
cpu_count = multiprocessing.cpu_count()
print("CPU count: %d" % cpu_count)

max_cluster_size = args.get("--max_cluster_size")
lineshape = args.get("--lineshape")
if max_cluster_size == "None":
    max_cluster_size = 1000
else:
    max_cluster_size = int(max_cluster_size)

x_radius = float(args.get("--x_radius"))
y_radius = float(args.get("--y_radius"))
min_rsq = float(args.get("--min_rsq"))
print("Using ", args)
log = open("log.txt", "a")

peaklist = Path(args.get("<peaklist>"))
if peaklist.suffix == ".csv":
    peaks = pd.read_csv(peaklist)
else:
    # assume that file is a pickle
    peaks = pd.read_pickle(peaklist)

# peaks = pd.read_pickle(args["<peaklist>"])
groups = peaks.groupby("CLUSTID")

plot = args.get("--plot")
if plot == "None":
    plot = None
else:
    plot = Path(plot)
    plot.mkdir(parents=True, exist_ok=True)

vclist = args.get("--vclist")
print("vclist", vclist)
if vclist == "None":
    add_vclist = False
else:
    vclist = np.genfromtxt(vclist)
    add_vclist = True

# vcpmg = np.genfromtxt("vclist")
# read NMR data
dic, data = ng.pipe.read(args["<data>"])
udic = ng.pipe.guess_udic(dic, data)
dims = args.get("--dims")
dims = [int(i) for i in dims.split(",")]
planes, f1_dim, f2_dim = dims
udic = ng.pipe.guess_udic(dic, data)
uc_f2 = ng.pipe.make_uc(dic, data, dim=f2_dim)
uc_f1 = ng.pipe.make_uc(dic, data, dim=f1_dim)
uc_dics = {"f1": uc_f1, "f2": uc_f2}

# ppm per point
ppm_per_pt_f1 = (udic[f1_dim]["sw"] / udic[f1_dim]["obs"]) / udic[f1_dim]["size"]
ppm_per_pt_f2 = (udic[f2_dim]["sw"] / udic[f2_dim]["obs"]) / udic[f2_dim]["size"]

# convert radii from ppm to points
pt_per_ppm_f1 = 1.0 / ppm_per_pt_f1
pt_per_ppm_f2 = 1.0 / ppm_per_pt_f2

# set x_radius and y_radius to points
x_radius = x_radius * pt_per_ppm_f2
y_radius = y_radius * pt_per_ppm_f1

# data = rescale_intensity(data, in_range=(-1,1))
#  rearrange data if dims not in standard order
if dims != [0, 1, 2]:
    data = np.transpose(data, dims)

# sum planes for initial fit
summed_planes = data.sum(axis=0)

# make weights for fit
# noise = args.get("--noise")
# if noise == "None":
#    weights = None
#    print("You did not provide a noise estimate!")
#
# else:
#    weights = np.empty(summed_planes.shape)
#    weights[:, :] = float(noise) ** 2.0
#    weight = float(noise) ** 2.0
# print(weights)


def pool_func(name, group):
    # for saving data
    amps = []
    amp_errs = []

    center_xs = []
    center_x_errs = []

    center_ys = []
    center_y_errs = []

    sigma_ys = []
    sigma_y_errs = []

    sigma_xs = []
    sigma_x_errs = []

    fractions = []
    names = []
    indices = []
    assign = []
    clustids = []
    # vclists = []
    #    for name, group in groups:
    print(f"Running fit for cluster {name}")
    #  max cluster size
    if len(group) <= max_cluster_size:
        # fits sum of all planes first
        # first, mask = fit_first_plane(group, summed_planes, plot=True)
        print("Before fit_first")
        first, mask = fit_first_plane(
            group,
            summed_planes,
            x_radius,
            y_radius,
            uc_dics,
            lineshape=lineshape,
            plot=plot,
            show=args.get("--show"),
        )
        print("After fit_first")

        # fix sigma center and fraction parameters
        # could add an option to select params to fix
        to_fix = ["sigma", "center", "fraction"]
        # to_fix = ["center", "fraction"]
        fix_params(first.params, to_fix)

        # r_sq = r_square(summed_planes[mask], first.residual)
        # print("R^2 = ", r_sq)
        # print(first.residual,np.sum(first.residual**2.)/(28*25e8))
        # print(np.corrcoef(summed_planes[mask].ravel(),first.best_fit))
        # plt.plot(first.residual.ravel())
        # plt.show()
        # exit()
        # chi_sq = chi_square(first.residual, weights[mask].ravel())
        # r_sq = r_square(data[0][mask], first.residual)
        # print("Chi square = ",chi_sq)
        # if r_sq <= min_rsq:
        #    failed_ass = "\n".join(i for i in group.ASS)
        #    error = f"Fit failed for {name} with R2 or {r_sq}"
        #    print(error)
        #    log.write("\n--------------------------------------\n")
        #    log.write(f"{error}\n")
        #    log.write(f"{failed_ass}\n")
        #    log.write("\n--------------------------------------\n")

        # else:
        # get amplitudes and errors fitted from first plane
        # amp, amp_err, name = get_params(first.params,"amplitude")

        # fit all plane amplitudes while fixing sigma/center/fraction
        # refitting first plane reduces the error
        for d in data:
            first.fit(data=d[mask], params=first.params)  # noise=weights[mask].ravel())
            # print(first.fit_report())
            # r_sq = r_square(d[mask], first.residual)
            # chi_sq = chi_square(first.residual, weight)

            # print("R^2 = ", r_sq)
            # print("X^2 = ", chi_sq)
            amp, amp_err, name = get_params(first.params, "amplitude")
            cen_x, cen_x_err, name = get_params(first.params, "center_x")
            cen_y, cen_y_err, name = get_params(first.params, "center_y")
            sig_x, sig_x_err, name = get_params(first.params, "sigma_x")
            sig_y, sig_y_err, name = get_params(first.params, "sigma_y")
            frac, frac_err, name = get_params(first.params, "fraction")

            amps.extend(amp)
            amp_errs.extend(amp_err)

            center_xs.extend(cen_x)
            center_x_errs.extend(cen_x_err)

            center_ys.extend(cen_y)
            center_y_errs.extend(cen_y_err)

            sigma_xs.extend(sig_x)
            sigma_x_errs.extend(sig_x_err)

            sigma_ys.extend(sig_y)
            sigma_y_errs.extend(sig_y_err)

            fractions.extend(frac)
            #  get prefix for fit
            names.extend([i.replace("fraction", "") for i in name])
            assign.extend(group["ASS"])
            clustids.extend(group["CLUSTID"])

            # if add_vclist:
            #    vclists.extend(vclist)

            # print(plane.fit_report())
        # exit()
    df_dic = {
        "fit_prefix": np.ravel(names),
        "assignment": np.ravel(assign),
        "amp": np.ravel(amps),
        "amp_err": np.ravel(amp_errs),
        "center_x": np.ravel(center_xs),
        # "center_x_err": np.ravel(center_x_errs),
        "center_y": np.ravel(center_ys),
        # "center_y_err": np.ravel(center_y_errs),
        "sigma_x": np.ravel(sigma_xs),
        # "sigma_x_err": np.ravel(sigma_x_errs),
        "sigma_y": np.ravel(sigma_ys),
        # "sigma_y_err": np.ravel(sigma_y_errs),
        "fraction": np.ravel(fractions),
        "clustid": np.ravel(clustids),
        # "vclist": np.ravel(vclists)
    }

    # remove vclist if not using
    # if not add_vclist:
    #    df_dic.pop("vclist")

    #  make dataframe
    # return df_dic
    df = pd.DataFrame(df_dic)
    print("Making df")
    return df


# iterate over groups of peaks
#  split into groups for each cpu
# split_groups = np.array_split(groups, cpu_count)

dfs = []


def append_dfs(df):
    print("Calling append_dfs")
    dfs.append(df)


# with Pool(cpu_count) as p:
p = Pool(cpu_count)
print(f"Splitting over {cpu_count} cores")

# for x in split_groups:
#    print(x.shape)
for num, group in groups:
    p.apply(pool_func, (num, group))  # , callback=append_dfs)
# p.map(pool_func, split_groups)
p.close()
# p.join()
# p.map_async(pool_func, split_groups, callback=append_dfs)
print("Finished imap")
# df = pd.concat(dfs)
# print("Finished concat")
#
##  convert sigmas to fwhm based on the model used to fit
# if lineshape == "PV":
#    # fwhm = 2*sigma
#    df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
#    df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)
# elif lineshape == "G":
#    # fwhm = 2*sigma * sqrt(2*ln2)
#    df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.3548)
#    df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.3548)
# else:
#    df["fwhm_x"] = df.sigma_x.apply(lambda x: x)
#    df["fwhm_y"] = df.sigma_y.apply(lambda x: x)
#
##  convert values to ppm
# df["center_x_ppm"] = df.center_x.apply(lambda x: uc_f2.ppm(x))
# df["center_y_ppm"] = df.center_y.apply(lambda x: uc_f1.ppm(x))
# df["sigma_x_ppm"] = df.sigma_x.apply(lambda x: x * ppm_per_pt_f2)
# df["sigma_y_ppm"] = df.sigma_y.apply(lambda x: x * ppm_per_pt_f1)
# df["fwhm_x_ppm"] = df.fwhm_x.apply(lambda x: x * ppm_per_pt_f2)
# df["fwhm_y_ppm"] = df.fwhm_y.apply(lambda x: x * ppm_per_pt_f1)
## Fill nan values
# df.fillna(value=np.nan, inplace=True)
## calculate errors and square amplitudes
# df["amp_err"] = df.apply(lambda x: 2.0 * (x.amp_err / x.amp) * x.amp ** 2.0, axis=1)
# df["amp"] = df.amp.apply(lambda x: x ** 2.0)
## df.to_pickle("fits.pkl")
# output = Path(args["<output>"])
# suffix = output.suffix
# if suffix == ".csv":
#    df.to_csv(output)
#
# elif suffix == ".tab":
#    df.to_csv(output, sep="\t")
#
# else:
#    df.to_pickle(output)
