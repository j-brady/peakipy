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
        --x_radius=<points>                    x_radius in ppm for fit mask [default: 0.05]
        --y_radius=<points>                    y_radius in ppm for fit mask [default: 0.5]
        --min_rsq=<float>                      minimum R2 required to accept fit [default: 0.85]
        --lineshape=<G/L/PV>                   lineshape to fit [default: PV]

        --plot=<dir>                           Whether to plot wireframe fits for each peak 
                                               (saved into <dir>) [default: None]

        --show                                 Whether to show (using plt.show()) wireframe
                                               fits for each peak

    ToDo: 
        1. per peak R^2, fit first summed spec (may need to adjust start params for this)
        2. currently x/y_radius has to be in int points since it is used to index
        3. decide clusters based on lw?
        4. add vclist data to output?
        5. add threshold to R2 so that you just give an error for the fit and suggest reselecting the group.
        6. estimate lw since fit lw seems to be ~0.5 of estimate from analysis (may throw off fit ).
        7. estimate peak height 


"""
import os
from pathlib import Path

import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lmfit import Model, report_fit
from mpl_toolkits.mplot3d import Axes3D
from docopt import docopt

from peak_deconvolution.core import (
    pvoigt2d,
    update_params,
    make_models,
    fix_params,
    get_params,
    r_square,
    make_mask,
)


def make_param_dict(peaks, lineshape="PV"):
    """ Make dict of parameter names using prefix """

    param_dict = {}

    for index, peak in peaks.iterrows():

        str_form = lambda x: "_%s_%s" % (peak.INDEX, x)
        # using exact value of points (i.e decimal)
        param_dict[str_form("center_x")] = peak.X_AXISf
        param_dict[str_form("center_y")] = peak.Y_AXISf
        param_dict[str_form("sigma_x")] = peak.XW / 2.0
        param_dict[str_form("sigma_y")] = peak.YW / 2.0
        param_dict[str_form("amplitude")] = peak.HEIGHT
        if lineshape == "G":
            param_dict[str_form("fraction")] = 0.0
        elif lineshape == "L":
            param_dict[str_form("fraction")] = 1.0
        else:
            param_dict[str_form("fraction")] = 0.5

    return param_dict


def make_models(model, peaks, lineshape="PV"):
    """ Make composite models for multiple peaks

        Arguments:
            -- models
            -- peaks: list of Peak objects [<Peak1>,<Peak2>,...]
            -- lineshape: PV/G/L

        Returns:
            -- mod: Composite model containing all peaks
            -- p_guess: params for composite model with starting values

        Maybe add mask making to this function
    """
    if len(peaks) == 1:
        # make model for first peak
        mod = Model(model, prefix="_%d_" % peaks.INDEX)
        # add parameters
        param_dict = make_param_dict(peaks, lineshape=lineshape)
        p_guess = mod.make_params(**param_dict)

    elif len(peaks) > 1:
        # make model for first peak
        first_peak, *remaining_peaks = peaks.iterrows()
        mod = Model(model, prefix="_%d_" % first_peak[1].INDEX)
        for index, peak in remaining_peaks:
            mod += Model(model, prefix="_%d_" % peak.INDEX)

        param_dict = make_param_dict(peaks, lineshape=lineshape)
        p_guess = mod.make_params(**param_dict)
        # add Peak params to p_guess

    update_params(p_guess, param_dict, lineshape=lineshape)

    return mod, p_guess


def update_params(params, param_dict, lineshape="PV"):
    """ Update lmfit parameters with values from Peak

        Arguments:
             -- params: lmfit parameter object
             -- peaks: list of Peak objects that parameters correspond to

        ToDo:
             -- deal with boundaries
             -- currently positions in points
             --

    """
    for k, v in param_dict.items():
        params[k].value = v
        print("update", k, v)
        #if "center" in k:
        #    params[k].min = v - 10
        #    params[k].max = v + 10
        #    print(
        #        "setting limit of %s, min = %.3e, max = %.3e"
        #        % (k, params[k].min, params[k].max)
        #    )
        if "sigma" in k:
            params[k].min = 0.0
            params[k].max = 1e4
            print(
                "setting limit of %s, min = %.3e, max = %.3e"
                % (k, params[k].min, params[k].max)
            )
        elif "fraction" in k:
            # fix weighting between 0 and 1
            params[k].min = 0.0
            params[k].max = 1.0

            if lineshape == "G":
                params[k].vary = False
            elif lineshape == "L":
                params[k].vary = False

    # return params


def fit_first_plane(
    group, data, x_radius, y_radius, uc_dics, lineshape="PV", plot=None, show=True
):
    """
        Arguments:

            group -- pandas data from containing group of peaks
            data  -- 
            x_radius 
            y_radius
            uc_dics -- unit conversion dics
            plot -- if True show wireframe plots

        To do:
            add model selection
    
    """

    mask = np.zeros(data.shape, dtype=bool)
    mod, p_guess = make_models(pvoigt2d, group, lineshape=lineshape)
    # print(p_guess)
    # get initial peak centers
    cen_x = [p_guess[k].value for k in p_guess if "center_x" in k]
    cen_y = [p_guess[k].value for k in p_guess if "center_y" in k]

    for index, peak in group.iterrows():
        #  minus 1 from X_AXIS and Y_AXIS to center peaks in mask
        # print(peak.X_AXIS,peak.Y_AXIS,row.HEIGHT)
        mask += make_mask(data, peak.X_AXISf, peak.Y_AXISf, x_radius, y_radius)
        # print(peak)

    # needs checking since this may not center peaks
    max_x, min_x = int(round(max(cen_x))) + x_radius, int(round(min(cen_x))) - x_radius
    max_y, min_y = int(round(max(cen_y))) + y_radius, int(round(min(cen_y))) - y_radius

    peak_slices = data.copy()[mask]

    # must be a better way to make the meshgrid
    x = np.arange(1, data.shape[-1] + 1)
    y = np.arange(1, data.shape[-2] + 1)
    XY = np.meshgrid(x, y)
    X, Y = XY

    XY_slices = [X.copy()[mask], Y.copy()[mask]]
    out = mod.fit(peak_slices, XY=XY_slices, params=p_guess)

    if plot != None:
        plot_path = Path(plot)
        Zsim = mod.eval(XY=XY, params=out.params)
        print(report_fit(out.params))
        Zsim[~mask] = np.nan

        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        Z_plot = data.copy()
        Z_plot[~mask] = np.nan
        # convert to ints may need tweeking
        min_x = int(np.floor(min_x))
        max_x = int(np.ceil(max_x))
        min_y = int(np.floor(min_y))
        max_y = int(np.ceil(max_y))
        X_plot = uc_dics["f2"].ppm(X[min_y - 1 : max_y, min_x - 1 : max_x])
        Y_plot = uc_dics["f1"].ppm(Y[min_y - 1 : max_y, min_x - 1 : max_x])

        ax.plot_wireframe(X_plot, Y_plot, Z_plot[min_y - 1 : max_y, min_x - 1 : max_x])
        ax.set_xlabel("F2 ppm")
        ax.set_ylabel("F1 ppm")
        ax.set_title("$R^2=%.3f$" % r_square(peak_slices.ravel(), out.residual))
        ax.plot_wireframe(
            X_plot,
            Y_plot,
            Zsim[min_y - 1 : max_y, min_x - 1 : max_x],
            color="r",
            linestyle="--",
        )
        # Annotate plots
        labs = []
        Z_lab = []
        Y_lab = []
        X_lab = []
        for k, v in out.params.valuesdict().items():

            if "amplitude" in k:
                Z_lab.append(v)
                # get prefix
                labs.append(k.split("_")[1])
            elif "center_x" in k:
                X_lab.append(uc_dics["f2"].ppm(v))
            elif "center_y" in k:
                Y_lab.append(uc_dics["f1"].ppm(v))

        for l, x, y, z in zip(labs, X_lab, Y_lab, Z_lab):
            print(l, x, y, z)
            ax.text(x, y, z, l, None)

        name = group.CLUSTID.iloc[0]
        if show:
            plt.savefig(plot_path / f"{name}.png", dpi=300)
            plt.show()
        else:
            plt.savefig(plot_path / f"{name}.png", dpi=300)
        #    print(p_guess)
    return out, mask


if __name__ == "__main__":
    args = docopt(__doc__)
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
    #
    peaklist = args.get("<peaklist>")
    if os.path.splitext(peaklist)[-1] == ".csv":
        peaks = pd.read_csv(peaklist)
    else:
        peaks = pd.read_pickle(peaklist)

    # peaks = pd.read_pickle(args["<peaklist>"])
    groups = peaks.groupby("CLUSTID")

    plot = args.get("--plot")
    if plot == "None":
        plot = None
    else:
        if os.path.exists(plot) and os.path.isdir(plot):
            pass
        else:
            os.mkdir(plot)
    # vcpmg = np.genfromtxt("vclist")
    # need to make these definable on a per peak basis
    # x_radius = 6
    # y_radius = 5
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

    # convert radii from ppm to points
    pt_per_ppm_f1 = udic[f1_dim]["size"]/(udic[f1_dim]["sw"]/udic[f1_dim]["obs"])
    pt_per_ppm_f2 = udic[f2_dim]["size"]/(udic[f2_dim]["sw"]/udic[f2_dim]["obs"])
    x_radius = x_radius * pt_per_ppm_f2
    y_radius = y_radius * pt_per_ppm_f1 
    print(x_radius,y_radius)
    # sum planes for initial fit
    # summed_planes = data.sum(axis=0)
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
    # iterate over groups of peaks
    for name, group in groups:
        #  max cluster size
        if len(group) <= max_cluster_size:
            # fits sum of all planes first
            # first, mask = fit_first_plane(group, summed_planes, plot=True)

            first, mask = fit_first_plane(
                group,
                data[0],
                x_radius,
                y_radius,
                uc_dics,
                lineshape=lineshape,
                plot=plot,
                show=args.get("--show"),
            )
            # fix sigma center and fraction parameters
            to_fix = ["sigma", "center", "fraction"]
            # to_fix = ["center", "fraction"]

            fix_params(first.params, to_fix)

            # r_sq = r_square(summed_planes[mask], first.residual)
            r_sq = r_square(data[0][mask], first.residual)
            if r_sq <= min_rsq:
                failed_ass = "\n".join(i for i in group.ASS)
                error = f"Fit failed for {name} with R2 or {r_sq}"
                print(error)
                log.write("\n--------------------------------------\n")
                log.write(f"{error}\n")
                log.write(f"{failed_ass}\n")
                log.write("\n--------------------------------------\n")

            else:
                # get amplitudes and errors fitted from first plane
                # amp, amp_err, name = get_params(first.params,"amplitude")

                # fit all plane amplitudes while fixing sigma/center/fraction
                # refitting first plane reduces the error
                for d in data:
                    first.fit(data=d[mask], params=first.params)
                    print(first.fit_report())
                    r_sq = r_square(d[mask], first.residual)

                    print("R^2 = ", r_sq)
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
                    names.extend(name)
                    assign.extend(group["ASS"])
                    clustids.extend(group["CLUSTID"])

                    # print(plane.fit_report())
            # exit()
        df = pd.DataFrame(
            {
                "names": np.ravel(names),
                "assign": np.ravel(assign),
                "amp": np.ravel(amps),
                "amp_err": np.ravel(amp_errs),
                "center_x": np.ravel(center_xs),
                # "center_x_err": np.ravel(center_x_errs),
                "center_y": np.ravel(center_ys),
                # "center_y_err": np.ravel(center_y_errs),
                "sigma_x": np.ravel(center_xs),
                # "sigma_x_err": np.ravel(sigma_x_errs),
                "sigma_y": np.ravel(center_ys),
                # "sigma_y_err": np.ravel(sigma_y_errs),
                "fraction": np.ravel(fractions),
                "clustid": np.ravel(clustids),
            }
        )
        # get peak numbers
        df["number"] = df.names.apply(lambda x: int(x.split("_")[1]))
        #  convert values to ppm
        df["center_x_ppm"] = df.center_x.apply(lambda x: uc_f2.ppm(x))
        df["center_y_ppm"] = df.center_y.apply(lambda x: uc_f1.ppm(x))
        df["sigma_x_ppm"] = df.sigma_x.apply(lambda x: uc_f2.ppm(x))
        df["sigma_y_ppm"] = df.sigma_y.apply(lambda x: uc_f1.ppm(x))
        # df.to_pickle("fits.pkl")
        #
        output = args["<output>"]
        extension = os.path.splitext(output)[-1]
        if extension == ".csv":
            df.to_csv(output)
        elif extension == ".tab":
            df.to_csv(output, sep="\t")
        else:
            df.to_pickle(output)

    # indices = np.vstack(

    #    print(amps)
    #    print(names)
    #    exit()
    # else:
    #    pass
    # plt.errorbar(vcpmg,np.array(amps),yerr=np.array(amp_errs),fmt="ro")
    # plt.show()
    # break
    # first, *rest = groups
    # print(first)
    # print(rest)


#    def cpmgfunc(T, ratio):
#        return np.log(ratio) / -T
#
#
#    def fit(data):
#        # path = os.path.join(out_path,f)
#        # data = np.genfromtxt(path,comments="#")
#        norm = data[:, 1] / data[:, 1][0]
#        reff = cpmgfunc(T=40e-3, ratio=norm)
#        errs = norm * np.sqrt(
#            (data[:, 2] / data[:, 1]) ** 2.0 + (data[:, 2][0] / data[:, 1][0]) ** 2.0
#        )
#        # plt.errorbar(vcpmg,norm,yerr=errs,fmt="ro")
#        plt.errorbar(vcpmg, reff, fmt="ro")
#        plt.title(path)
#        plt.ylabel(r"$R_{2}^{eff}(s^{-1})$")
#        plt.xlabel("$N_{180}$")
#        # plt.ylim(5,max(reff)+5)
#        plt.show()
#
#
#    if __name__ == "__main__":
#
#        arguments = docopt(__doc__, version="1.0")
#        print(arguments)
#
#        config = yaml.load(open(arguments["<config>"]))
#        #print(config)
#
#        INPUT_DATA = config["data"]
#
#        if arguments.get("<peaklist>"):
#
#            INPUT_PEAKS = arguments["<peaklist>"]
#        else:
#            INPUT_PEAKS = config["peaklist"]
#
#        # set up paths
#        if config.get("dir"):
#            BASE_DIR = os.path.abspath(config["dir"])
#        else:
#            BASE_DIR = os.path.abspath("./")
#            print("No basedir specified! Using ./ ...")
#
#        if config.get("outname"):
#            OUTPUT_DIR = config["outname"]
#        else:
#            OUTPUT_DIR = "out"
#            print("No outdir specified! Using ./out ...")
#
#        OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR)
#        INPUT_PEAKS_PATH = os.path.join(BASE_DIR, INPUT_PEAKS)
#
#        with open(INPUT_PEAKS_PATH, "rb") as input_peaks:
#            peaks = pickle.load(input_peaks)
#
#        # dic, data = ng.pipe.read(INPUT_DATA)
#        # uc_x = ng.pipe.make_uc(dic,data,dim=1)
#        # uc_y = ng.pipe.make_uc(dic,data,dim=0)
#        dic, data = ng.pipe.read(INPUT_DATA)
#        print("Loaded spectrum with shape:", data.shape)
#        uc_x = ng.pipe.make_uc(dic, data, dim=config.get("x", 2))
#        uc_y = ng.pipe.make_uc(dic, data, dim=config.get("y", 1))
#        # print(uc_x("8 ppm"))
#        # print(uc_y("108 ppm"))
#        # print(data.shape)
#        for p in peaks:
#            p.center_x = uc_x(p.center_x, "PPM")
#            p.center_y = uc_y(p.center_y, "PPM")
#            # probably avoid this silliness somehow
#            p.prefix = (
#                p.assignment.replace("{", "_")
#                .replace("}", "_")
#                .replace("[", "_")
#                .replace("]", "_")
#                .replace(",", "_")
#            )
#            print(p)
#
#        # group_of_peaks = peaks
#
#        x_radius = config.get("x_radius")
#        y_radius = config.get("y_radius")
#
#        if INPUT_PEAKS == "singles.pkl":
#            peaks_ass_df = []
#            peaks_name_df = []
#            peaks_amp_df = []
#            peaks_amp_errs_df = []
#            for p in peaks:
#
#                mod, p_guess = make_models(pvoigt2d, [p])
#
#                x = np.arange(1, data.shape[-1] + 1)
#                y = np.arange(1, data.shape[-2] + 1)
#                XY = np.meshgrid(x, y)
#                X, Y = XY
#                print("X", X.shape, "Y", Y.shape)
#                # for group in groups_of_peaks:
#
#                # fit first plane of data
#                first, mask = fit_first_plane([p], data[0])
#                # fix sigma center and fraction parameters
#                to_fix = ["sigma", "center", "fraction"]
#                fix_params(first.params, to_fix)
#                # get amplitudes and errors fitted from first plane
#                # amp, amp_err, name = get_params(first.params,"amplitude")
#                amps = []
#                amp_errs = []
#                names = []
#                # fit all plane amplitudes while fixing sigma/center/fraction
#                # refitting first plane reduces the error
#                for d in data:
#                    first.fit(data=d[mask], params=first.params)
#                    print(first.fit_report())
#                    print("R^2 = ", r_square(d[mask], first.residual))
#                    amp, amp_err, name = get_params(first.params, "amplitude")
#                    amps.append(amp)
#                    amp_errs.append(amp_err)
#                    names.append(name)
#                #    print(plane.fit_report())
#
#                amps = np.vstack(amps)
#                names = np.vstack(names)
#                amp_errs = np.vstack(amp_errs)
#
#                peaks_name_df.append(names.ravel())
#                peaks_ass_df.append(p.assignment)
#                peaks_amp_df.append(amps.ravel())
#                peaks_amp_errs_df.append(amp_errs.ravel())
#                print(names, amps)
#
#            df = pd.DataFrame({"assignment":peaks_ass_df,"name":peaks_name_df,"amp":peaks_amp_df,"amp_errs":peaks_amp_errs_df})
#            df.to_pickle("single_fits.pkl")
#            df.to_pickle("single_fits.csv")
#                # plot fits
#                #for i in range(len(amps[0])):
#                #    A = amps[:, i]
#                #    x = np.array(np.genfromtxt("vclist"))
#                #    # mod = ExponentialModel()
#                #    # pars = mod.guess(A,x=x)
#                #    # out = mod.fit(A,pars, x=x)
#                #    x_sort = np.argsort(x)
#                #    fig = plt.figure()
#                #    ax = fig.add_subplot(111)
#                #    # ax.plot(x[x_sort],out.best_fit[x_sort],"--")
#
#                #    T = 20e-3
#                #    norm = A / A[0]
#                #    reff = np.log(norm) / -T
#                #    # reff_err = norm*np.sqrt((data[:,2]/data[:,1])**2.+(data[:,2][0]/data[:,1][0])**2.)
#                #    # ax.errorbar(x, A, yerr=amp_errs[:, i], fmt="ro", label=names[0, i])
#                #    ax.errorbar(x[1:], reff[1:], fmt="ro", label=names[0, i])
#
#                #    # ax.errorbar(x, A, yerr=amp_errs[:, i], fmt="ro", label=names[0, i])
#                #    ax.set_title(names[0, i])
#                #    ax.legend()
#                #    plt.show()
#
#        elif INPUT_PEAKS == "group.pkl":
#
#            mod, p_guess = make_models(pvoigt2d, peaks)
#
#            x = np.arange(1, data.shape[-1] + 1)
#            y = np.arange(1, data.shape[-2] + 1)
#            XY = np.meshgrid(x, y)
#            X, Y = XY
#            print("X", X.shape, "Y", Y.shape)
#            # for group in groups_of_peaks:
#
#            # fit first plane of data
#            first, mask = fit_first_plane(peaks, data[0])
#            # fix sigma center and fraction parameters
#            to_fix = ["sigma", "center", "fraction"]
#            fix_params(first.params, to_fix)
#            # get amplitudes and errors fitted from first plane
#            # amp, amp_err, name = get_params(first.params,"amplitude")
#            amps = []
#            amp_errs = []
#            names = []
#            # fit all plane amplitudes while fixing sigma/center/fraction
#            # refitting first plane reduces the error
#            for d in data:
#                first.fit(data=d[mask], params=first.params)
#                print(first.fit_report())
#                print("R^2 = ", r_square(d[mask], first.residual))
#                amp, amp_err, name = get_params(first.params, "amplitude")
#                amps.append(amp)
#                amp_errs.append(amp_err)
#                names.append(name)
#            #    print(plane.fit_report())
#            amps = np.vstack(amps)
#            names = np.vstack(names)
#            amp_errs = np.vstack(amp_errs)
#            print(names, amps)
#            # plot fits
#            for i in range(len(amps[0])):
#                A = amps[:, i]
#                x = np.array(np.genfromtxt("vclist"))
#                # mod = ExponentialModel()
#                # pars = mod.guess(A,x=x)
#                # out = mod.fit(A,pars, x=x)
#                x_sort = np.argsort(x)
#                fig = plt.figure()
#                ax = fig.add_subplot(111)
#                # ax.plot(x[x_sort],out.best_fit[x_sort],"--")
#                T = 40e-3
#                norm = A / A[0]
#                reff = np.log(norm) / -T
#                # reff_err = norm*np.sqrt((data[:,2]/data[:,1])**2.+(data[:,2][0]/data[:,1][0])**2.)
#                # ax.errorbar(x, A, yerr=amp_errs[:, i], fmt="ro", label=names[0, i])
#                ax.errorbar(x[1:], reff[1:], fmt="ro", label=names[0, i])
#                ax.set_title(names[0, i])
#                ax.legend()
#                plt.show()
#
#        # class Index(object):
#        #    ind = 0
#        #
#        #    def no(self, event):
#        #        pass
#        #        plt.close()
#        #
#        #    def yes(self, event):
#        #        self.ind += 1
#        #        print(self.ind)
#        #        plt.close()
#        ##
#        # callback = Index()
#        # plt.figure(figsize=(1,1))
#        # axprev = plt.axes([0.05, 0.05, 0.45, 0.75])
#        # axnext = plt.axes([0.5, 0.05, 0.45, 0.75])
#        # bnext = Button(axnext, 'No')
#        # bnext.on_clicked(callback.no)
#        # bprev = Button(axprev, 'Yes')
#        # bprev.on_clicked(callback.yes)
#        # plt.show()
