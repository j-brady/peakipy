#!/usr/bin/env python3
"""Fit and deconvolute NMR peaks

    Usage:
        fit_pipe_peaks.py <peaklist> <data> <output> [options]

    Options:
        -h --help  Show this page
        -v --version Show version

        --max_cluster_size=<max_cluster_size>  Maximum size of cluster to fit (i.e exclude large clusters) [default: None]
        --x_radius=<points>  x_radius in points for fit mask [default: 6]
        --y_radius=<points>  y_radius in points for fit mask [default: 5]
        --min_rsq=<float>  minimum R2 required to accept fit [default: 0.85]

        --show  Whether to show wireframe fits for each peak


    ToDo: 
        1. per peak R2, fit first summed spec (may need to adjust start params for this)
        2. currently x/y_radius has to be in int points since it is used to index
        3. decide clusters based on lw?
        4. add vclist data to output?
        5. add threshold to R2 so that you just give an error for the fit and suggest reselecting the group.
        6. estimate lw since fit lw seems to be ~0.5 of estimate from analysis (may throw off fit ).
        7. convert scales to ppm


"""
import os
import sys

import yaml
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


def make_param_dict(peaks):
    """ Make dict of parameter names using prefix """

    param_dict = {}

    for index, peak in peaks.iterrows():

        str_form = lambda x: "_%s_%s" % (peak.INDEX, x)
        # using exact value of points (i.e decimal)
        param_dict[str_form("center_x")] = peak.X_AXISf
        param_dict[str_form("center_y")] = peak.Y_AXISf
        param_dict[str_form("sigma_x")] = peak.XW/2.
        param_dict[str_form("sigma_y")] = peak.YW/2.
        param_dict[str_form("amplitude")] = peak.HEIGHT
        param_dict[str_form("fraction")] = 0.5

    return param_dict


def make_models(model, peaks):
    """ Make composite models for multiple peaks

        Arguments:
            -- models
            -- peaks: list of Peak objects [<Peak1>,<Peak2>,...]

        Returns:
            -- mod: Composite model containing all peaks
            -- p_guess: params for composite model with starting values

        Maybe add mask making to this function
    """
    if len(peaks) == 1:
        # make model for first peak
        mod = Model(model, prefix="_%d_" % peaks.INDEX)
        # add parameters
        p_guess = mod.make_params(**make_param_dict(peaks))

    elif len(peaks) > 1:
        # make model for first peak
        first_peak, *remaining_peaks = peaks.iterrows()
        mod = Model(model, prefix="_%d_" % first_peak[1].INDEX)
        for index, peak in remaining_peaks:
            mod += Model(model, prefix="_%d_" % peak.INDEX)

        p_guess = mod.make_params(**make_param_dict(peaks))
        # add Peak params to p_guess
        # update_params(p_guess, peaks)

    return mod, p_guess


def update_params(params, peaks):
    """ Update lmfit parameters with values from Peak

        Arguments:
             -- params: lmfit parameter object
             -- peaks: list of Peak objects that parameters correspond to

        ToDo:
             -- deal with boundaries
             -- currently positions in points
             --

    """
    for peak in peaks:
        # print(peak)
        for k, v in peak.param_dict().items():
            params[k].value = v
            print("update", k, v)
            if "center" in k:
                params[k].min = v - 2.5
                params[k].max = v + 2.5
                print(
                    "setting limit of %s, min = %.3e, max = %.3e"
                    % (k, params[k].min, params[k].max)
                )
            elif "sigma" in k:
                params[k].min = 0.0
                params[k].max = 1e6
                print(
                    "setting limit of %s, min = %.3e, max = %.3e"
                    % (k, params[k].min, params[k].max)
                )
            elif "fraction" in k:
                # fix weighting between 0 and 1
                params[k].min = 0.0
                params[k].max = 1.0


def fit_first_plane(group, data, x_radius, y_radius, plot=True):
    """
        Arguments:

            group -- pandas data from containing group of peaks
            data  -- 
    
    """

    mask = np.zeros(data.shape, dtype=bool)
    mod, p_guess = make_models(pvoigt2d, group)
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

    if plot:
        Zsim = mod.eval(XY=XY, params=out.params)
        print(report_fit(out.params))
        Zsim[~mask] = np.nan

        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        Z_plot = data.copy()
        Z_plot[~mask] = np.nan
        X_plot = X[min_y - 1 : max_y, min_x - 1 : max_x]
        Y_plot = Y[min_y - 1 : max_y, min_x - 1 : max_x]

        ax.plot_wireframe(X_plot, Y_plot, Z_plot[min_y - 1 : max_y, min_x - 1 : max_x])
        ax.set_xlabel("x")
        ax.set_title("$R^2=%.3f$" % r_square(peak_slices.ravel(), out.residual))
        ax.plot_wireframe(
            X_plot,
            Y_plot,
            Zsim[min_y - 1 : max_y, min_x - 1 : max_x],
            color="r",
            linestyle="--",
        )
        plt.show()
        #    print(p_guess)
    return out, mask


if __name__ == "__main__":
    args = docopt(__doc__)
    max_cluster_size = args.get("--max_cluster_size")
    if max_cluster_size == "None":
        max_cluster_size = 1000
    else:
        max_cluster_size = int(max_cluster_size)

    x_radius = int(args.get("--x_radius"))
    y_radius = int(args.get("--y_radius"))
    min_rsq = float(args.get("--min_rsq"))
    print("Using ", args)
    log = open("log.txt", "a")
    peaks = pd.read_pickle(args["<peaklist>"])
    groups = peaks.groupby("CLUSTID")
    # vcpmg = np.genfromtxt("vclist")
    # need to make these definable on a per peak basis
    #x_radius = 6
    #y_radius = 5
    # read NMR data
    dic, data = ng.pipe.read(args["<data>"])
    udic = ng.pipe.guess_udic(dic,data)
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
    # iterate over groups of peaks
    for name, group in groups:
        #  max cluster size
        if len(group) <= max_cluster_size:
            # fits sum of all planes first
            # first, mask = fit_first_plane(group, summed_planes, plot=True)
            first, mask = fit_first_plane(group, data[0], x_radius, y_radius, plot=args.get("--show"))
            # fix sigma center and fraction parameters
            to_fix = ["sigma", "center", "fraction"]
            #to_fix = ["center", "fraction"]
            
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

                    # print(plane.fit_report())
            #exit()
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
                "fractions": np.ravel(fractions),
            }
        )
        #  get peak numbers
        df["number"] = df.names.apply(lambda x: int(x.split("_")[1]))
        # df.to_pickle("fits.pkl")
        df.to_pickle(args["<output>"])

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
