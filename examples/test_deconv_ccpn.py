#!/Users/jacobbrady/virtual_envs/py37/bin/python
""" Script to deconvolute NMR peaks

    TODO:
        Sort out Peak Class dependency i.e. (test_ccpn.py)
        Split processes across multiple cores
"""
import pickle
import os

import nmrglue as ng
import numpy as np
import yaml
from lmfit import Model, report_fit
from lmfit.models import ExponentialModel
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

from peak_deconvolution.core import pvoigt2d, update_params, make_models, fix_params, get_params, r_square


config = yaml.load(open("config.yml"))

INPUT_DATA = config["data"]
INPUT_PEAKS = "temp.pkl"

# set up paths
if config.get("dir"):
    BASE_DIR = os.path.abspath(config["dir"])
else:
    BASE_DIR = os.path.abspath("./")
    print("No basedir specified! Using ./ ...")

if config.get("outname"):
    OUTPUT_DIR = config["outname"]
else:
    OUTPUT_DIR = "out"
    print("No outdir specified! Using ./out ...")

OUTPUT_DIR = os.path.join(BASE_DIR,OUTPUT_DIR)
INPUT_PEAKS = os.path.join(BASE_DIR,INPUT_PEAKS)

with open(INPUT_PEAKS,"rb") as input_peaks:
    peaks = pickle.load(input_peaks)

#dic, data = ng.pipe.read(INPUT_DATA)
#uc_x = ng.pipe.make_uc(dic,data,dim=1)
#uc_y = ng.pipe.make_uc(dic,data,dim=0)
dic, data = ng.pipe.read(INPUT_DATA)
print("Loaded spectrum with shape:", data.shape)
uc_x = ng.pipe.make_uc(dic,data,dim=config.get("x", 2))
uc_y = ng.pipe.make_uc(dic,data,dim=config.get("y", 1))
#print(uc_x("8 ppm"))
#print(uc_y("108 ppm"))
#print(data.shape)
for p in peaks:
    p.center_x = uc_x(p.center_x, "PPM")
    p.center_y = uc_y(p.center_y, "PPM")
    # probably avoid this silliness somehow
    p.prefix = p.assignment.replace("{", "_")\
                .replace("}", "_")\
                .replace("[", "_")\
                .replace("]", "_")\
                .replace(",", "_")
    print(p)

group_of_peaks = peaks

x_radius = 5
y_radius = 5

mod, p_guess = make_models(pvoigt2d, group_of_peaks)

x = np.arange(1, data.shape[-1]+1)
y = np.arange(1, data.shape[-2]+1)
XY = np.meshgrid(x, y)
X, Y = XY
print("X", X.shape, "Y", Y.shape)
#for group in groups_of_peaks:


def fit_first_plane(group, data):

    mask = np.zeros(data.shape, dtype=bool)
    mod, p_guess = make_models(pvoigt2d, group)
    cen_x = [p_guess[k].value for k in p_guess if "center_x" in k ]
    cen_y = [p_guess[k].value for k in p_guess if "center_y" in k ]
    for peak in group:
        mask += peak.mask(data, x_radius, y_radius)
        print(peak)

    max_x, min_x = int(round(max(cen_x)))+x_radius, int(round(min(cen_x)))-x_radius
    max_y, min_y = int(round(max(cen_y)))+y_radius, int(round(min(cen_y)))-y_radius
    peak_slices = data.copy()[mask]
    XY_slices = [X.copy()[mask], Y.copy()[mask]]
    out = mod.fit(peak_slices, XY=XY_slices, params=p_guess)
    Zsim = mod.eval(XY=XY, params=out.params)
    print(report_fit(out.params))
    Zsim[~mask] = np.nan

    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Z_plot = data.copy()
    Z_plot[~mask] = np.nan
    ax.plot_wireframe(X[min_y-1:max_y, min_x-1:max_x],
                      Y[min_y-1:max_y, min_x-1:max_x],
                      Z_plot[min_y-1:max_y, min_x-1:max_x])
    ax.set_xlabel("x")
    ax.set_title("$R^2=%.3f$" % r_square(peak_slices.ravel(), out.residual))
    ax.plot_wireframe(X[min_y-1:max_y, min_x-1:max_x],
            Y[min_y-1:max_y, min_x-1:max_x],
            Zsim[min_y-1:max_y, min_x-1:max_x], color="r")
    plt.show()
#    print(p_guess)
    return out, mask

# fit first plane of data
first, mask = fit_first_plane(group_of_peaks,data[0])
# fix sigma center and fraction parameters
to_fix = ["sigma","center","fraction"]
fix_params(first.params,to_fix)
# get amplitudes and errors fitted from first plane
# amp, amp_err, name = get_params(first.params,"amplitude")
amps = []
amp_errs = []
names = []
# fit all plane amplitudes while fixing sigma/center/fraction
# refitting first plane reduces the error
for d in data:
    first.fit(data=d[mask], params=first.params)
    print(first.fit_report())
    print("R^2 = ", r_square(d[mask],first.residual))
    amp, amp_err, name = get_params(first.params,"amplitude") 
    amps.append(amp)
    amp_errs.append(amp_err)
    names.append(name)
#    print(plane.fit_report())
amps = np.vstack(amps)
names = np.vstack(names)
amp_errs = np.vstack(amp_errs)
print(names,amps)
#plot fits
for i in range(len(amps[0])):
    A = amps[:, i]
    x = np.array([1,2,3,4,5,6])
    mod = ExponentialModel()
    pars = mod.guess(A,x=x)
    out = mod.fit(A,pars, x=x)
    x_sort = np.argsort(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x[x_sort],out.best_fit[x_sort],"--")
    ax.errorbar(x, A, yerr=amp_errs[:, i], fmt="ro", label=names[0, i])
    ax.set_title(names[0,i])
    ax.legend()
    plt.show()

#class Index(object):
#    ind = 0
#
#    def no(self, event):
#        pass
#        plt.close()
#
#    def yes(self, event):
#        self.ind += 1
#        print(self.ind)
#        plt.close()
##
#callback = Index()
#plt.figure(figsize=(1,1))
#axprev = plt.axes([0.05, 0.05, 0.45, 0.75])
#axnext = plt.axes([0.5, 0.05, 0.45, 0.75])
#bnext = Button(axnext, 'No')
#bnext.on_clicked(callback.no)
#bprev = Button(axprev, 'Yes')
#bprev.on_clicked(callback.yes)
#plt.show()
