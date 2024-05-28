""" Simultate some peaks add noise and fit using peakipy """

import numpy as np
import pandas as pd
from lmfit import Model
from skimage.filters import threshold_otsu
from nmrglue.fileio.fileiobase import unit_conversion
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from peakipy.lineshapes import pvoigt2d, Lineshape
from peakipy.fitting import (
    simulate_lineshapes_from_fitted_peak_parameters,
    make_models,
    make_meshgrid,
    FitPeaksArgs,
    FitPeaksInput,
    fit_peak_clusters,
)


uc_x = unit_conversion(400, False, 2400, 600, 4800)
uc_y = unit_conversion(200, False, 1200, 60, 7200)
uc_dics = {"f1": uc_y, "f2": uc_x}

p1 = dict(
    amp=10,
    center_x=200.0,
    center_y=100.0,
    sigma_x=10.0,
    sigma_y=18.0,
    fraction=0.5,
    lineshape="PV",
    ASS="one",
)

p2 = dict(
    amp=12,
    center_x=220.0,
    center_y=130.0,
    sigma_x=20.0,
    sigma_y=15.0,
    fraction=0.5,
    lineshape="PV",
    ASS="two",
)

peak_parameters = pd.DataFrame([p1, p2])
peak_parameters["X_AXIS"] = peak_parameters.center_x
peak_parameters["Y_AXIS"] = peak_parameters.center_y
peak_parameters["X_AXISf"] = peak_parameters.center_x
peak_parameters["Y_AXISf"] = peak_parameters.center_y
peak_parameters["X_RADIUS"] = 60
peak_parameters["Y_RADIUS"] = 30
peak_parameters["X_PPM"] = peak_parameters.center_x.apply(uc_x.ppm)
peak_parameters["Y_PPM"] = peak_parameters.center_x.apply(uc_x.ppm)
peak_parameters["XW"] = peak_parameters.sigma_x
peak_parameters["YW"] = peak_parameters.sigma_y
peak_parameters["CLUSTID"] = 1
peak_parameters["MEMCNT"] = peak_parameters.shape[0]
peak_parameters["plane"] = 0

x = 400
y = 200
data_shape_Y_X = (y, x)
data_shape_X_Y = (x, y)
XY = make_meshgrid(data_shape_Y_X)
X, Y = XY
Z_sim = np.random.normal(loc=0.0, scale=0.0001, size=data_shape_Y_X)
Z_sim_singles = []
Z_sim, Z_sim_singles = simulate_lineshapes_from_fitted_peak_parameters(
    peak_parameters=peak_parameters,
    XY=XY,
    sim_data=Z_sim,
    sim_data_singles=Z_sim_singles,
)

fit_peaks_args = FitPeaksArgs(
    noise=threshold_otsu(Z_sim),
    uc_dics=uc_dics,
    lineshape=Lineshape.PV,
    max_cluster_size=10,
    reference_plane_indices=[],
    xy_bounds=None,
    initial_fit_threshold=None,
    vclist=None,
)

fit_peaks_input = FitPeaksInput(
    fit_peaks_args, Z_sim.reshape(1, y, x), dict(dims=[0, 1]), plane_numbers=[0]
)

fit_peaks_result = fit_peak_clusters(peak_parameters, fit_peaks_input)


def test_fit_from_simulated_data():
    pd.testing.assert_series_equal(
        fit_peaks_result.df.center_x,
        fit_peaks_result.df.center_x_init,
        check_exact=False,
        check_names=False,
        rtol=1e-3,
    )
    pd.testing.assert_series_equal(
        fit_peaks_result.df.center_y,
        fit_peaks_result.df.center_y_init,
        check_exact=False,
        check_names=False,
        rtol=1e-3,
    )
    pd.testing.assert_series_equal(
        fit_peaks_result.df.sigma_x,
        fit_peaks_result.df.sigma_x_init,
        check_exact=False,
        check_names=False,
        rtol=1e-2,
    )
    pd.testing.assert_series_equal(
        fit_peaks_result.df.sigma_y,
        fit_peaks_result.df.sigma_y_init,
        check_exact=False,
        check_names=False,
        rtol=1e-2,
    )


def test_fit_from_simulated_data_jack_knife():
    fit_peaks_input = FitPeaksInput(
        fit_peaks_args, Z_sim.reshape(1, y, x), dict(dims=[0, 1]), plane_numbers=[0]
    )
    fit_peaks_input.args.jack_knife_sample_errors = True
    fit_peaks_result = fit_peak_clusters(peak_parameters, fit_peaks_input)


# def plot3D(X,Y,Z,Z_singles):
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.plot_wireframe(X,Y,Z)
#     for Z_single in Z_singles:
#         ax.plot_surface(X, Y, Z_single,alpha=0.5)
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()
