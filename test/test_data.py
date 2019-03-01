from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from docopt import docopt
from lmfit import Model, report_fit

from peakipy.core import (
    pvoigt2d,
    fix_params,
    get_params,
    r_square,
    make_mask,
    # fit_first_plane,
    make_models,
)


def fit_first_plane(
    group, data, x_radius, y_radius, noise=None, lineshape="PV", plot=None, show=True
):

    mask = np.zeros(data.shape, dtype=bool)
    mod, p_guess = make_models(pvoigt2d, group, data, lineshape=lineshape)
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
    out = mod.fit(
        peak_slices, XY=XY_slices, params=p_guess
    )  # , weights=weights[mask].ravel())

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
        X_plot = X[min_y - 1 : max_y, min_x - 1 : max_x]
        Y_plot = Y[min_y - 1 : max_y, min_x - 1 : max_x]

        ax.plot_wireframe(
            X_plot, Y_plot, Z_plot[min_y - 1 : max_y, min_x - 1 : max_x], color="k"
        )
        # ax.contour3D(X_plot, Y_plot, Z_plot[min_y - 1 : max_y, min_x - 1 : max_x],cmap='viridis')

        ax.set_xlabel("F2 pts")
        ax.set_ylabel("F1 pts")
        ax.set_title("$R^2=%.3f$" % r_square(peak_slices.ravel(), out.residual))
        ax.plot_wireframe(
            X_plot,
            Y_plot,
            Zsim[min_y - 1 : max_y, min_x - 1 : max_x],
            color="r",
            linestyle="--",
            label="fit",
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
                labs.append(" ".join(k.split("_")[:-1]))
            elif "center_x" in k:
                X_lab.append(v)
            elif "center_y" in k:
                Y_lab.append(v)
        #  this is dumb as !£$@
        Z_lab = [data[int(round(y)), int(round(x))] for x, y in zip(X_lab, Y_lab)]
        print(Z_lab)

        for l, x, y, z in zip(labs, X_lab, Y_lab, Z_lab):
            # print(l, x, y, z)
            ax.text(x, y, z * 1.4, l, None)

        plt.legend()

        name = group.CLUSTID.iloc[0]
        if show:
            plt.savefig(plot_path / f"{name}.png", dpi=300)
            plt.show()
        else:
            plt.savefig(plot_path / f"{name}.png", dpi=300)
        #    print(p_guess)
    return out, mask


params = {
    "id": {"planes": 3},
    "f1": {"sw": 2400.0, "obs": 80.0, "size": 256},
    "f2": {"sw": 12000.0, "obs": 800.0, "size": 512},
    "noise": 5e4,
}


# data = np.ones((params["id"]["planes"], params["f1"]["size"], params["f2"]["size"])) * params["noise"]

peaks = [
    [-1e5, 205, 103, 3, 3],
    [-0.6e5, 217, 105, 3, 3],
    [-0.5e5, 210, 110, 3, 3],
    [-1e5, 100, 90, 2, 2],
    [-0.6e5, 100, 96, 2, 2],
    [-0.5e5, 94, 84, 2, 2],
    [-0.5e5, 95, 90, 2, 2],
]

data = np.zeros((params["id"]["planes"], params["f1"]["size"], params["f2"]["size"]))
data = data[0]
mask = np.zeros(data.shape, dtype=bool)

XY = np.meshgrid(np.arange(params["f2"]["size"]), np.arange(params["f1"]["size"]))
for p in peaks:
    data += pvoigt2d(
        XY,
        *p
        # amplitude=1e8,
        # center_x=200,
        # center_y=100,
        # sigma_x=5,
        # sigma_y=5,
    )

    mask += make_mask(data, p[1], p[2], p[3] * 2, p[4] * 2)

#  add noise
data = data + data * np.random.normal(1, 0.10, data.shape)
# data[~mask] = np.nan

peak_dic = {
    "ASS": ["test1", "test2", "test3", "test4", "test5", "test6", "test7"],
    "X_AXISf": [205, 217, 210, 100, 100, 94, 95],
    "Y_AXISf": [103, 105, 110, 90, 96, 84, 90],
    "X_AXIS": [205, 217, 210, 100, 100, 94, 95],
    "Y_AXIS": [103, 105, 110, 90, 96, 84, 90],
    "XW": [6, 6, 6, 4, 4, 4, 4],
    "YW": [6, 6, 6, 4, 4, 4, 4],
    "CLUSTID": [1, 1, 1, 2, 2, 2, 2],
}

peak_df = pd.DataFrame(peak_dic)

peaks = peak_df.groupby("CLUSTID")

x_radius = 6
y_radius = 6

for name, group in peaks:
    # fits sum of all planes first
    # first, mask = fit_first_plane(group, summed_planes, plot=True)
    first, mask = fit_first_plane(
        group,
        # data[0],
        data,
        x_radius,
        x_radius,
        lineshape="PV",
        plot="test_out",
        show=True,
    )

# min_x = int(np.floor(min_x))
# max_x = int(np.ceil(max_x))
# min_y = int(np.floor(min_y))
# max_y = int(np.ceil(max_y))

# min_x = 190
# max_x = 230
# min_y = 90
# max_y = 120
# X,Y = XY
#
# X_plot = X[min_y - 1 : max_y, min_x - 1 : max_x]
# Y_plot = Y[min_y - 1 : max_y, min_x - 1 : max_x]
# Z_plot = data[min_y - 1 : max_y, min_x - 1 : max_x]
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_wireframe(X_plot,Y_plot,Z_plot)
##plt.imshow(data)
# plt.show()
