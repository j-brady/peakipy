import pandas as pd
import numpy as np
from lmfit.models import ExponentialModel
import matplotlib.pyplot as plt

data = pd.read_csv("fits.csv")

groups = data.groupby("assignment")


for ind, group in groups:
    plt.figure(figsize=(4, 3))
    plt.errorbar(
        group.vclist,
        group.amp,
        yerr=group.amp_err,
        fmt="o",
        markersize=10,
        label=group.assignment.iloc[0],
        c="#33a02c",
    )
    mod = ExponentialModel()
    pars = mod.guess(group.amp, x=group.vclist)
    fit = mod.fit(group.amp, x=group.vclist, params=pars)
    sim_x = np.linspace(group.vclist.min(), group.vclist.max())
    sim_y = fit.eval(x=sim_x)
    plt.plot(sim_x, sim_y, "-", c="#1f78b4", linewidth=4)
    plot = plt.gca()
    plot.axes.xaxis.set_ticklabels([])
    plot.axes.yaxis.set_ticklabels([])
    for axis in ["top", "bottom", "left", "right"]:
        plot.axes.spines[axis].set_linewidth(2.0)
    # plt.legend()
    plt.savefig("eg_fit.pdf")
    plt.show()
    exit()
