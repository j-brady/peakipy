import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("fits_mp.csv")
groups = data.groupby("assignment")
t = np.genfromtxt("vclist")

for i, g in groups:
    plt.errorbar(
        t, g.amp, yerr=g.amp_err, label=g["assignment"].iloc[0], fmt="o", markersize=0.5
    )
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()
