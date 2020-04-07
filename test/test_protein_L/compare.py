import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import column

#  read data
pipe_data = pd.read_csv("nlin.csv")
peakipy_data = pd.read_csv("fits.csv")
#  take first plane
peakipy_data = peakipy_data[peakipy_data.plane == 0]
#  make new columns called amp and ASS
pipe_data["amp"] = pipe_data["VOL"]
peakipy_data["ASS"] = peakipy_data["assignment"]

#  merge dataframes on assignment column
merged = pd.merge(peakipy_data, pipe_data, on="ASS")
source = ColumnDataSource(merged)
p1 = figure(
    tooltips=[("Assignment", "@assignment, @ASS")],
    x_axis_label="peakipy",
    y_axis_label="pipe",
    title="VOL",
)
p1.circle("amp_x", "VOL", source=source, size=10, alpha=0.75)
p2 = figure(
    tooltips=[("Assignment", "@assignment, @ASS")],
    x_axis_label="peakipy",
    y_axis_label="pipe",
    title="X LW",
)
p2.circle("fwhm_x", "XW", source=source, size=10, alpha=0.75)
p3 = figure(
    tooltips=[("Assignment", "@assignment, @ASS")],
    x_axis_label="peakipy",
    y_axis_label="pipe",
    title="Y LW",
)
p3.circle("fwhm_y", "YW", source=source, size=10, alpha=0.75)

#  make matplotlib figure
fig, axes = plt.subplots(2, 3, figsize=(9, 6))

ax1 = axes[0][0]
ax2 = axes[0][1]
ax3 = axes[0][2]
ax4 = axes[1][0]
ax5 = axes[1][1]
ax6 = axes[1][2]

axes = [ax1, ax2, ax3, ax4, ax5, ax6]
titles = ["volume", "F2 center (ppm)", "F1 center (ppm)", "F1 LW (pts)", "F2 LW (pts)"]

mod = LinearModel()
fit = mod.fit(merged.amp_y, x=merged.amp_x)
ax1.plot(
    merged.amp_x, merged.amp_y, "o", label=f"slope={fit.params['slope'].value:.2f}"
)
ax1.plot([2.5e8, 6.5e8], [2.5e8, 6.5e8], "k--", alpha=0.75)

mod = LinearModel()
fit = mod.fit(merged.X_PPM, x=merged.center_x_ppm)
ax2.plot(
    merged.center_x_ppm,
    merged.X_PPM,
    "o",
    label=f"slope={fit.params['slope'].value:.2f}",
)
ax2.plot([7, 10.5], [7, 10.5], "k--", alpha=0.75)

mod = LinearModel()
fit = mod.fit(merged.Y_PPM, x=merged.center_y_ppm)
ax3.plot(
    merged.center_y_ppm,
    merged.Y_PPM,
    "o",
    label=f"slope={fit.params['slope'].value:.2f}",
)
ax3.plot([106, 130], [106, 130], "k--", alpha=0.75)

mod = LinearModel()
fit = mod.fit(merged.YW, x=merged.fwhm_y)
ax4.plot(merged.fwhm_y, merged.YW, "o", label=f"slope={fit.params['slope'].value:.2f}")
ax4.plot([2.3, 2.7], [2.3, 2.7], "k--", alpha=0.75)

mod = LinearModel()
fit = mod.fit(merged.XW, x=merged.fwhm_x)
ax5.plot(merged.fwhm_x, merged.XW, "o", label=f"slope={fit.params['slope'].value:.2f}")
ax5.plot([2.2, 2.8], [2.2, 2.8], "k--", alpha=0.75)

ax6.text(
    0.4,
    0.5,
    "Fit using Gaussian lineshape",
    ha="center",
    va="center",
    transform=ax6.transAxes,
)

for title, ax in zip(titles, axes):
    ax.set_ylabel("NMRPipe")
    ax.set_xlabel("peakipy")
    ax.legend()
    ax.set_title(title)

ax6.axis("off")
plt.tight_layout()
plt.savefig("correlation.svg")
plt.savefig("correlation.pdf")
plt.savefig("correlation.png", dpi=300)
output_file("correlation.html", title="Comparison of PIPE and peakipy")
show(column(p1, p2, p3))  # show the plot
