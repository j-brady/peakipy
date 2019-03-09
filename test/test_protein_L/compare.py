import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import column

# read data
pipe_data = pd.read_csv("nlin.csv")
peakipy_data = pd.read_csv("fits.csv")
# take first plane
peakipy_data = peakipy_data[peakipy_data.plane == 0]
# make new columns called amp and ASS
pipe_data["amp"] = pipe_data["VOL"]
peakipy_data["ASS"] = peakipy_data["assignment"]

# merge dataframes on assignment column
merged = pd.merge(peakipy_data, pipe_data, on="ASS")
source = ColumnDataSource(merged)
p1 = figure(tooltips = [('Assignment', '@assignment, @ASS')], x_axis_label="peakipy", y_axis_label="pipe", title="VOL")
p1.circle("amp_x", "VOL", source=source, size=10, alpha=0.75)
p2 = figure(tooltips = [('Assignment', '@assignment, @ASS')], x_axis_label="peakipy", y_axis_label="pipe", title="X LW")
p2.circle("fwhm_x", "XW", source=source, size=10, alpha=0.75)
p3 = figure(tooltips = [('Assignment', '@assignment, @ASS')], x_axis_label="peakipy", y_axis_label="pipe", title="Y LW")
p3.circle("fwhm_y", "YW", source=source, size=10, alpha=0.75)

# make matplotlib figure
fig, axes = plt.subplots(2,3,figsize=(9,6))

ax1 = axes[0][0]
ax2 = axes[0][1]
ax3 = axes[0][2]
ax4 = axes[1][0]
ax5 = axes[1][1]
ax6 = axes[1][2]

axes = [ax1,ax2,ax3,ax4,ax5,ax6]
titles = ["VOL","X center","Y center","Y LW","X LW"]
ax1.plot(merged.amp_x,merged.amp_y,"o")
ax1.plot([2.5e8,6.5e8],[2.5e8,6.5e8],"k--",alpha=0.75)

ax2.plot(merged.center_x_ppm, merged.X_PPM,"o")
ax2.plot([7,10.5],[7,10.5],"k--",alpha=0.75)

ax3.plot(merged.center_y_ppm, merged.Y_PPM,"o")
ax3.plot([106,130],[106,130],"k--",alpha=0.75)

ax4.plot(merged.fwhm_y, merged.YW,"o")
ax4.plot([2.3,2.7],[2.3,2.7],"k--",alpha=0.75)

ax5.plot(merged.fwhm_x, merged.XW,"o")
ax5.plot([2.2,2.8],[2.2,2.8],"k--",alpha=0.75)
#ax6.plot(merged.fwhm_y, merged.YW,"o")

for title,ax in zip(titles,axes):
    ax.set_ylabel("Pipe")
    ax.set_xlabel("Peakipy")
    ax.set_title(title)

ax6.remove()
plt.tight_layout()
plt.savefig("correlation.pdf")
output_file("correlation.html", title="Comparison of PIPE and peakipy")
show(column(p1,p2,p3)) # show the plot
