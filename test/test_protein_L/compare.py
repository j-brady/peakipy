import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import column

#from lmfit.models import LinearModel

cols = "INDEX X_AXIS Y_AXIS DX DY X_PPM Y_PPM X_HZ Y_HZ XW YW XW_HZ YW_HZ X1 X3 Y1 Y3 HEIGHT DHEIGHT VOL PCHI2 TYPE ASS CLUSTID MEMCNT".split(" ")
cols
pipe_data = pd.read_csv("nlin.tab",skiprows=13,delim_whitespace=True,names=cols)
pipe_data["ASS"] = ["_%d"%i for i in range(len(pipe_data))]

peakipy_data = pd.read_csv("_fits.csv")
# take first plane
peakipy_data = peakipy_data[peakipy_data.plane == 0]
pipe_data["amp"] = pipe_data["VOL"]
peakipy_data["ASS"] = peakipy_data["assignment"]

merged = pd.merge(peakipy_data, pipe_data, on="ASS")
source = ColumnDataSource(merged)
p1 = figure(tooltips = [('Assignment', '@assignment, @ASS')], x_axis_label="peakipy", y_axis_label="pipe", title="VOL")
p1.circle("amp_x", "VOL", source=source, size=10, alpha=0.75)
p2 = figure(tooltips = [('Assignment', '@assignment, @ASS')], x_axis_label="peakipy", y_axis_label="pipe", title="X LW")
p2.circle("fwhm_x", "XW", source=source, size=10, alpha=0.75)
p3 = figure(tooltips = [('Assignment', '@assignment, @ASS')], x_axis_label="peakipy", y_axis_label="pipe", title="Y LW")
p3.circle("fwhm_y", "YW", source=source, size=10, alpha=0.75)
#print(merged)
#print(pipe_data.amp)
#print(peakipy_data.amp)
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
ax3.plot(merged.center_y_ppm, merged.Y_PPM,"o")
ax4.plot(merged.fwhm_y, merged.YW,"o")
ax5.plot(merged.fwhm_x, merged.XW,"o")
#ax6.plot(merged.fwhm_y, merged.YW,"o")

for title,ax in zip(titles,axes):
    ax.set_ylabel("Pipe")
    ax.set_xlabel("Peakipy")
    ax.set_title(title)
ax6.remove()
plt.tight_layout()
plt.savefig("correlation.pdf")
#plt.show()
output_file("correlation.html", title="Comparison of PIPE and peakipy")
show(column(p1,p2,p3)) # show the plot

#print(peakipy_data)
#print(pd.merge(peakipy_data,pipe_data,"amp"))

