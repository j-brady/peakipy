#!/usr/bin/env python3
""" Script for checking fits and editing fit params

    Usage:
        edit_fits_script.py <peaklist> <data> [options]

    Arguments:
        <peaklist>  peaklist output from read_peaklist.py (csv, tab or pkl)
        <data>      NMRPipe data

    Options:
        --dims=<id,f1,f2>  order of dimensions [default: 0,1,2]


    peakipy - deconvolute overlapping NMR peaks
    Copyright (C) 2019  Jacob Peter Brady

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
import os
import json

from docopt import docopt
from pathlib import Path

import pandas as pd
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import magma, autumn

from scipy import ndimage

# from scipy.ndimage.morphology import binary_dilation
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, square, rectangle, disk

from bokeh.events import ButtonClick, DoubleTap
from bokeh.layouts import row, column, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import (
    Slider,
    Select,
    Button,
    DataTable,
    TableColumn,
    NumberFormatter,
    NumberEditor,
    IntEditor,
    StringEditor,
    StringFormatter,
    SelectEditor,
    TextInput,
    RadioButtonGroup,
    Div,
)
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.palettes import PuBuGn9, Category20

from peakipy.core import Pseudo3D


def clusters(
    df, data, thres=None, struc_el="square", struc_size=(3,), iterations=1, l_struc=None
):
    """ Find clusters of peaks

    Need to update these docs.

    thres : float
        threshold for signals above which clusters are selected
    ndil: int
        number of iterations of ndimage.binary_dilation function if set to 0 then function not used
    """

    peaks = [[y, x] for y, x in zip(df.Y_AXIS, df.X_AXIS)]

    if thres == None:
        thresh = threshold_otsu(data[0])
    else:
        thresh = thres

    thresh_data = np.bitwise_or(data[0] < (thresh * -1.0), data[0] > thresh)

    if struc_el == "disk":
        radius = struc_size[0]
        print(f"using disk with {radius}")
        closed_data = binary_closing(thresh_data, disk(int(radius)))
        # closed_data = binary_dilation(thresh_data, disk(radius), iterations=iterations)

    elif struc_el == "square":
        width = struc_size[0]
        print(f"using square with {width}")
        closed_data = binary_closing(thresh_data, square(int(width)))
        # closed_data = binary_dilation(thresh_data, square(width), iterations=iterations)

    elif struc_el == "rectangle":
        width, height = struc_size
        print(f"using rectangle with {width} and {height}")
        closed_data = binary_closing(thresh_data, rectangle(int(width), int(height)))
        # closed_data = binary_dilation(thresh_data, rectangle(width, height), iterations=iterations)

    else:
        closed_data = thresh_data
        print(f"Not using any closing function")

    labeled_array, num_features = ndimage.label(closed_data, l_struc)
    # print(labeled_array, num_features)

    df["CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

    #  renumber "0" clusters
    max_clustid = df["CLUSTID"].max()
    n_of_zeros = len(df[df["CLUSTID"] == 0]["CLUSTID"])
    df.loc[df[df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
        max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
    )

    for ind, group in df.groupby("CLUSTID"):
        df.loc[group.index, "MEMCNT"] = len(group)

    df["color"] = df.apply(
        lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black",
        axis=1,
    )

    source.data = {col: df[col] for col in df.columns}

    return df


def recluster_peaks(event):

    struc_size = tuple([int(i) for i in struct_el_size.value.split(",")])

    print(struc_size)
    clusters(
        df,
        data,
        thres=eval(contour_start.value),
        struc_el=struct_el.value,
        struc_size=struc_size,
        # iterations=int(iterations.value)
    )
    # print("struct", struct_el.value)
    # print("struct size", struct_el_size.value )
    # print(type(struct_el_size.value) )
    # print(type(eval(struct_el_size.value)) )
    # print(type([].extend(eval(struct_el_size.value)))


def update_memcnt(df):

    for ind, group in df.groupby("CLUSTID"):
        df.loc[group.index, "MEMCNT"] = len(group)

    # set cluster colors (set to black if singlet peaks)
    df["color"] = df.apply(
        lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black",
        axis=1,
    )
    # update source data
    source.data = {col: df[col] for col in df.columns}
    return df


def fit_selected(event):

    selectionIndex = source.selected.indices
    current = df.iloc[selectionIndex]

    df.loc[selectionIndex, "X_RADIUS_PPM"] = slider_X_RADIUS.value
    df.loc[selectionIndex, "Y_RADIUS_PPM"] = slider_Y_RADIUS.value

    df.loc[selectionIndex, "X_DIAMETER_PPM"] = current["X_RADIUS_PPM"] * 2.0
    df.loc[selectionIndex, "Y_DIAMETER_PPM"] = current["Y_RADIUS_PPM"] * 2.0

    selected_df = df[df.CLUSTID.isin(list(current.CLUSTID))]

    selected_df.to_csv("~tmp.csv")

    lineshape = lineshapes[radio_button_group.active]
    print("Using LS = ", lineshape)
    print(
        f"fit_peaks ~tmp.csv {data_path} ~tmp_out.csv --plot=out --show --lineshape={lineshape} --dims={_dims}"
    )
    os.system(
        f"fit_peaks ~tmp.csv {data_path} ~tmp_out.csv --plot=out --show --lineshape={lineshape} --dims={_dims}"
    )


def save_peaks(event):
    if savefilename.value:
        to_save = Path(savefilename.value)
    else:
        to_save = Path(savefilename.placeholder)

    if to_save.exists():
        os.system(f"cp {to_save} {to_save}.bak")
        print(f"Making backup {to_save}.bak")

    print(f"Saving peaks to {to_save}")
    if to_save.suffix == ".csv":
        df.to_csv(to_save, float_format="%.4f", index=False)
    else:
        df.to_pickle(to_save)


def select_callback(attrname, old, new):
    # print("Calling Select Callback")
    selectionIndex = source.selected.indices
    current = df.iloc[selectionIndex]

    # update memcnt
    update_memcnt(df)


def peak_pick_callback(event):
    print(event.x, event.y)


def slider_callback(attrname, old, new):

    selectionIndex = source.selected.indices
    current = df.iloc[selectionIndex]

    df.loc[selectionIndex, "X_RADIUS"] = slider_X_RADIUS.value * pt_per_ppm_f2
    df.loc[selectionIndex, "Y_RADIUS"] = slider_Y_RADIUS.value * pt_per_ppm_f1
    df.loc[selectionIndex, "X_RADIUS_PPM"] = slider_X_RADIUS.value
    df.loc[selectionIndex, "Y_RADIUS_PPM"] = slider_Y_RADIUS.value

    df.loc[selectionIndex, "X_DIAMETER_PPM"] = current["X_RADIUS_PPM"] * 2.0
    df.loc[selectionIndex, "Y_DIAMETER_PPM"] = current["Y_RADIUS_PPM"] * 2.0
    df.loc[selectionIndex, "X_DIAMETER"] = current["X_RADIUS"] * 2.0
    df.loc[selectionIndex, "Y_DIAMETER"] = current["Y_RADIUS"] * 2.0

    # set edited rows to True
    df.loc[selectionIndex, "Edited"] = True

    # selected_df = df[df.CLUSTID.isin(list(current.CLUSTID))]
    # print(list(selected_df))
    source.data = {col: df[col] for col in df.columns}


def get_contour_data(data, levels, **kwargs):
    cs = plt.contour(data, levels, **kwargs)
    xs = []
    ys = []
    xt = []
    yt = []
    col = []
    text = []
    isolevelid = 0
    for isolevel in cs.collections:
        isocol = isolevel.get_color()[0]
        thecol = 3 * [None]
        theiso = str(cs.get_array()[isolevelid])
        isolevelid += 1
        for i in range(3):
            thecol[i] = int(255 * isocol[i])
        thecol = "#%02x%02x%02x" % (thecol[0], thecol[1], thecol[2])

        for path in isolevel.get_paths():
            v = path.vertices
            x = v[:, 0]
            y = v[:, 1]
            xs.append(x.tolist())
            ys.append(y.tolist())
            indx = int(len(x) / 2)
            indy = int(len(y) / 2)
            xt.append(x[indx])
            yt.append(y[indy])
            text.append(theiso)
            col.append(thecol)

    source = ColumnDataSource(
        data={"xs": xs, "ys": ys, "line_color": col, "xt": xt, "yt": yt, "text": text}
    )
    return source


def update_contour(attrname, old, new):
    new_cs = eval(contour_start.value)
    cl = new_cs * contour_factor ** np.arange(contour_num)
    spec_source.data = get_contour_data(data[0], cl, extent=extent).data


def exit_edit_peaks(event):
    exit()


#  Script starts here

args = docopt(__doc__)
path = Path(args.get("<peaklist>"))

if path.suffix == ".csv":
    df = pd.read_csv(path)  # , comment="#")
elif path.suffix == ".tab":
    df = pd.read_csv(path, sep="\t")  # comment="#")
else:
    df = pd.read_pickle(path)

# make diameter columns
if "X_DIAMETER_PPM" in df.columns:
    pass
else:
    df["X_DIAMETER_PPM"] = df["X_RADIUS_PPM"] * 2.0
    df["Y_DIAMETER_PPM"] = df["Y_RADIUS_PPM"] * 2.0

#  make a column to track edited peaks
if "Edited" in df.columns:
    pass
else:
    df["Edited"] = np.zeros(len(df), dtype=bool)

if "include" in df.columns:
    pass
else:
    df["include"] = df.apply(lambda _: "yes", axis=1)
#    df["color"] = df.Edited.apply(lambda x: 'red' if x else 'black')

# color clusters
df["color"] = df.apply(
    lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black", axis=1
)

# make datasource
source = ColumnDataSource(data=dict())
source.data = {col: df[col] for col in df.columns}


#  read dims from config
config_path = Path("peakipy.config")
if config_path.exists():
    config = json.load(open(config_path))
    print(f"Using config file with --dims={config.get('--dims')}")
    dims = config.get("--dims", [0, 1, 2])
    _dims = ",".join(str(i) for i in dims)

else:
    # get dim numbers from commandline
    _dims = args.get("--dims")
    dims = [int(i) for i in _dims.split(",")]


# read pipe data
data_path = args.get("<data>")
dic, data = ng.pipe.read(data_path)
pseudo3D = Pseudo3D(dic, data, dims)
data = pseudo3D.data
udic = pseudo3D.udic

dims = pseudo3D.dims
planes, f1, f2 = dims
# size of f1 and f2 in points
f2pts = pseudo3D.f2_size
f1pts = pseudo3D.f1_size

#  points per ppm
pt_per_ppm_f1 = pseudo3D.pt_per_ppm_f1
pt_per_ppm_f2 = pseudo3D.pt_per_ppm_f2

# get ppm limits for ppm scales
uc_f1 = pseudo3D.uc_f1
ppm_f1 = uc_f1.ppm_scale()
ppm_f1_0, ppm_f1_1 = uc_f1.ppm_limits()

uc_f2 = pseudo3D.uc_f2
ppm_f2 = uc_f2.ppm_scale()
ppm_f2_0, ppm_f2_1 = uc_f2.ppm_limits()

f2_label = pseudo3D.f2_label
f1_label = pseudo3D.f1_label
#  make bokeh figure
tools = [
    "redo",
    "undo",
    "tap",
    "box_zoom",
    "lasso_select",
    "box_select",
    "wheel_zoom",
    "pan",
    "reset",
]
p = figure(
    x_range=(ppm_f2_0, ppm_f2_1),
    y_range=(ppm_f1_0, ppm_f1_1),
    x_axis_label=f"{f2_label} - ppm",
    y_axis_label=f"{f1_label} - ppm",
    title="Check fits",
    tools=tools,
    active_drag="pan",
    active_scroll="wheel_zoom",
    active_tap=None,
)

thres = threshold_otsu(data[0])
contour_start = thres  # contour level start value
contour_num = 20  # number of contour levels
contour_factor = 1.20  # scaling factor between contour levels
cl = contour_start * contour_factor ** np.arange(contour_num)
extent = (ppm_f2_0, ppm_f2_1, ppm_f1_0, ppm_f1_1)
spec_source = get_contour_data(data[0], cl, extent=extent)
#  negative contours
spec_source_neg = get_contour_data(data[0] * -1.0, cl, extent=extent, cmap=autumn)
p.multi_line(xs="xs", ys="ys", line_color="line_color", source=spec_source)
p.multi_line(xs="xs", ys="ys", line_color="line_color", source=spec_source_neg)
# contour_num = Slider(title="contour number", value=20, start=1, end=50,step=1)
# contour_start = Slider(title="contour start", value=100000, start=1000, end=10000000,step=1000)
contour_start = TextInput(value="%.2e" % thres, title="Contour level:")
# contour_factor = Slider(title="contour factor", value=1.20, start=1., end=2.,step=0.05)
contour_start.on_change("value", update_contour)
# for w in [contour_num,contour_start,contour_factor]:
#    w.on_change("value",update_contour)

#  plot mask outlines
el = p.ellipse(
    x="X_PPM",
    y="Y_PPM",
    width="X_DIAMETER_PPM",
    height="Y_DIAMETER_PPM",
    source=source,
    fill_color="color",
    fill_alpha=0.1,
    line_dash="dotted",
    line_color="red",
)

p.add_tools(
    HoverTool(
        tooltips=[
            ("Index", "$index"),
            ("Assignment", "@ASS"),
            ("CLUSTID", "@CLUSTID"),
            ("RADII", "@X_RADIUS_PPM{0.000}, @Y_RADIUS_PPM{0.000}"),
            (f"{f2_label},{f1_label}", "$x{0.000} ppm, $y{0.000} ppm"),
        ],
        mode="mouse",
        # add renderers
        renderers=[el],
    )
)
# p.toolbar.active_scroll = "auto"

p.circle(x="X_PPM", y="Y_PPM", source=source, color="color")
# plot cluster numbers
p.text(
    x="X_PPM",
    y="Y_PPM",
    text="CLUSTID",
    text_color="color",
    source=source,
    text_font_size="8pt",
    text_font_style="bold",
)

p.on_event(DoubleTap, peak_pick_callback)

# configure sliders
slider_X_RADIUS = Slider(
    title="X_RADIUS - ppm",
    start=0.001,
    end=0.200,
    value=0.040,
    step=0.001,
    format="0[.]000",
)
slider_Y_RADIUS = Slider(
    title="Y_RADIUS - ppm",
    start=0.010,
    end=2.000,
    value=0.400,
    step=0.001,
    format="0[.]000",
)

slider_X_RADIUS.on_change(
    "value", lambda attr, old, new: slider_callback(attr, old, new)
)
slider_Y_RADIUS.on_change(
    "value", lambda attr, old, new: slider_callback(attr, old, new)
)

# save file
savefilename = TextInput(
    title="Save file as (.csv or .pkl)", placeholder="edited_peaks.csv"
)
button = Button(label="Save", button_type="success")
button.on_event(ButtonClick, save_peaks)
# call fit_peaks
fit_button = Button(label="Fit selected cluster", button_type="primary")
radio_button_group = RadioButtonGroup(
    labels=["PV", "G", "L", "PV_L", "PV_G", "PV_PV", "G_L"], active=0
)
lineshapes = {0: "PV", 1: "G", 2: "L", 3: "PV_L", 4: "PV_G", 5: "PV_PV", 6: "G_L"}
ls_div = Div(
    text="Choose lineshape you wish to fit. This can be Pseudo-voigt (PV), Gaussian (G), Lorentzian (L), PV/G, PV/L, PV_PV, G/L. PV/G fits a PV lineshape to the direct dimension and a G lineshape to the indirect."
)
clust_div = Div(
    text="""If you want to adjust how the peaks are automatically clustered then try changing the
        width/diameter/height (integer values) of the structuring element used during the binary dilation step
        (you can also remove it by selecting 'None'). Increasing the size of the structuring element will cause
        peaks to be more readily incorporated into clusters."""
)

#  not sure this is needed
selected_df = df.copy()

fit_button.on_event(ButtonClick, fit_selected)

# selected_columns = [
#    "ASS",
#    "CLUSTID",
#    "X_PPM",
#    "Y_PPM",
#    "X_RADIUS_PPM",
#    "Y_RADIUS_PPM",
#    "XW_HZ",
#    "YW_HZ",
#    "VOL",
#    "include",
#    "MEMCNT",
# ]
#
# columns = [TableColumn(field=field, title=field) for field in selected_columns]
columns = [
    TableColumn(field="ASS", title="Assignment"),
    TableColumn(field="CLUSTID", title="Cluster", editor=IntEditor()),
    TableColumn(
        field="X_PPM",
        title=f"{f2_label}",
        editor=NumberEditor(step=0.0001),
        formatter=NumberFormatter(format="0.0000"),
    ),
    TableColumn(
        field="Y_PPM",
        title=f"{f1_label}",
        editor=NumberEditor(step=0.0001),
        formatter=NumberFormatter(format="0.0000"),
    ),
    TableColumn(
        field="X_RADIUS_PPM",
        title=f"{f2_label} radius (ppm)",
        editor=NumberEditor(step=0.0001),
        formatter=NumberFormatter(format="0.0000"),
    ),
    TableColumn(
        field="Y_RADIUS_PPM",
        title=f"{f1_label} radius (ppm)",
        editor=NumberEditor(step=0.0001),
        formatter=NumberFormatter(format="0.0000"),
    ),
    TableColumn(
        field="XW_HZ",
        title=f"{f2_label} LW (Hz)",
        editor=NumberEditor(step=0.01),
        formatter=NumberFormatter(format="0.00"),
    ),
    TableColumn(
        field="YW_HZ",
        title=f"{f1_label} LW (Hz)",
        editor=NumberEditor(step=0.01),
        formatter=NumberFormatter(format="0.00"),
    ),
    TableColumn(field="VOL", title="Volume", formatter=NumberFormatter(format="0.0")),
    TableColumn(
        field="include", title="Include", editor=SelectEditor(options=["yes", "no"])
    ),
    TableColumn(field="MEMCNT", title="MEMCNT", editor=IntEditor()),
]

data_table = DataTable(
    source=source, columns=columns, editable=True, fit_columns=True, width=800
)

# callback for adding
# source.selected.on_change('indices', callback)
source.selected.on_change("indices", select_callback)

# controls = column(slider, button)
exit_button = Button(label="Quit", button_type="warning")
exit_button.on_event(ButtonClick, exit_edit_peaks)

controls = column(
    row(slider_X_RADIUS, slider_Y_RADIUS),
    row(
        column(contour_start, fit_button, widgetbox(ls_div), radio_button_group),
        column(savefilename, button, exit_button),
    ),
)

# widgetbox(radio_button_group)
struct_el = Select(
    title="Structuring element:",
    value="disk",
    options=["square", "disk", "rectangle", "None"],
)

struct_el_size = TextInput(
    value="3", title="Size(width/radius or width,height for rectangle):"
)
# iterations = TextInput(value="1", title="Number of iterations of binary dilation")
recluster = Button(label="Re-cluster", button_type="warning")
recluster.on_event(ButtonClick, recluster_peaks)

# cluster_widget = widgetbox(struct_el, struct_el_size)
# recluster)
curdoc().add_root(
    row(
        column(p, widgetbox(clust_div), row(struct_el, struct_el_size), recluster),
        column(data_table, controls),
    )
)
curdoc().title = "peakipy: Edit Fits"
# curdoc().theme = 'dark_minimal'
# update()
