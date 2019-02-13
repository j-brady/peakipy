#!/usr/bin/env python3
""" Script for checking fits and editing fit params

    Usage:
        check_fits.py <peaklist> <data> [options]

    Arguments:
        <peaklist>  peaklist output from read_peaklist.py (csv, tab or pkl)
        <data>      NMRPipe data

    Options:
        --dims=<id,f1,f2>  order of dimensions [default: 0,1,2]

"""
import os

from docopt import docopt
from pathlib import Path

import pandas as pd
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from bokeh.events import ButtonClick
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, Button, DataTable, TableColumn, TextInput
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.palettes import PuBuGn9


def fit_selected(event):

    selectionIndex=source.selected.indices
    current = df.iloc[selectionIndex]
    
    df.loc[selectionIndex,"X_RADIUS_PPM"] = slider_X_RADIUS.value
    df.loc[selectionIndex,"Y_RADIUS_PPM"] = slider_Y_RADIUS.value

    df.loc[selectionIndex,"X_DIAMETER_PPM"] = current["X_RADIUS_PPM"] * 2.0
    df.loc[selectionIndex,"Y_DIAMETER_PPM"] = current["Y_RADIUS_PPM"] * 2.0
    
    df.loc[selectionIndex,"Edited"] = True

    selected_df = df[df.CLUSTID.isin(list(current.CLUSTID))]

    selected_df.to_csv("~tmp.csv")
    os.system(f"fit_peaks.py ~tmp.csv {data_path} ~tmp_out.csv --plot=out --show")


def save_peaks(event):
    if savefilename.value:
        to_save = Path(savefilename.value)
    else: 
        to_save = Path(savefilename.placeholder)

    if to_save.exists():
        os.system(f"cp {to_save} {to_save}.bak")
        print("Making backup {to_save}.bak")
    
    print(f"Saving peaks to {to_save}")
    if to_save.suffix == '.csv':
        df.to_csv(to_save)
    else:
        df.to_pickle(to_save)


def select_callback(attrname, old, new):

    selectionIndex=source.selected.indices
    current = df.iloc[selectionIndex]

def callback(attrname, old, new):

    selectionIndex=source.selected.indices
    current = df.iloc[selectionIndex]
    
    df.loc[selectionIndex,"X_RADIUS"] = slider_X_RADIUS.value * pt_per_ppm_f2
    df.loc[selectionIndex,"Y_RADIUS"] = slider_Y_RADIUS.value * pt_per_ppm_f1
    df.loc[selectionIndex,"X_RADIUS_PPM"] = slider_X_RADIUS.value
    df.loc[selectionIndex,"Y_RADIUS_PPM"] = slider_Y_RADIUS.value

    df.loc[selectionIndex,"X_DIAMETER_PPM"] = current["X_RADIUS_PPM"] * 2.0
    df.loc[selectionIndex,"Y_DIAMETER_PPM"] = current["Y_RADIUS_PPM"] * 2.0
    df.loc[selectionIndex,"X_DIAMETER"] = current["X_RADIUS"] * 2.0
    df.loc[selectionIndex,"Y_DIAMETER"] = current["Y_RADIUS"] * 2.0
    
    # set edited rows to True
    df.loc[selectionIndex,"Edited"] = True

    selected_df = df[df.CLUSTID.isin(list(current.CLUSTID))]
    #print(list(selected_df))
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
        thecol = '#%02x%02x%02x' % (thecol[0], thecol[1], thecol[2])

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

    source = ColumnDataSource(data={'xs': xs, 'ys': ys, 'line_color': col,'xt':xt,'yt':yt,'text':text})
    return source


def update_contour(attrname, old, new):
    new_cs = eval(contour_start.value)
    cl = new_cs * contour_factor ** np.arange(contour_num)
    spec_source.data = get_contour_data(data[0], cl, extent=extent).data

args = docopt(__doc__)
path = Path(args.get('<peaklist>'))

if path.suffix == '.csv':
    df = pd.read_csv(path)
elif path.suffix == '.tab':
    df = pd.read_csv(path,sep='\t')
else:
    df = pd.read_pickle(path)

# make diameter columns
if "X_DIAMETER_PPM" in df.columns:
    pass
else:
    df["X_DIAMETER_PPM"] = df["X_RADIUS_PPM"] * 2.0
    df["Y_DIAMETER_PPM"] = df["Y_RADIUS_PPM"] * 2.0

# make a column to track edited peaks
if "Edited" in df.columns:
    pass
else:
    df["Edited"] = np.zeros(len(df),dtype=bool)
    df["color"] = df.Edited.apply(lambda x: 'red' if x else 'black')

# make datasource
source = ColumnDataSource(data=dict())
source.data = {col: df[col] for col in df.columns}

# read pipe data
data_path = args.get('<data>')
dic, data = ng.pipe.read(data_path)

udic = ng.pipe.guess_udic(dic, data)
ndim = udic["ndim"]
# get dim numbers
dims = args.get('--dims')
dims = [int(i) for i in dims.split(',')]
planes, f1, f2 = dims
# size of f1 and f2 in points
f2pts = udic[f2]["size"]
f1pts = udic[f1]["size"]
# points per ppm
pt_per_ppm_f1 = f1pts / (udic[f1]['sw'] / udic[f1]['obs'])
pt_per_ppm_f2 = f2pts / (udic[f2]['sw'] / udic[f2]['obs'])

# get ppm limits for ppm scales
uc_f1 = ng.pipe.make_uc(dic, data, dim=f1)
ppm_f1 = uc_f1.ppm_scale()
ppm_f1_0, ppm_f1_1 = uc_f1.ppm_limits()

uc_f2 = ng.pipe.make_uc(dic, data, dim=f2)
ppm_f2 = uc_f2.ppm_scale()
ppm_f2_0, ppm_f2_1 = uc_f2.ppm_limits()

# make bokeh figure
p = figure(
    x_range=(ppm_f2_0, ppm_f2_1),
    y_range=(ppm_f1_0, ppm_f1_1),
    tooltips=[
        ("Assignment", "@ASS"),
        ("CLUSTID", "@CLUSTID"),
        ("RADII", "@X_RADIUS_PPM, @Y_RADIUS_PPM"),
    ],
)

# rearrange dims
if dims != [0, 1, 2]:
    data = np.transpose(data, dims)

# plot NMR data
#p.image(image=[data[0]],
#        x=ppm_f2_0, y=ppm_f1_0, dw=(ppm_f2_0-ppm_f2_1), dh=(ppm_f1_0-ppm_f1_1),
#        palette=PuBuGn9[::-1])

thres = threshold_otsu(data[0])
contour_start = thres          # contour level start value
contour_num = 20               # number of contour levels
contour_factor = 1.20          # scaling factor between contour levels
cl = contour_start * contour_factor ** np.arange(contour_num)
extent=(ppm_f2_0,ppm_f2_1,ppm_f1_0,ppm_f1_1)
spec_source = get_contour_data(data[0],cl,extent=extent)
p.multi_line(xs='xs', ys='ys', line_color='line_color', source=spec_source)
#contour_num = Slider(title="contour number", value=20, start=1, end=50,step=1)
#contour_start = Slider(title="contour start", value=100000, start=1000, end=10000000,step=1000)
contour_start = TextInput(value="%.2e"%thres, title="Contour level:")
#contour_factor = Slider(title="contour factor", value=1.20, start=1., end=2.,step=0.05)
contour_start.on_change("value", update_contour)
#for w in [contour_num,contour_start,contour_factor]:
#    w.on_change("value",update_contour)

# plot mask outlines
p.ellipse(
    x="X_PPM",
    y="Y_PPM",
    width="X_DIAMETER_PPM",
    height="Y_DIAMETER_PPM",
    source=source,
    fill_color="black",
    fill_alpha=0.1,
    line_dash="dotted",
    line_color="red",
)

p.circle(
        x="X_PPM",
        y="Y_PPM",
        source=source,
        color="color",
)
# plot cluster numbers
p.text(x="X_PPM", y="Y_PPM", text="CLUSTID", source=source)

# configure sliders
slider_X_RADIUS = Slider(
    title="X_RADIUS - ppm", start=0.001, end=0.2, value=0.04, step=0.001
)
slider_Y_RADIUS = Slider(
    title="Y_RADIUS - ppm", start=0.01, end=2.0, value=0.4, step=0.001
)

slider_X_RADIUS.on_change('value', lambda attr, old, new: callback(attr, old, new))
slider_Y_RADIUS.on_change('value', lambda attr, old, new: callback(attr, old, new))

# save file
savefilename = TextInput(title="Save file as (.csv or .pkl)", placeholder='edited_peaks.csv')
button = Button(label="Save", button_type="success")
button.on_event(ButtonClick, save_peaks)
# call fit_peaks.py
fit_button = Button(label="Fit selected", button_type="primary")
# not sure this is needed
selected_df = df.copy()

fit_button.on_event(ButtonClick, fit_selected)

selected_columns = [
    "ASS",
    "CLUSTID",
    "X_PPM",
    "Y_PPM",
    "X_RADIUS_PPM",
    "Y_RADIUS_PPM",
    "XW_HZ",
    "YW_HZ",
    "VOL",
    "Edited",
]

columns = [
    TableColumn(field=field, title=field)
    for field in selected_columns
]

data_table = DataTable(
    source=source, columns=columns, editable=True, fit_columns=True, width=600
)

# callback for adding 
#source.selected.on_change('indices', callback)
source.selected.on_change('indices', select_callback)

# controls = column(slider, button)
controls = column(row(slider_X_RADIUS, slider_Y_RADIUS), row(column(contour_start, fit_button), column(savefilename,button)))

curdoc().add_root(row(p, column(data_table, controls)))
# curdoc().title = "Export CSV"


#update()
