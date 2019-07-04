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
import sys
import shutil
import json

from pathlib import Path
from subprocess import check_output
from docopt import docopt
from schema import Schema, And, SchemaError

import pandas as pd
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import magma, autumn

from scipy import ndimage

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
    SelectEditor,
    TextInput,
    RadioButtonGroup,
    CheckboxGroup,
    Div,
    Tabs,
    Panel,
)
from bokeh.plotting import figure
# from bokeh.io import curdoc
from bokeh.server.server import Server
from bokeh.palettes import PuBuGn9, Category20

from peakipy.core import Pseudo3D, run_log


def bokeh_script(doc):
    # Temp files
    TEMP_PATH = Path("tmp")
    TEMP_PATH.mkdir(parents=True, exist_ok=True)

    TEMP_OUT_CSV = TEMP_PATH / Path("tmp_out.csv")
    TEMP_INPUT_CSV = TEMP_PATH / Path("tmp.csv")

    TEMP_OUT_PLOT = TEMP_PATH / Path("plots")
    TEMP_OUT_PLOT.mkdir(parents=True, exist_ok=True)


    def clusters(df, data, thres=None, struc_el="square", struc_size=(3,)):
        """ Find clusters of peaks

        :param thres: threshold for signals above which clusters are selected
        :type thres : float

        :param df: DataFrame containing peak list
        :type df: pandas.DataFrame

        :param data: NMR data
        :type data: numpy.array

        :param struc_el:
        :type struc_el:

        :param struc_size:
        :type struc_size:

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

        labeled_array, num_features = ndimage.label(closed_data)
        # print(labeled_array, num_features)

        df.loc[:, "CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

        #  renumber "0" clusters
        max_clustid = df["CLUSTID"].max()
        n_of_zeros = len(df[df["CLUSTID"] == 0]["CLUSTID"])
        df.loc[df[df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
            max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
        )

        for ind, group in df.groupby("CLUSTID"):
            df.loc[group.index, "MEMCNT"] = len(group)

        df.loc[:, "color"] = df.apply(
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
        # change color of excluded peaks
        include_no = df.include == "no"
        df.loc[include_no, "color"] = "ghostwhite"
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

        selected_df.to_csv(TEMP_INPUT_CSV)

        lineshape = lineshapes[radio_button_group.active]
        if checkbox_group.active == []:
            print("Using LS = ", lineshape)
            fit_command = f"peakipy fit {TEMP_INPUT_CSV} {data_path} {TEMP_OUT_CSV} --plot={TEMP_OUT_PLOT} --show --lineshape={lineshape} --dims={_dims} --nomp"
        else:
            plane_index = select_plane.value
            print("Using LS = ", lineshape)
            fit_command = f"peakipy fit {TEMP_INPUT_CSV} {data_path} {TEMP_OUT_CSV} --plot={TEMP_OUT_PLOT} --show --lineshape={lineshape} --dims={_dims} --plane={plane_index} --nomp"

        print(fit_command)
        fit_reports.text += fit_command + "<br>"

        stdout = check_output(fit_command.split(" "))
        fit_reports.text += stdout.decode() + "<br><hr><br>"
        fit_reports.text = fit_reports.text.replace("\n", "<br>")


    def save_peaks(event):
        if savefilename.value:
            to_save = Path(savefilename.value)
        else:
            to_save = Path(savefilename.placeholder)

        if to_save.exists():
            shutil.copy(f"{to_save}", f"{to_save}.bak")
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
        # global so that df is updated globally
        global df
        x_radius_ppm = 0.035
        y_radius_ppm = 0.35
        x_radius = x_radius_ppm * pt_per_ppm_f2
        y_radius = y_radius_ppm * pt_per_ppm_f1
        x_diameter_ppm = x_radius_ppm * 2.0
        y_diameter_ppm = y_radius_ppm * 2.0
        clustid = df.CLUSTID.max() + 1
        index = df.INDEX.max() + 1
        x_ppm = event.x
        y_ppm = event.y
        x_axis = uc_f2.f(x_ppm, "ppm")
        y_axis = uc_f1.f(y_ppm, "ppm")
        xw_hz = 20.0
        yw_hz = 20.0
        xw = xw_hz * pt_per_hz_f2
        yw = yw_hz * pt_per_hz_f1
        assignment = f"test_peak_{index}_{clustid}"
        height = data[0][int(y_axis), int(x_axis)]
        volume = height
        print(f"""Adding peak at {assignment}: {event.x:.3f},{event.y:.3f}""")

        new_peak = {
            "INDEX": index,
            "X_PPM": x_ppm,
            "Y_PPM": y_ppm,
            "HEIGHT": height,
            "VOL": volume,
            "XW_HZ": xw_hz,
            "YW_HZ": yw_hz,
            "X_AXIS": int(np.floor(x_axis)),  # integers
            "Y_AXIS": int(np.floor(y_axis)),  # integers
            "X_AXISf": x_axis,
            "Y_AXISf": y_axis,
            "XW": xw,
            "YW": yw,
            "ASS": assignment,
            "X_RADIUS_PPM": x_radius_ppm,
            "Y_RADIUS_PPM": y_radius_ppm,
            "X_RADIUS": x_radius,
            "Y_RADIUS": y_radius,
            "CLUSTID": clustid,
            "MEMCNT": 1,
            "X_DIAMETER_PPM": x_diameter_ppm,
            "Y_DIAMETER_PPM": y_diameter_ppm,
            "Edited": True,
            "include": "yes",
            "color": "black",
        }
        df = df.append(new_peak, ignore_index=True)
        update_memcnt(df)


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
        plane_index = select_planes_dic[select_plane.value]
        spec_source.data = get_contour_data(data[plane_index], cl, extent=extent).data
        # print("Value of checkbox",checkbox_group.active)


    def exit_edit_peaks(event):
        exit()


    #  Script starts here
    args = docopt(__doc__, argv=argv)
    args = check_input(args)
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

    # color clusters
    df["color"] = df.apply(
        lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black", axis=1
    )

    # get rid of unnamed columns
    unnamed_cols = [i for i in df.columns if "Unnamed:" in i]
    df = df.drop(columns=unnamed_cols)

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
    #  points per hz
    pt_per_hz_f1 = pseudo3D.pt_per_hz_f1
    pt_per_hz_f2 = pseudo3D.pt_per_hz_f2

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
        # "redo",
        # "undo",
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
    contour_start = TextInput(value="%.2e" % thres, title="Contour level:", width=100)
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
    savefilename = TextInput(title="Save file as (.csv)", placeholder="edited_peaks.csv")
    button = Button(label="Save", button_type="success")
    button.on_event(ButtonClick, save_peaks)

    # call fit_peaks
    fit_button = Button(label="Fit selected cluster", button_type="primary")
    # lineshape selection
    lineshapes = {0: "PV", 1: "G", 2: "L", 3: "PV_PV", 4: "PV_L", 5: "PV_G", 6: "G_L"}
    radio_button_group = RadioButtonGroup(
        labels=[lineshapes[i] for i in lineshapes.keys()], active=0
    )
    ls_div = Div(
        text="""Choose lineshape you wish to fit. This can be Pseudo-voigt (PV), Gaussian (G), Lorentzian (L),
        PV/G, PV/L, PV_PV, G/L. PV/G fits a PV lineshape to the direct dimension and a G lineshape to the indirect."""
    )
    clust_div = Div(
        text="""If you want to adjust how the peaks are automatically clustered then try changing the
            width/diameter/height (integer values) of the structuring element used during the binary dilation step
            (you can also remove it by selecting 'None'). Increasing the size of the structuring element will cause
            peaks to be more readily incorporated into clusters. Be sure to save your peak list before doing this as
            any manual edits will be lost."""
    )
    intro_div = Div(
        text="""<h2>peakipy - interactive fit adjustment </h2> 
        """
    )

    doc_link = Div(
        text="<h3><a href='https://j-brady.github.io/peakipy/build/usage/instructions.html', target='_blank'> ℹ️ click here for documentation</a></h3>"
    )

    fit_reports = Div(
        text="", height=400, sizing_mode="scale_width", style={"overflow-y": "scroll"}
    )
    # Plane selection
    select_planes_list = [f"{i+1}" for i in range(data.shape[planes])]
    select_plane = Select(
        title="Select plane:", value=select_planes_list[0], options=select_planes_list
    )
    select_planes_dic = {f"{i+1}": i for i in range(data.shape[planes])}
    select_plane.on_change("value", update_contour)

    checkbox_group = CheckboxGroup(labels=["fit current plane only"], active=[])

    #  not sure this is needed
    selected_df = df.copy()

    fit_button.on_event(ButtonClick, fit_selected)

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

    data_table = DataTable(source=source, columns=columns, editable=True, fit_columns=True)

    # callback for adding
    # source.selected.on_change('indices', callback)
    source.selected.on_change("indices", select_callback)

    # Quit button
    exit_button = Button(label="Quit", button_type="warning")
    exit_button.on_event(ButtonClick, exit_edit_peaks)

    # Document layout
    fitting_controls = column(
        row(column(slider_X_RADIUS, slider_Y_RADIUS), column(contour_start, fit_button)),
        row(
            column(widgetbox(ls_div), radio_button_group),
            column(select_plane, widgetbox(checkbox_group)),
        ),
    )

    # reclustering tab
    struct_el = Select(
        title="Structuring element:",
        value="disk",
        options=["square", "disk", "rectangle", "None"],
        width=100,
    )
    struct_el_size = TextInput(
        value="3", title="Size(width/radius or width,height for rectangle):", width=100
    )

    recluster = Button(label="Re-cluster", button_type="warning")
    recluster.on_event(ButtonClick, recluster_peaks)

    # edit_fits tabs
    fitting_layout = fitting_controls
    log_layout = fit_reports
    recluster_layout = widgetbox(
        row(clust_div, column(contour_start, struct_el, struct_el_size, recluster))
    )
    save_layout = column(savefilename, button, exit_button)

    fitting_tab = Panel(child=fitting_layout, title="Peak fitting")
    log_tab = Panel(child=log_layout, title="Log")
    recluster_tab = Panel(child=recluster_layout, title="Re-cluster peaks")
    save_tab = Panel(child=save_layout, title="Save edited peaklist")
    tabs = Tabs(tabs=[fitting_tab, log_tab, recluster_tab, save_tab])

    # for running fit_peaks from edit_fits
    # fit_all_layout =
    # fit_all_tab = Panel(child=fit_all_layout)
    # fit_all_result = Panel(child=fit_all_result_layout)
    # fit_all_tabs = Tabs(tabs=[fit_all_tab, fit_all_result])


    #curdoc().add_root(
    #    column(
    #        intro_div,
    #        row(column(p, doc_link), column(data_table, tabs)),
    #        sizing_mode="stretch_both",
    #    )
    #)
    #curdoc().title = "peakipy: Edit Fits"
    doc.add_root(
        column(
            intro_div,
            row(column(p, doc_link), column(data_table, tabs)),
            sizing_mode="stretch_both",
        )
    )
    doc.title = "peakipy: Edit Fits"


def check_input(args):
    """ validate commandline input """
    schema = Schema(
        {
            "<peaklist>": And(
                os.path.exists,
                open,
                error=f"{args['<peaklist>']} should exist and be readable",
            ),
            "<data>": And(
                os.path.exists,
                ng.pipe.read,
                error=f"{args['<data>']} either does not exist or is not an NMRPipe format 2D or 3D",
            ),
            "--dims": And(
                lambda n: [int(i) for i in eval(n)],
                error="--dims should be list of integers e.g. --dims=0,1,2",
            ),
        }
    )

    try:
        args = schema.validate(args)
        return args
    except SchemaError as e:
        exit(e)


def main(args):
    global argv
    argv = args
    # docopt(__doc__, argv)
    run_log()
    server = Server({'/': bokeh_script})
    server.start()
    print('Opening peakipy: Edit fits on http://localhost:5006/')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


if __name__ == "__main__":

    args = sys.argv[1:]
    main(args)