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
from colorama import Fore, init

init(autoreset=True)

import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import magma, autumn, viridis
from skimage.filters import threshold_otsu

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

from bokeh.server.server import Server
from bokeh.palettes import PuBuGn9, Category20

from peakipy.core import run_log, LoadData

log_style = "overflow:scroll;"
log_div = """<div style=%s>%s</div>"""


class BokehScript:
    def __init__(self, argv):

        args = docopt(__doc__, argv=argv)
        self._args = check_input(args)
        self._path = Path(args.get("<peaklist>"))
        self._data_path = args.get("<data>")
        self.read_config()
        self._peakipy_data = LoadData(
            self._path, self._data_path, dims=self.dims, verbose=True
        )
        # check dataframe is usable
        self.peakipy_data.check_data_frame()
        # make temporary paths
        self.make_temp_files()

        self.make_data_source()
        self.setup_radii_sliders()
        self.setup_save_buttons()
        self.setup_quit_button()
        self.setup_plot()

    def init(self, doc):
        """ initialise the bokeh app """

        doc.add_root(
            column(
                self.intro_div,
                row(column(self.p, self.doc_link), column(self.data_table, self.tabs)),
                sizing_mode="stretch_both",
            )
        )
        doc.title = "peakipy: Edit Fits"

    @property
    def args(self):
        return self._args

    @property
    def path(self):
        return self._path

    @property
    def data_path(self):
        return self._data_path

    @property
    def peakipy_data(self):
        return self._peakipy_data

    def make_temp_files(self):
        # Temp files
        self.TEMP_PATH = Path("tmp")
        self.TEMP_PATH.mkdir(parents=True, exist_ok=True)

        self.TEMP_OUT_CSV = self.TEMP_PATH / Path("tmp_out.csv")
        self.TEMP_INPUT_CSV = self.TEMP_PATH / Path("tmp.csv")

        self.TEMP_OUT_PLOT = self.TEMP_PATH / Path("plots")
        self.TEMP_OUT_PLOT.mkdir(parents=True, exist_ok=True)

    def make_data_source(self):
        # make datasource
        self.source = ColumnDataSource()
        self.source.data = ColumnDataSource.from_df(self.peakipy_data.df)
        return self.source

    def read_config(self):
        #  read dims from config
        config_path = Path("peakipy.config")
        if config_path.exists():
            try:
                config = json.load(open(config_path))
                print(f"Using config file with --dims={config.get('--dims')}")
                dims = config.get("--dims", [0, 1, 2])
                self._dims = ",".join(str(i) for i in dims)
                self.thres = config.get("--thres")
            except json.decoder.JSONDecodeError:
                print(
                    Fore.RED
                    + "Your peakipy.config file is corrupted - maybe your JSON is not correct..."
                )
                print(Fore.RED + "Not using it.")
                self._dims = self.args.get("--dims")
                self.thres = False

        else:
            # get dim numbers from commandline
            self._dims = self.args.get("--dims")
            self.thres = False

        self.dims = [int(i) for i in self._dims.split(",")]

    def setup_radii_sliders(self):
        # configure sliders for setting radii
        self.slider_X_RADIUS = Slider(
            title="X_RADIUS - ppm",
            start=0.001,
            end=0.200,
            value=0.040,
            step=0.001,
            format="0[.]000",
        )
        self.slider_Y_RADIUS = Slider(
            title="Y_RADIUS - ppm",
            start=0.010,
            end=2.000,
            value=0.400,
            step=0.001,
            format="0[.]000",
        )

        self.slider_X_RADIUS.on_change(
            "value", lambda attr, old, new: self.slider_callback_x(attr, old, new)
        )
        self.slider_Y_RADIUS.on_change(
            "value", lambda attr, old, new: self.slider_callback_y(attr, old, new)
        )

    def setup_save_buttons(self):
        # save file
        self.savefilename = TextInput(
            title="Save file as (.csv)", placeholder="edited_peaks.csv"
        )
        self.button = Button(label="Save", button_type="success")
        self.button.on_event(ButtonClick, self.save_peaks)

    def setup_quit_button(self):
        # Quit button
        self.exit_button = Button(label="Quit", button_type="warning")
        self.exit_button.on_event(ButtonClick, self.exit_edit_peaks)

    def setup_plot(self):
        """" code to setup the bokeh plots """
        #  make bokeh figure
        tools = [
            "tap",
            "box_zoom",
            "lasso_select",
            "box_select",
            "wheel_zoom",
            "pan",
            "reset",
        ]
        self.p = figure(
            x_range=(self.peakipy_data.f2_ppm_0, self.peakipy_data.f2_ppm_1),
            y_range=(self.peakipy_data.f1_ppm_0, self.peakipy_data.f1_ppm_1),
            x_axis_label=f"{self.peakipy_data.f2_label} - ppm",
            y_axis_label=f"{self.peakipy_data.f1_label} - ppm",
            tools=tools,
            active_drag="pan",
            active_scroll="wheel_zoom",
            active_tap=None,
        )
        if not self.thres:
            self.thres = threshold_otsu(self.peakipy_data.data[0])
        self.contour_start = self.thres  # contour level start value
        self.contour_num = 20  # number of contour levels
        self.contour_factor = 1.20  # scaling factor between contour levels
        cl = self.contour_start * self.contour_factor ** np.arange(self.contour_num)
        if len(cl) > 1 and np.min(np.diff(cl)) <= 0.0:
            print(f"Setting contour levels to np.abs({cl})")
            cl = np.abs(cl)
        self.extent = (
            self.peakipy_data.f2_ppm_0,
            self.peakipy_data.f2_ppm_1,
            self.peakipy_data.f1_ppm_0,
            self.peakipy_data.f1_ppm_1,
        )
        self.spec_source = get_contour_data(
            self.peakipy_data.data[0], cl, extent=self.extent, cmap=viridis
        )
        #  negative contours
        self.spec_source_neg = get_contour_data(
            self.peakipy_data.data[0] * -1.0, cl, extent=self.extent, cmap=autumn
        )
        self.p.multi_line(
            xs="xs", ys="ys", line_color="line_color", source=self.spec_source
        )
        self.p.multi_line(
            xs="xs", ys="ys", line_color="line_color", source=self.spec_source_neg
        )
        # contour_num = Slider(title="contour number", value=20, start=1, end=50,step=1)
        # contour_start = Slider(title="contour start", value=100000, start=1000, end=10000000,step=1000)
        self.contour_start = TextInput(
            value="%.2e" % self.thres, title="Contour level:", width=100
        )
        # contour_factor = Slider(title="contour factor", value=1.20, start=1., end=2.,step=0.05)
        self.contour_start.on_change("value", self.update_contour)
        # for w in [contour_num,contour_start,contour_factor]:
        #    w.on_change("value",update_contour)

        #  plot mask outlines
        el = self.p.ellipse(
            x="X_PPM",
            y="Y_PPM",
            width="X_DIAMETER_PPM",
            height="Y_DIAMETER_PPM",
            source=self.source,
            fill_color="color",
            fill_alpha=0.1,
            line_dash="dotted",
            line_color="red",
        )

        self.p.add_tools(
            HoverTool(
                tooltips=[
                    ("Index", "$index"),
                    ("Assignment", "@ASS"),
                    ("CLUSTID", "@CLUSTID"),
                    ("RADII", "@X_RADIUS_PPM{0.000}, @Y_RADIUS_PPM{0.000}"),
                    (
                        f"{self.peakipy_data.f2_label},{self.peakipy_data.f1_label}",
                        "$x{0.000} ppm, $y{0.000} ppm",
                    ),
                ],
                mode="mouse",
                # add renderers
                renderers=[el],
            )
        )
        # p.toolbar.active_scroll = "auto"
        # draw border around spectrum area
        spec_border_x = [
            self.peakipy_data.f2_ppm_min,
            self.peakipy_data.f2_ppm_min,
            self.peakipy_data.f2_ppm_max,
            self.peakipy_data.f2_ppm_max,
            self.peakipy_data.f2_ppm_min,
        ]

        spec_border_y = [
            self.peakipy_data.f1_ppm_min,
            self.peakipy_data.f1_ppm_max,
            self.peakipy_data.f1_ppm_max,
            self.peakipy_data.f1_ppm_min,
            self.peakipy_data.f1_ppm_min,
        ]

        self.p.line(
            spec_border_x,
            spec_border_y,
            line_width=1,
            line_color="black",
            line_dash="dotted",
            line_alpha=0.5,
        )
        self.p.circle(x="X_PPM", y="Y_PPM", source=self.source, color="color")
        # plot cluster numbers
        self.p.text(
            x="X_PPM",
            y="Y_PPM",
            text="CLUSTID",
            text_color="color",
            source=self.source,
            text_font_size="8pt",
            text_font_style="bold",
        )

        self.p.on_event(DoubleTap, self.peak_pick_callback)

        self.pos_neg_contour_dic = {0: "pos/neg", 1: "pos", 2: "neg"}
        self.pos_neg_contour_radiobutton = RadioButtonGroup(
            labels=[
                self.pos_neg_contour_dic[i] for i in self.pos_neg_contour_dic.keys()
            ],
            active=0,
        )
        self.pos_neg_contour_radiobutton.on_change("active", self.update_contour)
        # call fit_peaks
        self.fit_button = Button(label="Fit selected cluster", button_type="primary")
        # lineshape selection
        self.lineshapes = {
            0: "PV",
            1: "V",
            2: "G",
            3: "L",
            4: "PV_PV",
            # 5: "PV_L",
            # 6: "PV_G",
            # 7: "G_L",
        }
        self.radio_button_group = RadioButtonGroup(
            labels=[self.lineshapes[i] for i in self.lineshapes.keys()], active=0
        )
        self.ls_div = Div(
            text="""Choose lineshape you wish to fit. This can be Voigt (V), pseudo-Voigt (PV), Gaussian (G), Lorentzian (L).
            PV_PV fits a PV lineshape with independent "fraction" parameters for the direct and indirect dimensions"""
        )
        self.clust_div = Div(
            text="""If you want to adjust how the peaks are automatically clustered then try changing the
                width/diameter/height (integer values) of the structuring element used during the binary dilation step
                (you can also remove it by selecting 'None'). Increasing the size of the structuring element will cause
                peaks to be more readily incorporated into clusters. Be sure to save your peak list before doing this as
                any manual edits will be lost."""
        )
        self.intro_div = Div(
            text="""<h2>peakipy - interactive fit adjustment </h2> 
            """
        )

        self.doc_link = Div(
            text="<h3><a href='https://j-brady.github.io/peakipy/build/usage/instructions.html', target='_blank'> ℹ️ click here for documentation</a></h3>"
        )
        self.fit_reports = ""
        self.fit_reports_div = Div(text="", height=400, style={"overflow": "scroll"})
        # Plane selection
        self.select_planes_list = [
            f"{i}"
            for i in range(self.peakipy_data.data.shape[self.peakipy_data.planes])
        ]
        self.select_plane = Select(
            title="Select plane:",
            value=self.select_planes_list[0],
            options=self.select_planes_list,
        )
        self.select_planes_dic = {
            f"{i}": i
            for i in range(self.peakipy_data.data.shape[self.peakipy_data.planes])
        }
        self.select_plane.on_change("value", self.update_contour)

        self.checkbox_group = CheckboxGroup(
            labels=["fit current plane only"], active=[]
        )

        #  not sure this is needed
        selected_df = self.peakipy_data.df.copy()

        self.fit_button.on_event(ButtonClick, self.fit_selected)

        columns = [
            TableColumn(field="ASS", title="Assignment"),
            TableColumn(field="CLUSTID", title="Cluster", editor=IntEditor()),
            TableColumn(
                field="X_PPM",
                title=f"{self.peakipy_data.f2_label}",
                editor=NumberEditor(step=0.0001),
                formatter=NumberFormatter(format="0.0000"),
            ),
            TableColumn(
                field="Y_PPM",
                title=f"{self.peakipy_data.f1_label}",
                editor=NumberEditor(step=0.0001),
                formatter=NumberFormatter(format="0.0000"),
            ),
            TableColumn(
                field="X_RADIUS_PPM",
                title=f"{self.peakipy_data.f2_label} radius (ppm)",
                editor=NumberEditor(step=0.0001),
                formatter=NumberFormatter(format="0.0000"),
            ),
            TableColumn(
                field="Y_RADIUS_PPM",
                title=f"{self.peakipy_data.f1_label} radius (ppm)",
                editor=NumberEditor(step=0.0001),
                formatter=NumberFormatter(format="0.0000"),
            ),
            TableColumn(
                field="XW_HZ",
                title=f"{self.peakipy_data.f2_label} LW (Hz)",
                editor=NumberEditor(step=0.01),
                formatter=NumberFormatter(format="0.00"),
            ),
            TableColumn(
                field="YW_HZ",
                title=f"{self.peakipy_data.f1_label} LW (Hz)",
                editor=NumberEditor(step=0.01),
                formatter=NumberFormatter(format="0.00"),
            ),
            TableColumn(
                field="VOL", title="Volume", formatter=NumberFormatter(format="0.0")
            ),
            TableColumn(
                field="include",
                title="Include",
                editor=SelectEditor(options=["yes", "no"]),
            ),
            TableColumn(field="MEMCNT", title="MEMCNT", editor=IntEditor()),
        ]

        self.data_table = DataTable(
            source=self.source, columns=columns, editable=True, fit_columns=True
        )

        # callback for adding
        # source.selected.on_change('indices', callback)
        self.source.selected.on_change("indices", self.select_callback)

        # Document layout
        fitting_controls = column(
            row(
                column(self.slider_X_RADIUS, self.slider_Y_RADIUS),
                column(
                    row(
                        widgetbox(self.contour_start, self.pos_neg_contour_radiobutton)
                    ),
                    widgetbox(self.fit_button),
                ),
            ),
            row(
                column(widgetbox(self.ls_div), widgetbox(self.radio_button_group)),
                column(widgetbox(self.select_plane), widgetbox(self.checkbox_group)),
            ),
        )

        # reclustering tab
        self.struct_el = Select(
            title="Structuring element:",
            value="disk",
            options=["square", "disk", "rectangle", "None", "mask_method"],
            width=100,
        )
        self.struct_el_size = TextInput(
            value="3",
            title="Size(width/radius or width,height for rectangle):",
            width=100,
        )

        self.recluster = Button(label="Re-cluster", button_type="warning")
        self.recluster.on_event(ButtonClick, self.recluster_peaks)

        # edit_fits tabs
        fitting_layout = fitting_controls
        log_layout = self.fit_reports_div
        recluster_layout = row(
            self.clust_div,
            column(
                self.contour_start, self.struct_el, self.struct_el_size, self.recluster
            ),
        )
        save_layout = column(self.savefilename, self.button, self.exit_button)

        fitting_tab = Panel(child=fitting_layout, title="Peak fitting")
        log_tab = Panel(child=log_layout, title="Log")
        recluster_tab = Panel(child=recluster_layout, title="Re-cluster peaks")
        save_tab = Panel(child=save_layout, title="Save edited peaklist")
        self.tabs = Tabs(
            tabs=[fitting_tab, log_tab, recluster_tab, save_tab],
            sizing_mode="scale_both",
        )

    def recluster_peaks(self, event):

        if self.struct_el.value == "mask_method":
            self.struc_size = tuple(
                [float(i) for i in self.struct_el_size.value.split(",")]
            )
            print(self.struc_size)
            self.peakipy_data.mask_method(overlap=self.struc_size[0])
        else:
            self.struc_size = tuple(
                [int(i) for i in self.struct_el_size.value.split(",")]
            )
            print(self.struc_size)
            self.peakipy_data.clusters(
                thres=eval(self.contour_start.value),
                struc_el=self.struct_el.value,
                struc_size=self.struc_size,
            )
        # update data source
        self.source.data = ColumnDataSource.from_df(self.peakipy_data.df)

        return self.peakipy_data.df

    def update_memcnt(self):

        for ind, group in self.peakipy_data.df.groupby("CLUSTID"):
            self.peakipy_data.df.loc[group.index, "MEMCNT"] = len(group)

        # set cluster colors (set to black if singlet peaks)
        self.peakipy_data.df["color"] = self.peakipy_data.df.apply(
            lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black",
            axis=1,
        )
        # change color of excluded peaks
        include_no = self.peakipy_data.df.include == "no"
        self.peakipy_data.df.loc[include_no, "color"] = "ghostwhite"
        # update source data
        self.source.data = ColumnDataSource.from_df(self.peakipy_data.df)
        return self.peakipy_data.df

    def fit_selected(self, event):

        selectionIndex = self.source.selected.indices
        current = self.peakipy_data.df.iloc[selectionIndex]

        self.peakipy_data.df.loc[
            selectionIndex, "X_RADIUS_PPM"
        ] = self.slider_X_RADIUS.value
        self.peakipy_data.df.loc[
            selectionIndex, "Y_RADIUS_PPM"
        ] = self.slider_Y_RADIUS.value

        self.peakipy_data.df.loc[selectionIndex, "X_DIAMETER_PPM"] = (
            current["X_RADIUS_PPM"] * 2.0
        )
        self.peakipy_data.df.loc[selectionIndex, "Y_DIAMETER_PPM"] = (
            current["Y_RADIUS_PPM"] * 2.0
        )

        selected_df = self.peakipy_data.df[
            self.peakipy_data.df.CLUSTID.isin(list(current.CLUSTID))
        ]

        selected_df.to_csv(self.TEMP_INPUT_CSV)

        lineshape = self.lineshapes[self.radio_button_group.active]
        if self.checkbox_group.active == []:
            print(Fore.YELLOW + "Using LS = ", lineshape)
            fit_command = f"peakipy fit {self.TEMP_INPUT_CSV} {self.data_path} {self.TEMP_OUT_CSV} --lineshape={lineshape} --dims={self._dims}"
            plot_command = f"peakipy check {self.TEMP_OUT_CSV} {self.data_path} -l -i -s --outname={self.TEMP_OUT_PLOT / Path('tmp.pdf')}"
        else:
            plane_index = self.select_plane.value
            print(Fore.YELLOW + "Using LS = ", lineshape)
            print(Fore.YELLOW + f"Only fitting plane {plane_index}")
            fit_command = f"peakipy fit {self.TEMP_INPUT_CSV} {self.data_path} {self.TEMP_OUT_CSV} --lineshape={lineshape} --dims={self._dims} --plane={plane_index}"
            plot_command = f"peakipy check {self.TEMP_OUT_CSV} {self.data_path} -l -i -s --outname={self.TEMP_OUT_PLOT / Path('tmp.pdf')} --plane={plane_index}"

        print(Fore.BLUE + fit_command)
        self.fit_reports += fit_command + "<br>"

        stdout = check_output(fit_command.split(" "))
        self.fit_reports += stdout.decode() + "<br><hr><br>"
        self.fit_reports = self.fit_reports.replace("\n", "<br>")
        self.fit_reports_div.text = log_div % (log_style, self.fit_reports)
        # plot data
        os.system(plot_command)

    def save_peaks(self, event):

        if self.savefilename.value:
            to_save = Path(self.savefilename.value)
        else:
            to_save = Path(self.savefilename.placeholder)

        if to_save.exists():
            shutil.copy(f"{to_save}", f"{to_save}.bak")
            print(f"Making backup {to_save}.bak")

        print(Fore.GREEN + f"Saving peaks to {to_save}")
        if to_save.suffix == ".csv":
            self.peakipy_data.df.to_csv(to_save, float_format="%.4f", index=False)
        else:
            self.peakipy_data.df.to_pickle(to_save)

    def select_callback(self, attrname, old, new):
        # print(Fore.RED + "Calling Select Callback")
        # selectionIndex = self.source.selected.indices
        # current = self.peakipy_data.df.iloc[selectionIndex]

        for col in self.peakipy_data.df.columns:
            self.peakipy_data.df.loc[:, col] = self.source.data[col]
        # self.source.data = ColumnDataSource.from_df(self.peakipy_data.df)
        # update memcnt
        self.update_memcnt()
        # print(Fore.YELLOW + "Finished Calling Select Callback")

    def peak_pick_callback(self, event):
        # global so that df is updated globally
        x_radius_ppm = 0.035
        y_radius_ppm = 0.35
        x_radius = x_radius_ppm * self.peakipy_data.pt_per_ppm_f2
        y_radius = y_radius_ppm * self.peakipy_data.pt_per_ppm_f1
        x_diameter_ppm = x_radius_ppm * 2.0
        y_diameter_ppm = y_radius_ppm * 2.0
        clustid = self.peakipy_data.df.CLUSTID.max() + 1
        index = self.peakipy_data.df.INDEX.max() + 1
        x_ppm = event.x
        y_ppm = event.y
        x_axis = self.peakipy_data.uc_f2.f(x_ppm, "ppm")
        y_axis = self.peakipy_data.uc_f1.f(y_ppm, "ppm")
        xw_hz = 20.0
        yw_hz = 20.0
        xw = xw_hz * self.peakipy_data.pt_per_hz_f2
        yw = yw_hz * self.peakipy_data.pt_per_hz_f1
        assignment = f"test_peak_{index}_{clustid}"
        height = self.peakipy_data.data[0][int(y_axis), int(x_axis)]
        volume = height
        print(
            Fore.BLUE + f"""Adding peak at {assignment}: {event.x:.3f},{event.y:.3f}"""
        )

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
        self.peakipy_data.df = self.peakipy_data.df.append(new_peak, ignore_index=True)
        self.update_memcnt()

    def slider_callback_x(self, attrname, old, new):

        selectionIndex = self.source.selected.indices
        current = self.peakipy_data.df.iloc[selectionIndex]
        self.peakipy_data.df.loc[selectionIndex, "X_RADIUS"] = (
            self.slider_X_RADIUS.value * self.peakipy_data.pt_per_ppm_f2
        )
        self.peakipy_data.df.loc[
            selectionIndex, "X_RADIUS_PPM"
        ] = self.slider_X_RADIUS.value

        self.peakipy_data.df.loc[selectionIndex, "X_DIAMETER_PPM"] = (
            current["X_RADIUS_PPM"] * 2.0
        )
        self.peakipy_data.df.loc[selectionIndex, "X_DIAMETER"] = (
            current["X_RADIUS"] * 2.0
        )

        # set edited rows to True
        self.peakipy_data.df.loc[selectionIndex, "Edited"] = True

        self.source.data = ColumnDataSource.from_df(self.peakipy_data.df)

    def slider_callback_y(self, attrname, old, new):

        selectionIndex = self.source.selected.indices
        current = self.peakipy_data.df.iloc[selectionIndex]
        self.peakipy_data.df.loc[selectionIndex, "Y_RADIUS"] = (
            self.slider_Y_RADIUS.value * self.peakipy_data.pt_per_ppm_f1
        )
        self.peakipy_data.df.loc[
            selectionIndex, "Y_RADIUS_PPM"
        ] = self.slider_Y_RADIUS.value

        self.peakipy_data.df.loc[selectionIndex, "Y_DIAMETER_PPM"] = (
            current["Y_RADIUS_PPM"] * 2.0
        )
        self.peakipy_data.df.loc[selectionIndex, "Y_DIAMETER"] = (
            current["Y_RADIUS"] * 2.0
        )

        # set edited rows to True
        self.peakipy_data.df.loc[selectionIndex, "Edited"] = True

        self.source.data = ColumnDataSource.from_df(self.peakipy_data.df)

    # def slider_callback(self, attrname, old, new, dim="X"):
    #
    #     selectionIndex = self.source.selected.indices
    #     current = self.peakipy_data.df.iloc[selectionIndex]
    #     self.peakipy_data.df.loc[selectionIndex, f"{dim}_RADIUS"] = (
    #             self.slider_Y_RADIUS.value * self.peakipy_data.pt_per_ppm_f1
    #     )
    #     self.peakipy_data.df.loc[
    #         selectionIndex, f"{dim}_RADIUS_PPM"
    #     ] = self.slider_Y_RADIUS.value
    #
    #     self.peakipy_data.df.loc[selectionIndex, f"{dim}_DIAMETER_PPM"] = (
    #             current[f"{dim}_RADIUS_PPM"] * 2.0
    #     )
    #     self.peakipy_data.df.loc[selectionIndex, f"{dim}_DIAMETER"] = (
    #             current[f"{dim}_RADIUS"] * 2.0
    #     )
    #
    #     set edited rows to True
    #     self.peakipy_data.df.loc[selectionIndex, "Edited"] = True

    # selected_df = df[df.CLUSTID.isin(list(current.CLUSTID))]
    # print(list(selected_df))
    # self.source.data = ColumnDataSource.from_df(self.peakipy_data.df)

    # def slider_callback_x(self, attrname, old, new):
    #
    #     self.slider_callback(attrname, old, new, dim="X")
    #
    # def slider_callback_y(self, attrname, old, new):
    #
    #     self.slider_callback(attrname, old, new, dim="Y")

    def update_contour(self, attrname, old, new):

        new_cs = eval(self.contour_start.value)
        cl = new_cs * self.contour_factor ** np.arange(self.contour_num)
        if len(cl) > 1 and np.min(np.diff(cl)) <= 0.0:
            print(f"Setting contour levels to np.abs({cl})")
            cl = np.abs(cl)
        plane_index = self.select_planes_dic[self.select_plane.value]

        pos_neg = self.pos_neg_contour_dic[self.pos_neg_contour_radiobutton.active]
        if pos_neg == "pos/neg":
            self.spec_source.data = get_contour_data(
                self.peakipy_data.data[plane_index],
                cl,
                extent=self.extent,
                cmap=viridis,
            ).data
            self.spec_source_neg.data = get_contour_data(
                self.peakipy_data.data[plane_index] * -1.0,
                cl,
                extent=self.extent,
                cmap=autumn,
            ).data

        elif pos_neg == "pos":
            self.spec_source.data = get_contour_data(
                self.peakipy_data.data[plane_index],
                cl,
                extent=self.extent,
                cmap=viridis,
            ).data
            self.spec_source_neg.data = get_contour_data(
                self.peakipy_data.data[plane_index] * 0.0,
                cl,
                extent=self.extent,
                cmap=autumn,
            ).data

        elif pos_neg == "neg":
            self.spec_source.data = get_contour_data(
                self.peakipy_data.data[plane_index] * 0.0,
                cl,
                extent=self.extent,
                cmap=viridis,
            ).data
            self.spec_source_neg.data = get_contour_data(
                self.peakipy_data.data[plane_index] * -1.0,
                cl,
                extent=self.extent,
                cmap=autumn,
            ).data

        # print("Value of checkbox",checkbox_group.active)

    def exit_edit_peaks(self, event):
        sys.exit()


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


def check_input(args):
    """ validate commandline input """
    schema = Schema(
        {
            "<peaklist>": And(
                os.path.exists,
                open,
                error=Fore.RED
                + f"{args['<peaklist>']} should exist and be readable .csv file",
            ),
            "<data>": And(
                os.path.exists,
                ng.pipe.read,
                error=Fore.RED
                + f"{args['<data>']} either does not exist or is not an NMRPipe format 2D or 3D",
            ),
            "--dims": And(
                lambda n: [int(i) for i in eval(n)],
                error=Fore.RED + "--dims should be list of integers e.g. --dims=0,1,2",
            ),
        }
    )

    try:
        args = schema.validate(args)
        return args
    except SchemaError as e:
        sys.exit(e)


def main(args):
    from bokeh.util.browser import view

    run_log()
    bs = BokehScript(args)
    server = Server({"/edit": bs.init})
    server.start()
    print(Fore.GREEN + "Opening peakipy: Edit fits on http://localhost:5006/edit")
    server.io_loop.add_callback(server.show, "/edit")
    server.io_loop.start()


if __name__ == "__main__":

    args = sys.argv[1:]
    main(args)
