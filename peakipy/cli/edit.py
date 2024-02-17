#!/usr/bin/env python3
""" Script for checking fits and editing fit params
"""
import os
import sys
import shutil

from subprocess import check_output
from pathlib import Path


import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from rich import print

from bokeh.io import curdoc
from bokeh.events import ButtonClick, DoubleTap
from bokeh.layouts import row, column, grid
from bokeh.models import ColumnDataSource, Tabs, TabPanel, InlineStyleSheet
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
)
from bokeh.plotting import figure
from bokeh.plotting.contour import contour_data
from bokeh.palettes import PuBuGn9, Category20, Viridis256, RdGy11, Reds256, YlOrRd9

from peakipy.core import LoadData, read_config, StrucEl

log_style = "overflow:scroll;"
log_div = """<div style=%s>%s</div>"""


class BokehScript:
    def __init__(self, peaklist_path: Path, data_path: Path):
        self._path = peaklist_path
        self._data_path = data_path
        args, config = read_config({})
        # self.args = args
        # self.config = config
        self._dims = config.get("dims", [0, 1, 2])
        self.thres = config.get("thres", 1e6)
        self._peakipy_data = LoadData(
            self._path, self._data_path, dims=self._dims, verbose=True
        )
        # check dataframe is usable
        self.peakipy_data.check_data_frame()
        # make temporary paths
        self.make_temp_files()
        self.make_data_source()
        self.setup_radii_sliders()
        self.setup_save_buttons()
        self.setup_set_fixed_parameters()
        self.setup_xybounds()
        self.setup_set_reference_planes()
        self.setup_initial_fit_threshold()
        self.setup_quit_button()
        self.setup_plot()

    def init(self, doc):
        """initialise the bokeh app"""

        doc.add_root(
            column(
                self.intro_div,
                row(column(self.p, self.doc_link), column(self.data_table, self.tabs)),
                sizing_mode="stretch_both",
            )
        )
        doc.title = "peakipy: Edit Fits"
        doc.theme = "dark_minimal"

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

    def setup_set_fixed_parameters(self):
        self.select_fixed_parameters_help = Div(
            text="Select parameters to fix after initial lineshape parameters have been fitted"
        )
        self.select_fixed_parameters = TextInput(
            value="fraction sigma center", width=200
        )

    def setup_xybounds(self):
        self.set_xybounds_help = Div(
            text="If floating the peak centers you can bound the fits in the x and y dimensions. Units of ppm."
        )
        self.set_xybounds = TextInput(placeholder="e.g. 0.01 0.1")

    def get_xybounds(self):
        try:
            x_bound, y_bound = self.set_xybounds.value.split(" ")
            x_bound = float(x_bound)
            y_bound = float(y_bound)
            xy_bounds = x_bound, y_bound
        except:
            xy_bounds = None, None
        return xy_bounds

    def make_xybound_command(self, x_bound, y_bound):
        if (x_bound != None) and (y_bound != None):
            xy_bounds_command = f" --xy-bounds {x_bound} {y_bound}"
        else:
            xy_bounds_command = ""
        return xy_bounds_command

    def setup_set_reference_planes(self):
        self.select_reference_planes_help = Div(
            text="Select reference planes (index starts at 0)"
        )
        self.select_reference_planes = TextInput(placeholder="0 1 2 3")

    def get_reference_planes(self):
        if self.select_reference_planes.value:
            print("You have selected1")
            return self.select_reference_planes.value.split(" ")
        else:
            return []

    def make_reference_planes_command(self, reference_plane_list):
        reference_plane_command = ""
        for plane in reference_plane_list:
            reference_plane_command += f" --reference-plane-index {plane}"
        return reference_plane_command

    def setup_initial_fit_threshold(self):
        self.set_initial_fit_threshold_help = Div(
            text="Set an intensity threshold for selection of planes for initial estimation of lineshape parameters"
        )
        self.set_initial_fit_threshold = TextInput(placeholder="e.g. 1e7")

    def get_initial_fit_threshold(self):
        try:
            initial_fit_threshold = float(self.set_initial_fit_threshold.value)
        except ValueError:
            initial_fit_threshold = None
        return initial_fit_threshold

    def make_initial_fit_threshold_command(self, initial_fit_threshold):
        if initial_fit_threshold is not None:
            initial_fit_threshold_command = (
                f" --initial-fit-threshold {initial_fit_threshold}"
            )
        else:
            initial_fit_threshold_command = ""
        return initial_fit_threshold_command

    def setup_quit_button(self):
        # Quit button
        self.exit_button = Button(label="Quit", button_type="warning")
        self.exit_button.on_event(ButtonClick, self.exit_edit_peaks)

    def setup_plot(self):
        """ " code to setup the bokeh plots"""
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

        self.x_ppm_mesh, self.y_ppm_mesh = np.meshgrid(
            self.peakipy_data.f2_ppm_scale, self.peakipy_data.f1_ppm_scale
        )
        self.positive_contour_renderer = self.p.contour(
            self.x_ppm_mesh,
            self.y_ppm_mesh,
            self.peakipy_data.data[0],
            cl,
            fill_color=YlOrRd9,
            line_color="black",
            line_width=0.25,
        )
        self.negative_contour_renderer = self.p.contour(
            self.x_ppm_mesh,
            self.y_ppm_mesh,
            self.peakipy_data.data[0] * -1.0,
            cl,
            fill_color=Reds256,
            line_color="black",
            line_width=0.25,
        )

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
            fill_alpha=0.25,
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
            line_width=2,
            line_color="red",
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
        self.select_lineshape_radiobuttons = RadioButtonGroup(
            labels=[self.lineshapes[i] for i in self.lineshapes.keys()], active=0
        )
        self.select_lineshape_radiobuttons_help = Div(
            text="""Choose lineshape you wish to fit. This can be Voigt (V), pseudo-Voigt (PV), Gaussian (G), Lorentzian (L).
            PV_PV fits a PV lineshape with independent "fraction" parameters for the direct and indirect dimensions""",
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
            text="<h3><a href='https://j-brady.github.io/peakipy/', target='_blank'> ℹ️ click here for documentation</a></h3>"
        )
        self.fit_reports = ""
        self.fit_reports_div = Div(text="", height=400, styles={"overflow": "scroll"})
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
            TableColumn(field="ASS", title="Assignment", width=500),
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
                width=7,
                editor=SelectEditor(options=["yes", "no"]),
            ),
            TableColumn(field="MEMCNT", title="MEMCNT", editor=IntEditor()),
        ]

        self.data_table = DataTable(
            source=self.source,
            columns=columns,
            editable=True,
            width=1200,
        )
        self.table_style = InlineStyleSheet(
            css="""
                .slick-header-columns {
                    background-color: #00296b !important;
                    font-family: arial;
                    font-weight: bold;
                    font-size: 12pt;
                    color: #FFFFFF;
                    text-align: right;
                }
                .slick-header-column:hover {
                    background: none repeat scroll 0 0 #fdc500;
                }
                .slick-row {
                    font-size: 12pt;
                    font-family: arial;
                    text-align: left;
                }
                .slick-row:hover{
                    background: none repeat scroll 0 0 #7c7c7c;
                }
                .slick-cell {
                    header-font-weight: 500;
                    border-width: 1px 1px 1px 1px;
                    border-color: #d4d4d4;
                    background-color: #00509D;
                    color: #FFFFFF;
                    }
                .slick-cell.selected {
                    header-font-weight: 500;
                    border-width: 1px 1px 1px 1px;
                    border-color: #00509D;
                    background-color: #FDC500;
                    color: black;
                }


                """
        )

        self.data_table.stylesheets = [self.table_style]

        # callback for adding
        # source.selected.on_change('indices', callback)
        self.source.selected.on_change("indices", self.select_callback)

        # # Document layout
        # fitting_controls = column(
        #     row(
        #         column(self.slider_X_RADIUS, self.slider_Y_RADIUS),
        #         column(
        #             row(column(self.contour_start, self.pos_neg_contour_radiobutton)),
        #             column(self.fit_button),
        #         ),
        #     ),
        #     row(
        #         column(column(self.select_lineshape_radiobuttons_help), column(self.select_lineshape_radiobuttons)),
        #         column(column(self.select_plane), column(self.checkbox_group)),
        #         column(self.select_fixed_parameters_help, self.select_fixed_parameters),
        #         column(self.select_reference_planes)
        #     ),
        #     max_width=400,
        # )
        fitting_controls = row(
            column(
                self.slider_X_RADIUS,
                self.slider_Y_RADIUS,
                self.contour_start,
                self.pos_neg_contour_radiobutton,
                self.select_lineshape_radiobuttons_help,
                self.select_lineshape_radiobuttons,
                max_width=400,
            ),
            column(
                self.select_plane,
                self.checkbox_group,
                self.select_fixed_parameters_help,
                self.select_fixed_parameters,
                self.set_xybounds_help,
                self.set_xybounds,
                self.select_reference_planes_help,
                self.select_reference_planes,
                self.set_initial_fit_threshold_help,
                self.set_initial_fit_threshold,
                self.fit_button,
                max_width=400,
            ),
            max_width=800,
        )

        # reclustering tab
        self.struct_el = Select(
            title="Structuring element:",
            value=StrucEl.disk.value,
            options=[i.value for i in StrucEl] + ["None"],
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
        recluster_layout = column(
            row(
                self.clust_div,
            ),
            row(
                column(
                    self.contour_start,
                    self.struct_el,
                    self.struct_el_size,
                    self.recluster,
                )
            ),
            max_width=400,
        )
        save_layout = column(
            self.savefilename, self.button, self.exit_button, max_width=400
        )

        fitting_tab = TabPanel(child=fitting_layout, title="Peak fitting")
        log_tab = TabPanel(child=log_layout, title="Log")
        recluster_tab = TabPanel(child=recluster_layout, title="Re-cluster peaks")
        save_tab = TabPanel(child=save_layout, title="Save edited peaklist")
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
                struc_el=StrucEl(self.struct_el.value),
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

    def unpack_parameters_to_fix(self):
        return self.select_fixed_parameters.value.strip().split(" ")

    def make_fix_command_from_parameters(self, parameters):
        command = ""
        for parameter in parameters:
            command += f" --fix {parameter}"
        return command

    def fit_selected(self, event):
        selectionIndex = self.source.selected.indices
        current = self.peakipy_data.df.iloc[selectionIndex]

        self.peakipy_data.df.loc[selectionIndex, "X_RADIUS_PPM"] = (
            self.slider_X_RADIUS.value
        )
        self.peakipy_data.df.loc[selectionIndex, "Y_RADIUS_PPM"] = (
            self.slider_Y_RADIUS.value
        )

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
        fix_command = self.make_fix_command_from_parameters(
            self.unpack_parameters_to_fix()
        )
        xy_bounds_command = self.make_xybound_command(*self.get_xybounds())
        reference_planes_command = self.make_reference_planes_command(
            self.get_reference_planes()
        )
        initial_fit_threshold_command = self.make_initial_fit_threshold_command(
            self.get_initial_fit_threshold()
        )

        lineshape = self.lineshapes[self.select_lineshape_radiobuttons.active]
        print(f"[yellow]Using LS = {lineshape}[/yellow]")
        if self.checkbox_group.active == []:
            fit_command = f"peakipy fit {self.TEMP_INPUT_CSV} {self.data_path} {self.TEMP_OUT_CSV} --lineshape {lineshape}{fix_command}{reference_planes_command}{initial_fit_threshold_command}{xy_bounds_command}"
            # plot_command = f"peakipy check {self.TEMP_OUT_CSV} {self.data_path} --label --individual --show --outname {self.TEMP_OUT_PLOT / Path('tmp.pdf')}"
            plot_command = f"peakipy-check {self.TEMP_OUT_CSV} {self.data_path}"
        else:
            plane_index = self.select_plane.value
            print(f"[yellow]Only fitting plane {plane_index}[/yellow]")
            fit_command = f"peakipy fit {self.TEMP_INPUT_CSV} {self.data_path} {self.TEMP_OUT_CSV} --lineshape {lineshape} --plane {plane_index}{fix_command}{reference_planes_command}{initial_fit_threshold_command}{xy_bounds_command}"
            # plot_command = f"peakipy check {self.TEMP_OUT_CSV} {self.data_path} --label --individual --outname {self.TEMP_OUT_PLOT / Path('tmp.pdf')} --plane {plane_index} --show"
            plot_command = f"peakipy-check {self.TEMP_OUT_CSV} {self.data_path}"

        print(f"[blue]{fit_command}[/blue]")
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

        print(f"[green]Saving peaks to {to_save}[/green]")
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
            f"""[blue]Adding peak at {assignment}: {event.x:.3f},{event.y:.3f}[/blue]"""
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
        new_peak = {k: [v] for k, v in new_peak.items()}
        new_peak = pd.DataFrame(new_peak)
        self.peakipy_data.df = pd.concat(
            [self.peakipy_data.df, new_peak], ignore_index=True
        )
        self.update_memcnt()

    def slider_callback_x(self, attrname, old, new):
        selectionIndex = self.source.selected.indices
        current = self.peakipy_data.df.iloc[selectionIndex]
        self.peakipy_data.df.loc[selectionIndex, "X_RADIUS"] = (
            self.slider_X_RADIUS.value * self.peakipy_data.pt_per_ppm_f2
        )
        self.peakipy_data.df.loc[selectionIndex, "X_RADIUS_PPM"] = (
            self.slider_X_RADIUS.value
        )

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
        self.peakipy_data.df.loc[selectionIndex, "Y_RADIUS_PPM"] = (
            self.slider_Y_RADIUS.value
        )

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
            self.positive_contour_renderer.set_data(
                contour_data(
                    self.x_ppm_mesh,
                    self.y_ppm_mesh,
                    self.peakipy_data.data[plane_index],
                    cl,
                )
            )
            self.negative_contour_renderer.set_data(
                contour_data(
                    self.x_ppm_mesh,
                    self.y_ppm_mesh,
                    self.peakipy_data.data[plane_index] * -1.0,
                    cl,
                )
            )

        elif pos_neg == "pos":
            self.positive_contour_renderer.set_data(
                contour_data(
                    self.x_ppm_mesh,
                    self.y_ppm_mesh,
                    self.peakipy_data.data[plane_index],
                    cl,
                )
            )
            self.negative_contour_renderer.set_data(
                contour_data(
                    self.x_ppm_mesh,
                    self.y_ppm_mesh,
                    self.peakipy_data.data[plane_index] * 0,
                    cl,
                )
            )

        elif pos_neg == "neg":
            self.positive_contour_renderer.set_data(
                contour_data(
                    self.x_ppm_mesh,
                    self.y_ppm_mesh,
                    self.peakipy_data.data[plane_index] * 0.0,
                    cl,
                )
            )
            self.negative_contour_renderer.set_data(
                contour_data(
                    self.x_ppm_mesh,
                    self.y_ppm_mesh,
                    self.peakipy_data.data[plane_index] * -1.0,
                    cl,
                )
            )

    def exit_edit_peaks(self, event):
        sys.exit()
