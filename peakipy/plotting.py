from dataclasses import dataclass, field
from typing import List

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button
from matplotlib.backends.backend_pdf import PdfPages
from rich import print

from peakipy.io import Pseudo3D
from peakipy.utils import df_to_rich_table, bad_color_selection, bad_column_selection


@dataclass
class PlottingDataForPlane:
    pseudo3D: Pseudo3D
    plane_id: int
    plane_lineshape_parameters: pd.DataFrame
    X: np.array
    Y: np.array
    mask: np.array
    individual_masks: List[np.array]
    sim_data: np.array
    sim_data_singles: List[np.array]
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    fit_color: str
    data_color: str
    rcount: int
    ccount: int

    x_plot: np.array = field(init=False)
    y_plot: np.array = field(init=False)
    masked_data: np.array = field(init=False)
    masked_sim_data: np.array = field(init=False)
    residual: np.array = field(init=False)
    single_colors: List = field(init=False)

    def __post_init__(self):
        self.plane_data = self.pseudo3D.data[self.plane_id]
        self.masked_data = self.plane_data.copy()
        self.masked_sim_data = self.sim_data.copy()
        self.masked_data[~self.mask] = np.nan
        self.masked_sim_data[~self.mask] = np.nan

        self.x_plot = self.pseudo3D.uc_f2.ppm(
            self.X[self.min_y : self.max_y, self.min_x : self.max_x]
        )
        self.y_plot = self.pseudo3D.uc_f1.ppm(
            self.Y[self.min_y : self.max_y, self.min_x : self.max_x]
        )
        self.masked_data = self.masked_data[
            self.min_y : self.max_y, self.min_x : self.max_x
        ]
        self.sim_plot = self.masked_sim_data[
            self.min_y : self.max_y, self.min_x : self.max_x
        ]
        self.residual = self.masked_data - self.sim_plot

        for single_mask, single in zip(self.individual_masks, self.sim_data_singles):
            single[~single_mask] = np.nan
        self.sim_data_singles = [
            sim_data_single[self.min_y : self.max_y, self.min_x : self.max_x]
            for sim_data_single in self.sim_data_singles
        ]
        self.single_colors = [
            cm.viridis(i) for i in np.linspace(0, 1, len(self.sim_data_singles))
        ]


def plot_data_is_valid(plot_data: PlottingDataForPlane) -> bool:
    if len(plot_data.x_plot) < 1 or len(plot_data.y_plot) < 1:
        print(
            f"[red]Nothing to plot for cluster {int(plot_data.plane_lineshape_parameters.clustid)}[/red]"
        )
        print(f"[red]x={plot_data.x_plot},y={plot_data.y_plot}[/red]")
        print(
            df_to_rich_table(
                plot_data.plane_lineshape_parameters,
                title="",
                columns=bad_column_selection,
                styles=bad_color_selection,
            )
        )
        plt.close()
        validated = False
        # print(Fore.RED + "Maybe your F1/F2 radii for fitting were too small...")
    elif plot_data.masked_data.shape[0] == 0 or plot_data.masked_data.shape[1] == 0:
        print(f"[red]Nothing to plot for cluster {int(plot_data.plane.clustid)}[/red]")
        print(
            df_to_rich_table(
                plot_data.plane_lineshape_parameters,
                title="Bad plane",
                columns=bad_column_selection,
                styles=bad_color_selection,
            )
        )
        spec_lim_f1 = " - ".join(
            ["%8.3f" % i for i in plot_data.pseudo3D.f1_ppm_limits]
        )
        spec_lim_f2 = " - ".join(
            ["%8.3f" % i for i in plot_data.pseudo3D.f2_ppm_limits]
        )
        print(f"Spectrum limits are {plot_data.pseudo3D.f2_label:4s}:{spec_lim_f2} ppm")
        print(f"                    {plot_data.pseudo3D.f1_label:4s}:{spec_lim_f1} ppm")
        plt.close()
        validated = False
    else:
        validated = True
    return validated


def create_matplotlib_figure(
    plot_data: PlottingDataForPlane,
    pdf: PdfPages,
    individual=False,
    label=False,
    ccpn_flag=False,
    show=True,
    test=False,
):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection="3d")
    if plot_data_is_valid(plot_data):
        cset = ax.contourf(
            plot_data.x_plot,
            plot_data.y_plot,
            plot_data.residual,
            zdir="z",
            offset=np.nanmin(plot_data.masked_data) * 1.1,
            alpha=0.5,
            cmap=cm.coolwarm,
        )
        cbl = fig.colorbar(cset, ax=ax, shrink=0.5, format="%.2e")
        cbl.ax.set_title("Residual", pad=20)

        if individual:
            #  for plotting single fit surfaces
            single_colors = [
                cm.viridis(i)
                for i in np.linspace(0, 1, len(plot_data.sim_data_singles))
            ]
            [
                ax.plot_surface(
                    plot_data.x_plot,
                    plot_data.y_plot,
                    z_single,
                    color=c,
                    alpha=0.5,
                )
                for c, z_single in zip(single_colors, plot_data.sim_data_singles)
            ]
        ax.plot_wireframe(
            plot_data.x_plot,
            plot_data.y_plot,
            plot_data.sim_plot,
            # colors=[cm.coolwarm(i) for i in np.ravel(residual)],
            colors=plot_data.fit_color,
            linestyle="--",
            label="fit",
            rcount=plot_data.rcount,
            ccount=plot_data.ccount,
        )
        ax.plot_wireframe(
            plot_data.x_plot,
            plot_data.y_plot,
            plot_data.masked_data,
            colors=plot_data.data_color,
            linestyle="-",
            label="data",
            rcount=plot_data.rcount,
            ccount=plot_data.ccount,
        )
        ax.set_ylabel(plot_data.pseudo3D.f1_label)
        ax.set_xlabel(plot_data.pseudo3D.f2_label)

        # axes will appear inverted
        ax.view_init(30, 120)

        title = f"Plane={plot_data.plane_id},Cluster={plot_data.plane_lineshape_parameters.clustid.iloc[0]}"
        plt.title(title)
        print(f"[green]Plotting: {title}[/green]")
        out_str = "Volumes (Heights)\n===========\n"
        for _, row in plot_data.plane_lineshape_parameters.iterrows():
            out_str += f"{row.assignment} = {row.amp:.3e} ({row.height:.3e})\n"
            if label:
                ax.text(
                    row.center_x_ppm,
                    row.center_y_ppm,
                    row.height * 1.2,
                    row.assignment,
                    (1, 1, 1),
                )

        ax.text2D(
            -0.5,
            1.0,
            out_str,
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="sans-serif",
            va="top",
            bbox=dict(boxstyle="round", ec="k", fc="k", alpha=0.5),
        )

        ax.legend()

        if show:

            def exit_program(event):
                exit()

            def next_plot(event):
                plt.close()

            axexit = plt.axes([0.81, 0.05, 0.1, 0.075])
            bnexit = Button(axexit, "Exit")
            bnexit.on_clicked(exit_program)
            axnext = plt.axes([0.71, 0.05, 0.1, 0.075])
            bnnext = Button(axnext, "Next")
            bnnext.on_clicked(next_plot)
            if test:
                return
            if ccpn_flag:
                plt.show(windowTitle="", size=(1000, 500))
            else:
                plt.show()
        else:
            pdf.savefig()

            plt.close()


def create_plotly_wireframe_lines(plot_data: PlottingDataForPlane):
    lines = []
    show_legend = lambda x: x < 1
    showlegend = False
    # make simulated data wireframe
    line_marker = dict(color=plot_data.fit_color, width=4)
    counter = 0
    for i, j, k in zip(plot_data.x_plot, plot_data.y_plot, plot_data.sim_plot):
        showlegend = show_legend(counter)
        lines.append(
            go.Scatter3d(
                x=i,
                y=j,
                z=k,
                mode="lines",
                line=line_marker,
                name="fit",
                showlegend=showlegend,
            )
        )
        counter += 1
    for i, j, k in zip(plot_data.x_plot.T, plot_data.y_plot.T, plot_data.sim_plot.T):
        lines.append(
            go.Scatter3d(
                x=i, y=j, z=k, mode="lines", line=line_marker, showlegend=showlegend
            )
        )
    # make experimental data wireframe
    line_marker = dict(color=plot_data.data_color, width=4)
    counter = 0
    for i, j, k in zip(plot_data.x_plot, plot_data.y_plot, plot_data.masked_data):
        showlegend = show_legend(counter)
        lines.append(
            go.Scatter3d(
                x=i,
                y=j,
                z=k,
                mode="lines",
                name="data",
                line=line_marker,
                showlegend=showlegend,
            )
        )
        counter += 1
    for i, j, k in zip(plot_data.x_plot.T, plot_data.y_plot.T, plot_data.masked_data.T):
        lines.append(
            go.Scatter3d(
                x=i, y=j, z=k, mode="lines", line=line_marker, showlegend=showlegend
            )
        )

    return lines


def construct_surface_legend_string(row):
    surface_legend = ""
    surface_legend += row.assignment
    return surface_legend


def create_plotly_surfaces(plot_data: PlottingDataForPlane):
    data = []
    color_scale_values = np.linspace(0, 1, len(plot_data.single_colors))
    color_scale = [
        [val, f"rgb({', '.join('%d'%(i*255) for i in c[0:3])})"]
        for val, c in zip(color_scale_values, plot_data.single_colors)
    ]
    for val, individual_peak, row in zip(
        color_scale_values,
        plot_data.sim_data_singles,
        plot_data.plane_lineshape_parameters.itertuples(),
    ):
        name = construct_surface_legend_string(row)
        colors = np.zeros(shape=individual_peak.shape) + val
        data.append(
            go.Surface(
                z=individual_peak,
                x=plot_data.x_plot,
                y=plot_data.y_plot,
                opacity=0.5,
                surfacecolor=colors,
                colorscale=color_scale,
                showscale=False,
                cmin=0,
                cmax=1,
                name=name,
            )
        )
    return data


def create_residual_contours(plot_data: PlottingDataForPlane):
    contours = go.Contour(
        x=plot_data.x_plot[0], y=plot_data.y_plot.T[0], z=plot_data.residual
    )
    return contours


def create_residual_figure(plot_data: PlottingDataForPlane):
    data = create_residual_contours(plot_data)
    fig = go.Figure(data=data)
    fig.update_layout(
        title="Fit residuals",
        xaxis_title=f"{plot_data.pseudo3D.f2_label} ppm",
        yaxis_title=f"{plot_data.pseudo3D.f1_label} ppm",
        xaxis=dict(range=[plot_data.x_plot.max(), plot_data.x_plot.min()]),
        yaxis=dict(range=[plot_data.y_plot.max(), plot_data.y_plot.min()]),
        
    )
    return fig


def create_plotly_figure(plot_data: PlottingDataForPlane):
    lines = create_plotly_wireframe_lines(plot_data)
    surfaces = create_plotly_surfaces(plot_data)
    fig = go.Figure(data=lines + surfaces)
    fig = update_axis_ranges(fig, plot_data)
    return fig


def update_axis_ranges(fig, plot_data: PlottingDataForPlane):
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[plot_data.x_plot.max(), plot_data.x_plot.min()]),
            yaxis=dict(range=[plot_data.y_plot.max(), plot_data.y_plot.min()]),
            xaxis_title=f"{plot_data.pseudo3D.f2_label} ppm",
            yaxis_title=f"{plot_data.pseudo3D.f1_label} ppm",
            annotations=make_annotations(plot_data),
        )
    )
    return fig


def make_annotations(plot_data: PlottingDataForPlane):
    annotations = []
    for row in plot_data.plane_lineshape_parameters.itertuples():
        annotations.append(
            dict(
                showarrow=True,
                x=row.center_x_ppm,
                y=row.center_y_ppm,
                z=row.height * 1.0,
                text=row.assignment,
                opacity=0.8,
                textangle=0,
                arrowsize=1,
            )
        )
    return annotations


def validate_sample_count(sample_count):
    if type(sample_count) == int:
        sample_count = sample_count
    else:
        raise TypeError("Sample count (ccount, rcount) should be an integer")
    return sample_count


def unpack_plotting_colors(colors):
    match colors:
        case (data_color, fit_color):
            data_color, fit_color = colors
        case _:
            data_color, fit_color = "green", "blue"
    return data_color, fit_color
