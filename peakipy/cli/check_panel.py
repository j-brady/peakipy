from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache
import panel as pn
import pandas as pd
import typer

from peakipy.cli.main import check, validate_fit_dataframe

pn.extension()


@dataclass
class Data:
    fits_path: Path = Path("./fits.csv")
    data_path: Path = Path("./test.ft2")
    config_path: Path = Path("./peakipy.config")
    _df: pd.DataFrame = field(init=False)

    def load_dataframe(self):
        self._df = validate_fit_dataframe(pd.read_csv(self.fits_path))

    @property
    def df(self):
        return self._df


@lru_cache(maxsize=1)
def data_singleton():
    return Data()


def get_cluster(cluster):
    data = data_singleton()
    cluster_groups = data.df.groupby("clustid")
    cluster_group = cluster_groups.get_group(cluster)
    df_pane = pn.widgets.Tabulator(
        cluster_group[
            [
                "assignment",
                "clustid",
                "memcnt",
                "plane",
                "amp",
                "height",
                "center_x_ppm",
                "center_y_ppm",
                "fwhm_x_hz",
                "fwhm_y_hz",
                "lineshape",
            ]
        ],
        selectable=False,
        disabled=True,
    )
    return df_pane


@dataclass
class PlotContainer:
    main_figure: pn.pane.Plotly
    residual_figure: pn.pane.Plotly


def create_plotly_pane(cluster, plane):
    data = data_singleton()
    fig, residual_fig = check(
        fits=data.fits_path,
        data_path=data.data_path,
        clusters=[cluster],
        plane=[plane],
        config_path=data.config_path,
        plotly=True,
    )

    fig["layout"].update(height=800, width=800)
    fig = fig.to_dict()
    residual_fig = residual_fig.to_dict()
    return pn.Column(pn.pane.Plotly(fig), pn.pane.Plotly(residual_fig))


app = typer.Typer()


@app.command()
def create_check_panel(
    fits_path: Path,
    data_path: Path,
    config_path: Path = Path("./peakipy.config"),
    edit_panel: bool = False,
):
    data = data_singleton()
    data.fits_path = fits_path
    data.data_path = data_path
    data.config_path = config_path
    data.load_dataframe()

    clusters = [(row.clustid, row.memcnt) for _, row in data.df.iterrows()]

    select_cluster = pn.widgets.Select(
        name="Cluster (number of peaks)", options={f"{c} ({m})": c for c, m in clusters}
    )
    select_plane = pn.widgets.Select(
        name="Plane", options={f"{plane}": plane for plane in data.df.plane.unique()}
    )
    result_table_pane = pn.bind(get_cluster, select_cluster)
    interactive_plotly_pane = pn.bind(
        create_plotly_pane, cluster=select_cluster, plane=select_plane
    )
    info_pane = pn.pane.Markdown(
        "Select a cluster and plane to look at from the dropdown menus"
    )
    check_pane = pn.Card(
        # info_pane,
        # pn.Row(select_cluster, select_plane),
        pn.Row(
            pn.Column(
                pn.Row(
                    pn.Card(interactive_plotly_pane, title="Fitted cluster"),
                    pn.Column(info_pane, select_cluster, select_plane),
                ),
                pn.Card(result_table_pane, title="Fitted parameters for cluster"),
            )
        ),
        title="Peakipy check",
    )
    if edit_panel:
        return check_pane
    else:
        check_pane.show()


if __name__ == "__main__":
    app()
