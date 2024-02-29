from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache

import panel as pn
from typer import Typer

from peakipy.cli.edit import BokehScript
from peakipy.cli.check_panel import create_check_panel

app = Typer()

pn.extension("plotly")
pn.config.theme = "dark"


@dataclass
class Data:
    peaklist_path: Path = Path("./test.csv")
    data_path: Path = Path("./test.ft2")
    _bs: BokehScript = field(init=False)

    def load_data(self):
        self._bs = BokehScript(self.peaklist_path, self.data_path)

    @property
    def bs(self):
        return self._bs


@lru_cache(maxsize=1)
def data_singleton():
    return Data()


def panel_app():
    data = data_singleton()
    bs = data.bs
    bokeh_pane = pn.pane.Bokeh(bs.p)
    # table_pane = pn.pane.Bokeh(bs.data_table)
    table_pane = pn.widgets.Tabulator(
        bs.peakipy_data.df[
            [
                "ASS",
                "CLUSTID",
                "X_PPM",
                "Y_PPM",
                "X_RADIUS_PPM",
                "Y_RADIUS_PPM",
                "XW_HZ",
                "YW_HZ",
                "VOL",
                "include",
                "MEMCNT",
            ]
        ]
    )

    spectrum_view_settings = pn.WidgetBox(
        "# Contour settings", bs.pos_neg_contour_radiobutton, bs.contour_start
    )
    button = pn.widgets.Button(name="Fit selected cluster(s)", button_type="primary")
    fit_controls = pn.WidgetBox(
        "# Fit controls",
        bs.select_plane,
        bs.checkbox_group,
        pn.layout.Divider(),
        bs.select_reference_planes_help,
        bs.select_reference_planes,
        pn.layout.Divider(),
        bs.set_initial_fit_threshold_help,
        bs.set_initial_fit_threshold,
        pn.layout.Divider(),
        bs.select_fixed_parameters_help,
        bs.select_fixed_parameters,
        pn.layout.Divider(),
        bs.select_lineshape_radiobuttons_help,
        bs.select_lineshape_radiobuttons,
        pn.layout.Divider(),
        button,
    )

    mask_adjustment_controls = pn.WidgetBox(
        "# Fitting mask adjustment", bs.slider_X_RADIUS, bs.slider_Y_RADIUS
    )

    def b(event):
        check_app.loading = True
        bs.fit_selected(None)
        check_panel = create_check_panel(bs.TEMP_OUT_CSV, bs.data_path, edit_panel=True)
        check_app.objects = check_panel.objects
        check_app.loading = False

    button.on_click(b)
    template = pn.template.BootstrapTemplate(
        title="Peakipy",
        sidebar=[mask_adjustment_controls, fit_controls],
    )
    spectrum = pn.Card(
        pn.Column(pn.Row(bokeh_pane, spectrum_view_settings), table_pane),
        title="Peakipy fit",
    )
    check_app = pn.Card(title="Peakipy check")
    template.main.append(pn.Row(spectrum, check_app))
    template.show()


@app.command()
def main(peaklist_path: Path, data_path: Path):
    data = data_singleton()
    data.peaklist_path = peaklist_path
    data.data_path = data_path
    data.load_data()
    panel_app()


if __name__ == "__main__":
    app()
