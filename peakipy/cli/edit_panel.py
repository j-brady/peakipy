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


def update_peakipy_data_on_edit_of_table(event):
    data = data_singleton()
    column = event.column
    row = event.row
    value = event.value
    data.bs.peakipy_data.df.loc[row, column] = value
    data.bs.update_memcnt()


def panel_app():
    data = data_singleton()
    bs = data.bs
    bokeh_pane = pn.pane.Bokeh(bs.p)
    spectrum_view_settings = pn.WidgetBox(
        "# Contour settings", bs.pos_neg_contour_radiobutton, bs.contour_start
    )
    save_peaklist_box = pn.WidgetBox(
        "# Save your peaklist",
        bs.savefilename,
        bs.button,
        pn.layout.Divider(),
        bs.exit_button,
    )
    recluster_settings = pn.WidgetBox(
        "# Re-cluster your peaks",
        bs.clust_div,
        bs.struct_el,
        bs.struct_el_size,
        pn.layout.Divider(),
        bs.recluster_warning,
        bs.recluster,
        sizing_mode="stretch_width",
    )
    button = pn.widgets.Button(name="Fit selected cluster(s)", button_type="primary")
    fit_controls = pn.WidgetBox(
        "# Fit controls",
        button,
        pn.layout.Divider(),
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
    )

    mask_adjustment_controls = pn.WidgetBox(
        "# Fitting mask adjustment", bs.slider_X_RADIUS, bs.slider_Y_RADIUS
    )

    # bs.source.on_change()
    def fit_peaks_button_click(event):
        check_app.loading = True
        bs.fit_selected(None)
        check_panel = create_check_panel(bs.TEMP_OUT_CSV, bs.data_path, edit_panel=True)
        check_app.objects = check_panel.objects
        check_app.loading = False

    button.on_click(fit_peaks_button_click)

    def update_source_selected_indices(event):
        # print(bs.tablulator_widget.selection)
        # hack to make current selection however, only allows one selection
        # at a time
        bs.tablulator_widget._update_selection([event.value])
        bs.source.selected.indices = bs.tablulator_widget.selection
        # print(bs.tablulator_widget.selection)

    bs.tablulator_widget.on_click(update_source_selected_indices)
    bs.tablulator_widget.on_edit(update_peakipy_data_on_edit_of_table)

    template = pn.template.BootstrapTemplate(
        title="Peakipy",
        sidebar=[mask_adjustment_controls, fit_controls],
    )
    spectrum = pn.Card(
        pn.Column(
            pn.Row(
                bokeh_pane,
                pn.Column(spectrum_view_settings, save_peaklist_box),
                recluster_settings,
            ),
            bs.tablulator_widget,
        ),
        title="Peakipy fit",
    )
    check_app = pn.Card(title="Peakipy check")
    template.main.append(pn.Column(check_app, spectrum))
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
