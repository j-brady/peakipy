from pathlib import Path
import panel as pn

from edit import BokehScript
from check_panel import create_check_panel

pn.extension("plotly")
pn.config.theme = "dark"

bs = BokehScript(
    peaklist_path=Path("./edited_peaks.csv"), data_path=Path("./test1.ft2")
)
bokeh_pane = pn.pane.Bokeh(bs.p)
table_pane = pn.pane.Bokeh(bs.data_table)
spectrum_view_settings = pn.WidgetBox(
    "# View settings", bs.pos_neg_contour_radiobutton, bs.contour_start
)
button = pn.widgets.Button(name="Click me", button_type="primary")
fit_controls = pn.WidgetBox(
    "# Fit controls", bs.select_lineshape_radiobuttons, bs.fit_button, button
)


def b(event):
    check_app.loading = True
    check_panel = create_check_panel(bs.TEMP_OUT_CSV, bs.data_path, edit_panel=True)
    check_app.objects = check_panel.objects
    check_app.loading = False


button.on_click(b)
fit_app = pn.Card(
    pn.Column(bokeh_pane, fit_controls, spectrum_view_settings, table_pane),
    title="Peakipy fit",
)
check_app = pn.Card(title="Peakipy check")
app = pn.Column(fit_app, check_app)
server = app.show(threaded=True)
