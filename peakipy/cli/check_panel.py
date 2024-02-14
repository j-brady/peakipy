from pathlib import Path
import panel as pn
import pandas as pd

from peakipy.cli.main import check

pn.extension()


def get_cluster(cluster):
    cluster_groups = df.groupby("clustid")
    cluster_group = cluster_groups.get_group(cluster)
    df_pane = pn.pane.DataFrame(cluster_group)
    return df_pane


def create_plotly_pane(cluster):
    fig = check(
        fits=fits_path,
        data_path=data_path,
        clusters=[cluster],
        config_path=config_path,
        first=True,
        plotly=True,
    )

    fig["layout"].update(height=800, width=800)
    fig = fig.to_dict()
    return pn.pane.Plotly(fig)


if __name__ == "__main__":
    fits_path = Path("../../test/test_protein_L/fits.csv")
    data_path = Path("../../test/test_protein_L/test1.ft2")
    config_path = Path("../../test/test_protein_L/peakipy.config")
    df = pd.read_csv(fits_path)

    clusters = [(row.clustid, row.memcnt) for _, row in df.iterrows()]

    select = pn.widgets.Select(
        name="Cluster (number of peaks)", options={f"{c} ({m})": c for c, m in clusters}
    )
    interactive_cluster_pane = pn.bind(get_cluster, select)
    interactive_plotly_pane = pn.bind(create_plotly_pane, select)
    check_pane = pn.Card(
        select,
        pn.Row(interactive_plotly_pane, interactive_cluster_pane),
        title="Peakipy check",
    )
    check_pane.show()
