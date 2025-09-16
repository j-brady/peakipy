import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List
import shutil

from rich import print
from rich.table import Table

# for printing dataframes
peaklist_columns_for_printing = ["INDEX", "ASS", "X_PPM", "Y_PPM", "CLUSTID", "MEMCNT"]
bad_column_selection = [
    "clustid",
    "amp",
    "center_x_ppm",
    "center_y_ppm",
    "fwhm_x_hz",
    "fwhm_y_hz",
    "lineshape",
]
bad_color_selection = [
    "green",
    "blue",
    "yellow",
    "red",
    "yellow",
    "red",
    "magenta",
]


def mkdir_tmp_dir(base_path: Path = Path("./")):
    tmp_dir = base_path / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def create_log_path(base_path: Path = Path("./")):
    return base_path / "run_log.txt"


def run_log(log_name="run_log.txt"):
    """Write log file containing time script was run and with which arguments"""
    with open(log_name, "a") as log:
        sys_argv = sys.argv
        sys_argv[0] = Path(sys_argv[0]).name
        run_args = " ".join(sys_argv)
        time_stamp = datetime.now()
        time_stamp = time_stamp.strftime("%A %d %B %Y at %H:%M")
        log.write(f"# Script run on {time_stamp}:\n{run_args}\n")


def df_to_rich_table(df, title: str, columns: List[str], styles: str):
    """Print dataframe using rich library

    Parameters
    ----------
    df : pandas.DataFrame
    title : str
        title of table
    columns : List[str]
        list of column names (must be in df)
    styles : List[str]
        list of styles in same order as columns
    """
    table = Table(title=title)
    for col, style in zip(columns, styles):
        table.add_column(col, style=style)
    for _, row in df.iterrows():
        row = row[columns].values
        str_row = []
        for i in row:
            match i:
                case str():
                    str_row.append(f"{i}")
                case float() if i > 1e5:
                    str_row.append(f"{i:.1e}")
                case float():
                    str_row.append(f"{i:.3f}")
                case bool():
                    str_row.append(f"{i}")
                case int():
                    str_row.append(f"{i}")
        table.add_row(*str_row)
    return table


def load_config(config_path):
    if config_path.exists():
        with open(config_path) as opened_config:
            config_dic = json.load(opened_config)
            return config_dic
    else:
        return {}


def write_config(config_path, config_dic):
    """
    Write a configuration dictionary to a JSON file.
    
    Parameters
    ----------
    config_path : Path
        Path to where the config should be saved.
    config_dic : dict
        Dictionary containing configuration parameters to write to the file.
    """
    with open(config_path, "w") as config:
        config.write(json.dumps(config_dic, sort_keys=True, indent=4))


def update_config_file(config_path, config_kvs):
    config_dic = load_config(config_path)
    config_dic.update(config_kvs)
    write_config(config_path, config_dic)
    return config_dic


def update_args_with_values_from_config_file(args, config_path="peakipy.config"):
    """read a peakipy config file, extract params and update args dict

    :param args: dict containing params extracted from docopt command line
    :type args: dict
    :param config_path: path to peakipy config file [default: peakipy.config]
    :type config_path: str

    :returns args: updated args dict
    :rtype args: dict
    :returns config: dict that resulted from reading config file
    :rtype config: dict

    """
    # update args with values from peakipy.config file
    config_path = Path(config_path)
    if config_path.exists():
        try:
            config = load_config(config_path)
            print(
                f"[green]Using config file with dims [yellow]{config.get('dims')}[/yellow][/green]"
            )
            args["dims"] = config.get("dims", (0, 1, 2))
            noise = config.get("noise")
            if noise:
                noise = float(noise)

            colors = config.get("colors", ["#5e3c99", "#e66101"])
        except json.decoder.JSONDecodeError:
            print(
                "[red]Your peakipy.config file is corrupted - maybe your JSON is not correct...[/red]"
            )
            print("[red]Not using[/red]")
            noise = False
            colors = args.get("colors", ("#5e3c99", "#e66101"))
            config = {}
    else:
        print(
            "[red]No peakipy.config found - maybe you need to generate one with peakipy read or see docs[/red]"
        )
        noise = False
        colors = args.get("colors", ("#5e3c99", "#e66101"))
        config = {}

    args["noise"] = noise
    args["colors"] = colors

    return args, config


def update_linewidths_from_hz_to_points(peakipy_data):
    """in case they were adjusted when running edit.py"""
    peakipy_data.df["XW"] = peakipy_data.df.XW_HZ * peakipy_data.pt_per_hz_f2
    peakipy_data.df["YW"] = peakipy_data.df.YW_HZ * peakipy_data.pt_per_hz_f1
    return peakipy_data


def update_peak_positions_from_ppm_to_points(peakipy_data):
    # convert peak positions from ppm to points in case they were adjusted running edit.py
    peakipy_data.df["X_AXIS"] = peakipy_data.df.X_PPM.apply(
        lambda x: peakipy_data.uc_f2(x, "PPM")
    )
    peakipy_data.df["Y_AXIS"] = peakipy_data.df.Y_PPM.apply(
        lambda x: peakipy_data.uc_f1(x, "PPM")
    )
    peakipy_data.df["X_AXISf"] = peakipy_data.df.X_PPM.apply(
        lambda x: peakipy_data.uc_f2.f(x, "PPM")
    )
    peakipy_data.df["Y_AXISf"] = peakipy_data.df.Y_PPM.apply(
        lambda x: peakipy_data.uc_f1.f(x, "PPM")
    )
    return peakipy_data


def check_for_existing_output_file_and_backup(outname: Path):
    if outname.exists():
        shutil.copy(outname, outname.with_suffix(".bak"))
    else:
        pass
    return outname

def save_data(df, output_name):
    suffix = output_name.suffix

    if suffix == ".csv":
        df.to_csv(output_name, float_format="%.4f", index=False)

    elif suffix == ".tab":
        df.to_csv(output_name, sep="\t", float_format="%.4f", index=False)

    else:
        df.to_pickle(output_name)


def check_data_shape_is_consistent_with_dims(peakipy_data):
    # check data shape is consistent with dims
    if len(peakipy_data.dims) != len(peakipy_data.data.shape):
        print(
            f"Dims are {peakipy_data.dims} while data shape is {peakipy_data.data.shape}?"
        )
        exit()


def check_for_include_column_and_add_if_missing(peakipy_data):
    # only include peaks with 'include'
    if "include" in peakipy_data.df.columns:
        pass
    else:
        # for compatibility
        peakipy_data.df["include"] = peakipy_data.df.apply(lambda _: "yes", axis=1)
    return peakipy_data


def remove_excluded_peaks(peakipy_data):
    if len(peakipy_data.df[peakipy_data.df.include != "yes"]) > 0:
        excluded = peakipy_data.df[peakipy_data.df.include != "yes"][
            peaklist_columns_for_printing
        ]
        table = df_to_rich_table(
            excluded,
            title="[yellow] Excluded peaks [/yellow]",
            columns=excluded.columns,
            styles=["yellow" for i in excluded.columns],
        )
        print(table)
        peakipy_data.df = peakipy_data.df[peakipy_data.df.include == "yes"]
    return peakipy_data


def warn_if_trying_to_fit_large_clusters(max_cluster_size, peakipy_data):
    if max_cluster_size is None:
        max_cluster_size = peakipy_data.df.MEMCNT.max()
        if peakipy_data.df.MEMCNT.max() > 10:
            print(
                f"""[red]
                ##################################################################
                You have some clusters of as many as {max_cluster_size} peaks.
                You may want to consider reducing the size of your clusters as the
                fits will struggle.

                Otherwise you can use the --max-cluster-size flag to exclude large
                clusters
                ##################################################################
            [/red]"""
            )
    else:
        max_cluster_size = max_cluster_size
    return max_cluster_size
