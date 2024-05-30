""" Export selected peaks to TSV file """

import os
from pathlib import Path
import pandas as pd


import peakipy.cli.main.read
import peakipy.cli.main.fit
import peakipy.cli.main.check

# set temp path
path = Path(os.getenv("HOME")) / ".peakipy"
path.mkdir(exist_ok=True)
# spectrum
current_spectrum = "SP:test_protein_L"
current_spectrum_path = Path(get(current_spectrum).path)
# change to directory containing spectrum
os.chdir(current_spectrum_path.resolve().parent)


def PeakToTableRow(peak):
    dic = dict(
        Pid=peak.serial,
        Spectrum=peak.spectrum,
        PeakList=peak.peakList,
        Id=peak.id,
        Height=peak.height,
        HeightError=peak.heightError,
        Volume=peak.volume,
        VolumeError=peak.volumeError,
        Merit=peak.figureOfMerit,
        Annotation=peak.annotation,
        Comment=peak.comment,
    )
    for num, i in enumerate(peak.axisCodes):
        dim = num + 1
        dic[f"Assign F{dim}"] = peak.assignmentsByDimensions[num]
        dic[f"Pos F{dim}"] = peak.position[num]
        dic[f"LW F{dim} (Hz)"] = peak.lineWidths[num]
        dic[f"Pos F{dim}"] = peak.position[num]
    return dic


def PeaksToDataFrame(peaks):
    dic_list = []
    for peak in peaks:
        dic_list.append(PeakToTableRow(peak))
    return pd.DataFrame(dic_list)


def peakipy_read(path=path):
    # argv = [f"{path}/test.tsv", f"{current_spectrum_path}", "--a3"]
    args = dict(
        peaklist_path=path / "test.tsv",
        data_path=current_spectrum_path,
        peaklist_format="a3",
    )
    peakipy.cli.main.read(**args)


def peakipy_fit():
    out_path = current_spectrum_path.parent
    args = dict(
        peaklist_path=out_path / "test.csv",
        data_path=current_spectrum_path,
        output_path=out_path / "fits.csv",
    )
    peakipy.cli.main.fit(**args)


def peakipy_check():
    # plt = None
    out_path = current_spectrum_path.parent
    argv = [
        f"{out_path / 'fits.csv'}",
        f"{current_spectrum_path}",
        "--show",
        "--first",
        "--label",
        "--individual",
        "--ccpn",
    ]
    peakipy.commandline.check.main(argv)


if __name__ == "__main__":
    peaks = current.peaks
    df = PeaksToDataFrame(peaks)
    column_order = [
        "Pid",
        "Spectrum",
        "PeakList",
        "Id",
        "Assign F1",
        "Assign F2",
        "Pos F1",
        "Pos F2",
        "LW F1 (Hz)",
        "LW F2 (Hz)",
        "Height",
        "HeightError",
        "Volume",
        "VolumeError",
        "Merit",
        "Annotation",
        "Comment",
    ]
    df = df[column_order]
    out = path / "test.tsv"
    print(f"Saving selected peaks to {out}.")
    df.to_csv(out, sep="\t")
    print(df)
    peakipy_read()
    peakipy_fit()
    peakipy_check()
    # plt.show()
