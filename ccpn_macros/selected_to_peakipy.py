""" Export selected peaks to TSV file """
import os
from pathlib import Path
import pandas as pd


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


path = Path(os.getenv("HOME")) / "tmp"
path.mkdir(exist_ok=True)
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
