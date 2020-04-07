from pathlib import Path
import pandas as pd
import numpy as np


def split_peaklist(peaklist, n_cpu):
    tmp = Path("./tmp/")
    tmp.mkdir(exist_ok=True)
    clustids = peaklist.CLUSTID.unique()
    window = int(np.ceil(len(clustids) / n_cpu))
    clustids = [clustids[i : i + window] for i in range(0, len(clustids), window)]
    for i in range(n_cpu):
        split_peaks = peaklist[peaklist.CLUSTID.isin(clustids[i])]
        split_peaks.to_csv(tmp / f"peaks_{i}.csv")


peaklist = pd.read_csv("edited_peaks.csv")
split_peaklist(peaklist, 4)
