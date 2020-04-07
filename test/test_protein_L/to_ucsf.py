import pandas as pd

df = pd.read_csv("test.csv")

with open("peaks.sparky", "w") as f:

    headers = "Assignment       w1      w2      Volume     lw1 (hz)   lw2 (hz)"
    f.write(headers + "\n\n")
    for ind, row in df.iterrows():
        f.write(
            f"{row.ASS:>12s}\t{row.Y_PPM:.3f}\t{row.X_PPM:.3f}\t{row.VOL:3.2e}\t{row.YW:.1f}\t{row.XW:.1f}"
            + "\n"
        )
