# Fixes the 2kHz tracks by correcting the time columns.

import pandas as pd
import numpy as np

fname = "180912 20nm GoldNP on 75DOPC25Chol (2kHz 100pc power)_Position_converted"

df = pd.read_csv(fname+".csv")

ids =df["id"].unique()

for id in ids:
    df_track = df.loc[df["id"] == id]
    t = np.arange(0.0000, 0.0005 * df_track["t"].size, 0.0005)
    df["t"].loc[df["id"] == id] = t

df.to_csv(fname+"_fixed.csv", index=False, float_format='%.10f')