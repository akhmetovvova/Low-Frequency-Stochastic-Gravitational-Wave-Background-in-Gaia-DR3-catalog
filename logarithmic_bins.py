#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

# -------------------------------------------------
# command line arguments
# -------------------------------------------------

if len(sys.argv) != 3:
    print("Usage:")
    print("python3 running_quantiles.py input.csv output.dat")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

print("Input :", input_file)
print("Output:", output_file)

# -------------------------------------------------
# read data
# columns:
# injected,recovered,error
# -------------------------------------------------

df = pd.read_csv(
    input_file,
    sep=",",
    names=["inj", "rec", "err"]
)

# convert strings -> float
for col in ["inj", "rec", "err"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# remove bad rows
df = df.dropna()

print(f"Rows read: {len(df)}")

# -------------------------------------------------
# logarithmic bins in injected amplitude
# -------------------------------------------------

nbins = 40

bins = np.logspace(
    np.log10(df["inj"].min()),
    np.log10(df["inj"].max()),
    nbins + 1
)

centers = np.sqrt(bins[:-1] * bins[1:])

# assign bin number
df["bin"] = np.digitize(df["inj"], bins)

# -------------------------------------------------
# statistics in each bin
# -------------------------------------------------

rows = []

for i in range(1, nbins + 1):

    g = df[df["bin"] == i]

    if len(g) < 5:
        continue

    rec = g["rec"].values
    err = g["err"].values

    median = np.median(rec)

    q15 = np.percentile(rec, 15)
    q85 = np.percentile(rec, 85)

    low2sig = np.percentile(rec, 2.5)
    up2sig  = np.percentile(rec, 97.5)

    err_median = np.median(err)

    # significance estimators
    frac2 = np.mean(rec > 2.0 * err)
    frac3 = np.mean(rec > 3.0 * err)

    # median S/N
    Z_median = np.median(rec / err)

    rows.append([ centers[i - 1], len(g), abs(median), low2sig, up2sig, q15, q85, err_median, Z_median, frac2,frac3])

# -------------------------------------------------
# save output
# -------------------------------------------------

out = pd.DataFrame(rows,
    columns=[ "inj", "N", "median", "low2sig", "up2sig", "q15", "q85", "err_median", "Z_median", "frac2", "frac3" ])

out.to_csv( output_file, sep=" ",index=False, float_format="%.6e")

print()
print(out.head())
print()
print("Saved:", output_file)