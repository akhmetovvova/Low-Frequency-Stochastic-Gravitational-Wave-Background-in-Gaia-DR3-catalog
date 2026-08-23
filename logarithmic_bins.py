#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
running_quantiles.py

Statistical analysis of Monte Carlo injection-recovery simulations
for the detection of a low-frequency stochastic gravitational-wave
background using astrometric proper motions of Gaia DR3 quasars.

The program analyses the recovered GW amplitudes obtained from a set
of Monte Carlo realizations and calculates their statistical
properties as a function of the injected GW amplitude.

Input:
    CSV file containing three columns:

        injected,recovered,error

    where

        injected  - injected GW amplitude (strain),
        recovered - recovered GW amplitude from the analysis,
        error     - formal uncertainty of the recovered amplitude.

    The input file does not require a header.

Method:
    The injected GW amplitudes are divided into logarithmically
    spaced bins. For each bin the program calculates:

        - number of Monte Carlo realizations;
        - median recovered amplitude;
        - 2.5th and 97.5th percentiles
          (approximately the central 95% interval);
        - 15th and 85th percentiles
          (central 70% interval);
        - median formal uncertainty of the recovered amplitude;
        - median signal-to-noise ratio Z = recovered/error;
        - fraction of realizations with recovered/error > 2;
        - fraction of realizations with recovered/error > 3.

    The median recovered amplitude and the percentile intervals
    characterize the bias and statistical scatter of the recovered
    GW amplitude for each injected amplitude.

    The fractions

        f_2sigma = N(recovered > 2 * error) / N
        f_3sigma = N(recovered > 3 * error) / N

    provide estimates of the detection probability at 2-sigma
    and 3-sigma significance levels.

Output:
    A space-separated ASCII file containing:

        inj
        N
        median
        low2sig
        up2sig
        q15
        q85
        err_median
        Z_median
        frac2
        frac3

    where:

        inj        - logarithmic bin centre of injected GW amplitude;
        N          - number of MC realizations in the bin;
        median     - median recovered amplitude;
        low2sig    - 2.5th percentile of recovered amplitudes;
        up2sig     - 97.5th percentile of recovered amplitudes;
        q15        - 15th percentile;
        q85        - 85th percentile;
        err_median - median formal uncertainty;
        Z_median   - median recovered signal-to-noise ratio;
        frac2      - fraction of realizations detected above 2-sigma;
        frac3      - fraction of realizations detected above 3-sigma.

Usage:
    python3 running_quantiles.py input.csv output.dat

Example:
    python3 running_quantiles.py mc_results.csv mc_statistics.dat

The resulting file can be used to construct detection-efficiency
curves, recovered-versus-injected amplitude plots, confidence
intervals, and Monte Carlo detection probability curves for the
Gaia DR3 astrometric gravitational-wave analysis.

Author:
    Volodymyr Akhmetov
"""

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
