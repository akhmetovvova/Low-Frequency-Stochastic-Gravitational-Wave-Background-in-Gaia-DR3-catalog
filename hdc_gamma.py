#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hdc_gamma.py

Hellings-Downs / astrometric angular correlation function
from quasar proper motions.

The code calculates four correlation components:

    Gamma_parallel_parallel
    Gamma_perpendicular_perpendicular
    Gamma_parallel_perpendicular
    Gamma_perpendicular_parallel

using the full 2x2 covariance matrix of every quasar.

Input CSV:

    ra
    dec
    pmra
    pmdec
    pmra_error
    pmdec_error
    pmra_pmdec_corr

Units:

    ra, dec       : degrees
    proper motion : mas/yr
    errors        : mas/yr
    correlation   : dimensionless

Output:

    theta_deg
    Gamma_pp
    Gamma_pp_err
    Gamma_pt
    Gamma_pt_err
    Gamma_tp
    Gamma_tp_err
    Gamma_tt
    Gamma_tt_err
    N_pairs

where

    p = parallel
    t = perpendicular

The implementation does not construct all N(N-1)/2 pairs
simultaneously.

For large Gaia catalogues HEALPix is used to reduce the
number of candidate pairs.

Author:
    Volodymyr Akhmetov
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

try:
    import healpy as hp
except ImportError:
    hp = None

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ============================================================
# Constants
# ============================================================

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


# ============================================================
# Catalogue
# ============================================================

def load_catalogue(filename):
    """
    Read Gaia QSO catalogue.
    """

    print(f"Reading: {filename}")

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )

    names = data.dtype.names

    def get_column(candidates):

        for name in candidates:
            if name in names:
                return np.asarray(
                    data[name],
                    dtype=np.float64
                )

        return None

    ra = get_column([
        "ra",
        "RA",
        "ra_deg"
    ])

    dec = get_column([
        "dec",
        "DEC",
        "dec_deg"
    ])

    pmra = get_column([
        "pmra",
        "pmra_obs",
        "mu_alpha"
    ])

    pmdec = get_column([
        "pmdec",
        "pmdec_obs",
        "mu_delta"
    ])

    sig_ra = get_column([
        "pmra_error",
        "sig_pmra",
        "pmra_err"
    ])

    sig_dec = get_column([
        "pmdec_error",
        "sig_pmdec",
        "pmdec_err"
    ])

    corr = get_column([
        "pmra_pmdec_corr",
        "corr_pmra_pmdec",
        "corr"
    ])

    if ra is None:
        raise ValueError("RA column not found.")

    if dec is None:
        raise ValueError("DEC column not found.")

    if pmra is None:
        raise ValueError("pmra column not found.")

    if pmdec is None:
        raise ValueError("pmdec column not found.")

    if sig_ra is None:
        raise ValueError(
            "pmra_error/sig_pmra column not found."
        )

    if sig_dec is None:
        raise ValueError(
            "pmdec_error/sig_pmdec column not found."
        )

    if corr is None:
        raise ValueError(
            "pmra_pmdec_corr column not found."
        )

    return (
        ra,
        dec,
        pmra,
        pmdec,
        sig_ra,
        sig_dec,
        corr
    )


# ============================================================
# Sky vectors
# ============================================================

def radec_to_vectors(ra_deg, dec_deg):
    """
    Convert RA/DEC to Cartesian unit vectors.
    """

    ra = ra_deg * DEG2RAD
    dec = dec_deg * DEG2RAD

    c = np.cos(dec)

    n = np.empty(
        (len(ra), 3),
        dtype=np.float64
    )

    n[:, 0] = c * np.cos(ra)
    n[:, 1] = c * np.sin(ra)
    n[:, 2] = np.sin(dec)

    return n


def tangent_basis(ra_deg, dec_deg):
    """
    e_alpha and e_delta basis.
    """

    ra = ra_deg * DEG2RAD
    dec = dec_deg * DEG2RAD

    sa = np.sin(ra)
    ca = np.cos(ra)

    sd = np.sin(dec)
    cd = np.cos(dec)

    e_ra = np.empty(
        (len(ra), 3),
        dtype=np.float64
    )

    e_dec = np.empty(
        (len(ra), 3),
        dtype=np.float64
    )

    e_ra[:, 0] = -sa
    e_ra[:, 1] = ca
    e_ra[:, 2] = 0.0

    e_dec[:, 0] = -sd * ca
    e_dec[:, 1] = -sd * sa
    e_dec[:, 2] = cd

    return e_ra, e_dec


# ============================================================
# Pair geometry
# ============================================================

def pair_geometry(
        ni,
        nj,
        e_ra_i,
        e_dec_i,
        e_ra_j,
        e_dec_j
):
    """
    Construct local parallel/perpendicular bases.

    For the great circle connecting i and j:

        t_i = direction from i toward j
        t_j = direction from j toward i

    The perpendicular basis is

        p_i = n_i x t_i
        p_j = n_j x t_j

    """

    dot = np.dot(
        ni,
        nj
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0
    )

    theta = np.arccos(dot)

    s = np.sin(theta)

    if abs(s) < 1.0e-12:

        return None

    # --------------------------------------------------------
    # Direction at i toward j
    # --------------------------------------------------------

    t_i = (
        nj
        - dot * ni
    ) / s

    # Direction at j toward i
    t_j = (
        ni
        - dot * nj
    ) / s

    # --------------------------------------------------------
    # Perpendicular directions
    # --------------------------------------------------------

    p_i = np.cross(
        ni,
        t_i
    )

    p_j = np.cross(
        nj,
        t_j
    )

    p_i /= np.linalg.norm(p_i)
    p_j /= np.linalg.norm(p_j)

    # --------------------------------------------------------
    # Components in RA/DEC basis
    # --------------------------------------------------------

    # Parallel basis at i
    epar_i_ra = np.dot(
        t_i,
        e_ra_i
    )

    epar_i_dec = np.dot(
        t_i,
        e_dec_i
    )

    # Parallel basis at j
    epar_j_ra = np.dot(
        t_j,
        e_ra_j
    )

    epar_j_dec = np.dot(
        t_j,
        e_dec_j
    )

    # Perpendicular basis at i
    eperp_i_ra = np.dot(
        p_i,
        e_ra_i
    )

    eperp_i_dec = np.dot(
        p_i,
        e_dec_i
    )

    # Perpendicular basis at j
    eperp_j_ra = np.dot(
        p_j,
        e_ra_j
    )

    eperp_j_dec = np.dot(
        p_j,
        e_dec_j
    )

    return (
        theta,

        epar_i_ra,
        epar_i_dec,

        eperp_i_ra,
        eperp_i_dec,

        epar_j_ra,
        epar_j_dec,

        eperp_j_ra,
        eperp_j_dec
    )


# ============================================================
# Project covariance matrix
# ============================================================

def project_covariance(
        sig_ra,
        sig_dec,
        corr,
        ex,
        ey
):
    """
    Project full 2x2 covariance matrix onto an arbitrary
    tangent direction e=(ex,ey).

    C = [[sra^2, rho*sra*sdec],
         [rho*sra*sdec, sdec^2]]

    sigma_e^2 = e^T C e
    """

    c = (
        corr
        * sig_ra
        * sig_dec
    )

    variance = (
        ex * ex * sig_ra * sig_ra
        +
        ey * ey * sig_dec * sig_dec
        +
        2.0 * ex * ey * c
    )

    return max(
        variance,
        0.0
    )


# ============================================================
# Pair-product variance
# ============================================================

def product_variance(
        mu_a_i,
        mu_b_j,
        var_a_i,
        var_b_j
):
    """
    Variance of product X*Y for independent Gaussian X,Y.

    Var(XY) =
        mu_X^2 Var(Y)
        +
        mu_Y^2 Var(X)
        +
        Var(X) Var(Y)

    """

    return (
        mu_a_i * mu_a_i * var_b_j
        +
        mu_b_j * mu_b_j * var_a_i
        +
        var_a_i * var_b_j
    )


# ============================================================
# Bin accumulation
# ============================================================

class GammaAccumulator:

    def __init__(
            self,
            n_bins
    ):

        self.sum = np.zeros(
            (4, n_bins),
            dtype=np.float64
        )

        self.weight = np.zeros(
            (4, n_bins),
            dtype=np.float64
        )

        self.sum_sq = np.zeros(
            (4, n_bins),
            dtype=np.float64
        )

        self.n_pairs = np.zeros(
            n_bins,
            dtype=np.int64
        )

    def add(
            self,
            b,
            values,
            variances
    ):

        for k in range(4):

            value = values[k]
            variance = variances[k]

            if (
                not np.isfinite(value)
                or not np.isfinite(variance)
                or variance <= 0.0
            ):
                continue

            w = 1.0 / variance

            self.sum[k, b] += (
                w * value
            )

            self.weight[k, b] += w

            self.sum_sq[k, b] += (
                w * value * value
            )

        self.n_pairs[b] += 1

    def result(self):

        gamma = np.full_like(
            self.sum,
            np.nan
        )

        error = np.full_like(
            self.sum,
            np.nan
        )

        for k in range(4):

            mask = (
                self.weight[k] > 0
            )

            gamma[k, mask] = (
                self.sum[k, mask]
                / self.weight[k, mask]
            )

            # ------------------------------------------------
            # Effective variance of weighted mean.
            #
            # This is primarily a diagnostic estimate.
            # For independent pair products:
            #
            # Var(mean) ~ 1/sum(w)
            #
            # when w = 1/Var(pair).
            # ------------------------------------------------

            error[k, mask] = np.sqrt(
                1.0
                / self.weight[k, mask]
            )

        return (
            gamma,
            error,
            self.n_pairs
        )


# ============================================================
# Pair processing
# ============================================================

def process_pairs(
        indices_i,
        indices_j,
        nvec,
        e_ra,
        e_dec,
        pmra,
        pmdec,
        sig_ra,
        sig_dec,
        corr,
        theta_min,
        theta_max,
        bin_width,
        accumulator
):
    """
    Process a block of pairs.

    No full N(N-1)/2 array is constructed.
    """

    for ii, jj in zip(
        indices_i,
        indices_j
    ):

        ni = nvec[ii]
        nj = nvec[jj]

        geometry = pair_geometry(
            ni,
            nj,
            e_ra[ii],
            e_dec[ii],
            e_ra[jj],
            e_dec[jj]
        )

        if geometry is None:
            continue

        (
            theta,

            epar_i_ra,
            epar_i_dec,

            eperp_i_ra,
            eperp_i_dec,

            epar_j_ra,
            epar_j_dec,

            eperp_j_ra,
            eperp_j_dec
        ) = geometry

        if (
            theta < theta_min
            or theta >= theta_max
        ):
            continue

        b = int(
            (theta - theta_min)
            / bin_width
        )

        if (
            b < 0
            or b >= len(
                accumulator.n_pairs
            )
        ):
            continue

        # ----------------------------------------------------
        # Project proper motions
        # ----------------------------------------------------

        mu_par_i = (
            pmra[ii] * epar_i_ra
            +
            pmdec[ii] * epar_i_dec
        )

        mu_perp_i = (
            pmra[ii] * eperp_i_ra
            +
            pmdec[ii] * eperp_i_dec
        )

        mu_par_j = (
            pmra[jj] * epar_j_ra
            +
            pmdec[jj] * epar_j_dec
        )

        mu_perp_j = (
            pmra[jj] * eperp_j_ra
            +
            pmdec[jj] * eperp_j_dec
        )

        # ----------------------------------------------------
        # Project full covariance
        # ----------------------------------------------------

        var_par_i = project_covariance(
            sig_ra[ii],
            sig_dec[ii],
            corr[ii],
            epar_i_ra,
            epar_i_dec
        )

        var_perp_i = project_covariance(
            sig_ra[ii],
            sig_dec[ii],
            corr[ii],
            eperp_i_ra,
            eperp_i_dec
        )

        var_par_j = project_covariance(
            sig_ra[jj],
            sig_dec[jj],
            corr[jj],
            epar_j_ra,
            epar_j_dec
        )

        var_perp_j = project_covariance(
            sig_ra[jj],
            sig_dec[jj],
            corr[jj],
            eperp_j_ra,
            eperp_j_dec
        )

        # ----------------------------------------------------
        # Four correlation components
        # ----------------------------------------------------

        c_pp = (
            mu_par_i
            * mu_par_j
        )

        c_pt = (
            mu_par_i
            * mu_perp_j
        )

        c_tp = (
            mu_perp_i
            * mu_par_j
        )

        c_tt = (
            mu_perp_i
            * mu_perp_j
        )

        # ----------------------------------------------------
        # Variances of pair products
        # ----------------------------------------------------

        v_pp = product_variance(
            mu_par_i,
            mu_par_j,
            var_par_i,
            var_par_j
        )

        v_pt = product_variance(
            mu_par_i,
            mu_perp_j,
            var_par_i,
            var_perp_j
        )

        v_tp = product_variance(
            mu_perp_i,
            mu_par_j,
            var_perp_i,
            var_par_j
        )

        v_tt = product_variance(
            mu_perp_i,
            mu_perp_j,
            var_perp_i,
            var_perp_j
        )

        accumulator.add(
            b,
            [
                c_pp,
                c_pt,
                c_tp,
                c_tt
            ],
            [
                v_pp,
                v_pt,
                v_tp,
                v_tt
            ]
        )


# ============================================================
# HEALPix pair generation
# ============================================================

def generate_pairs_healpix(
        nvec,
        nside,
        theta_min,
        theta_max
):
    """
    Generate candidate pairs using HEALPix.

    IMPORTANT:

    This routine generates pairs by pixel neighborhoods.
    The final angular separation cut is still applied later.

    It avoids constructing the full N(N-1)/2 array.
    """

    if hp is None:

        raise ImportError(
            "healpy is required for HEALPix mode.\n"
            "Install with:\n"
            "    pip install healpy"
        )

    npix = hp.nside2npix(
        nside
    )

    # --------------------------------------------------------
    # Pixel index of every source
    # --------------------------------------------------------

    theta = np.arccos(
        np.clip(
            nvec[:, 2],
            -1.0,
            1.0
        )
    )

    phi = np.arctan2(
        nvec[:, 1],
        nvec[:, 0]
    )

    phi[phi < 0.0] += 2.0 * np.pi

    pix = hp.ang2pix(
        nside,
        theta,
        phi
    )

    # --------------------------------------------------------
    # Sources in each pixel
    # --------------------------------------------------------

    order = np.argsort(
        pix
    )

    pix_sorted = pix[order]

    unique_pix, first, counts = np.unique(
        pix_sorted,
        return_index=True,
        return_counts=True
    )

    pixel_sources = {}

    for p, start, count in zip(
        unique_pix,
        first,
        counts
    ):

        pixel_sources[int(p)] = (
            order[start:start + count]
        )

    # --------------------------------------------------------
    # Maximum angular radius needed.
    #
    # Query disc radius:
    #
    # theta_max + pixel diagonal margin
    # --------------------------------------------------------

    pixel_radius = (
        np.pi / nside
    )

    query_radius = (
        theta_max
        + 2.0 * pixel_radius
    )

    processed_pixel_pairs = set()

    for p in unique_pix:

        p = int(p)

        vec_p = hp.pix2vec(
            nside,
            p
        )

        neighbours = hp.query_disc(
            nside,
            vec_p,
            query_radius,
            inclusive=True
        )

        sources_p = pixel_sources[p]

        for q in neighbours:

            q = int(q)

            if q not in pixel_sources:
                continue

            # Avoid processing pixel pair twice.
            pair_key = (
                min(p, q),
                max(p, q)
            )

            if pair_key in processed_pixel_pairs:
                continue

            processed_pixel_pairs.add(
                pair_key
            )

            sources_q = pixel_sources[q]

            yield (
                sources_p,
                sources_q
            )


# ============================================================
# Main HDC calculation
# ============================================================

def calculate_gamma(args):

    (
        ra,
        dec,
        pmra,
        pmdec,
        sig_ra,
        sig_dec,
        corr
    ) = load_catalogue(
        args.input
    )

    n = len(ra)

    print()
    print(
        f"Number of QSOs: {n:,}"
    )

    # --------------------------------------------------------
    # Sky vectors
    # --------------------------------------------------------

    print(
        "Constructing sky vectors..."
    )

    nvec = radec_to_vectors(
        ra,
        dec
    )

    e_ra, e_dec = tangent_basis(
        ra,
        dec
    )

    # --------------------------------------------------------
    # Angular bins
    # --------------------------------------------------------

    theta_min = (
        args.theta_min
        * DEG2RAD
    )

    theta_max = (
        args.theta_max
        * DEG2RAD
    )

    bin_width = (
        args.bin_width
        * DEG2RAD
    )

    n_bins = int(
        np.ceil(
            (
                theta_max
                - theta_min
            )
            / bin_width
        )
    )

    print()
    print(
        f"Theta range : "
        f"{args.theta_min} - "
        f"{args.theta_max} deg"
    )

    print(
        f"Bin width    : "
        f"{args.bin_width} deg"
    )

    print(
        f"Number bins  : "
        f"{n_bins}"
    )

    accumulator = GammaAccumulator(
        n_bins
    )

    # --------------------------------------------------------
    # Pair processing
    # --------------------------------------------------------

    t0 = time.time()

    if args.healpix:

        print()
        print(
            f"Using HEALPix nside={args.nside}"
        )

        pair_generator = (
            generate_pairs_healpix(
                nvec,
                args.nside,
                theta_min,
                theta_max
            )
        )

        npixel_pairs = 0

        for sources_i, sources_j in pair_generator:

            npixel_pairs += 1

            # ------------------------------------------------
            # Same pixel:
            #
            # use only i<j
            # ------------------------------------------------

            if (
                len(sources_i) == len(sources_j)
                and np.array_equal(
                    sources_i,
                    sources_j
                )
            ):

                if len(sources_i) < 2:
                    continue

                ii, jj = np.triu_indices(
                    len(sources_i),
                    k=1
                )

                idx_i = sources_i[ii]
                idx_j = sources_i[jj]

            else:

                # --------------------------------------------
                # Cross-pixel pairs
                # --------------------------------------------

                idx_i = np.repeat(
                    sources_i,
                    len(sources_j)
                )

                idx_j = np.tile(
                    sources_j,
                    len(sources_i)
                )

                # ------------------------------------------------
                # To guarantee unique pairs:
                #
                # source indices must be ordered.
                # ------------------------------------------------

                mask = (
                    idx_i < idx_j
                )

                idx_i = idx_i[mask]
                idx_j = idx_j[mask]

            if len(idx_i) == 0:
                continue

            # ------------------------------------------------
            # Process in blocks
            # ------------------------------------------------

            block = args.pair_block

            for start in range(
                0,
                len(idx_i),
                block
            ):

                stop = min(
                    start + block,
                    len(idx_i)
                )

                process_pairs(
                    idx_i[start:stop],
                    idx_j[start:stop],
                    nvec,
                    e_ra,
                    e_dec,
                    pmra,
                    pmdec,
                    sig_ra,
                    sig_dec,
                    corr,
                    theta_min,
                    theta_max,
                    bin_width,
                    accumulator
                )

            if (
                npixel_pairs % args.progress == 0
            ):

                elapsed = (
                    time.time()
                    - t0
                )

                print(
                    f"Processed pixel pairs: "
                    f"{npixel_pairs:,} | "
                    f"time={elapsed/60.0:.2f} min"
                )

    else:

        # ----------------------------------------------------
        # Brute-force mode.
        #
        # Intended only for small catalogues / testing.
        # ----------------------------------------------------

        print()
        print(
            "WARNING:"
        )

        print(
            "Running brute-force pair calculation."
        )

        print(
            "This mode is NOT recommended for "
            "1.5 million QSOs."
        )

        block = args.pair_block

        for i_start in range(
            0,
            n - 1,
            block
        ):

            i_stop = min(
                i_start + block,
                n - 1
            )

            for i in range(
                i_start,
                i_stop
            ):

                idx_i = np.full(
                    n - i - 1,
                    i,
                    dtype=np.int64
                )

                idx_j = np.arange(
                    i + 1,
                    n,
                    dtype=np.int64
                )

                process_pairs(
                    idx_i,
                    idx_j,
                    nvec,
                    e_ra,
                    e_dec,
                    pmra,
                    pmdec,
                    sig_ra,
                    sig_dec,
                    corr,
                    theta_min,
                    theta_max,
                    bin_width,
                    accumulator
                )

            if (
                i_start % (
                    block * args.progress
                ) == 0
            ):

                print(
                    f"Processed source "
                    f"{i_start:,}/{n:,}"
                )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    gamma, error, n_pairs = (
        accumulator.result()
    )

    theta_centers = (
        theta_min
        +
        (
            np.arange(n_bins)
            + 0.5
        )
        * bin_width
    )

    theta_deg = (
        theta_centers
        * RAD2DEG
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = np.column_stack([
        theta_deg,

        gamma[0],
        error[0],

        gamma[1],
        error[1],

        gamma[2],
        error[2],

        gamma[3],
        error[3],

        n_pairs
    ])

    header = (
        "theta_deg,"
        "Gamma_pp,Gamma_pp_err,"
        "Gamma_pt,Gamma_pt_err,"
        "Gamma_tp,Gamma_tp_err,"
        "Gamma_tt,Gamma_tt_err,"
        "N_pairs"
    )

    np.savetxt(
        args.output,
        output,
        delimiter=",",
        header=header,
        comments="",
        fmt=[
            "%.8f",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%d"
        ]
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - t0
    )

    print()
    print(
        "=" * 72
    )

    print(
        "HDC calculation completed."
    )

    print(
        "=" * 72
    )

    print(
        f"Time: {elapsed/60.0:.2f} min"
    )

    print()
    print(
        f"{'theta':>10} "
        f"{'Gamma_pp':>16} "
        f"{'Gamma_tt':>16} "
        f"{'Npairs':>12}"
    )

    for i in range(n_bins):

        if n_pairs[i] == 0:
            continue

        print(
            f"{theta_deg[i]:10.3f} "
            f"{gamma[0,i]:16.6e} "
            f"{gamma[3,i]:16.6e} "
            f"{n_pairs[i]:12d}"
        )

    print()
    print(
        f"Output: {args.output}"
    )


# ============================================================
# Argument parser
# ============================================================

def create_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Hellings-Downs angular correlation "
            "function using full QSO proper-motion covariance."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input Gaia QSO CSV."
    )

    parser.add_argument(
        "--output",
        default="gamma_hdc.csv",
        help="Output Gamma file."
    )

    # --------------------------------------------------------
    # Angular bins
    # --------------------------------------------------------

    parser.add_argument(
        "--theta-min",
        type=float,
        default=0.0,
        help="Minimum angular separation [deg]."
    )

    parser.add_argument(
        "--theta-max",
        type=float,
        default=180.0,
        help="Maximum angular separation [deg]."
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=5.0,
        help="Angular bin width [deg]."
    )

    # --------------------------------------------------------
    # HEALPix
    # --------------------------------------------------------

    parser.add_argument(
        "--healpix",
        action="store_true",
        help="Use HEALPix pair search."
    )

    parser.add_argument(
        "--nside",
        type=int,
        default=32,
        help="HEALPix NSIDE."
    )

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    parser.add_argument(
        "--pair-block",
        type=int,
        default=100000,
        help="Maximum number of pairs processed at once."
    )

    parser.add_argument(
        "--progress",
        type=int,
        default=100,
        help="Print progress every N pixel-pair groups."
    )

    return parser


# ============================================================
# Main
# ============================================================

def main():

    parser = create_parser()

    args = parser.parse_args()

    if args.bin_width <= 0:
        raise ValueError(
            "bin-width must be positive."
        )

    if args.theta_max <= args.theta_min:
        raise ValueError(
            "theta-max must be larger than theta-min."
        )

    if args.healpix and hp is None:
        raise ImportError(
            "healpy is required.\n"
            "Install it using:\n"
            "pip install healpy"
        )

    calculate_gamma(args)


if __name__ == "__main__":
    main()
