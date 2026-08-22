#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mc_gw_simulator.py

Monte-Carlo simulation of astrometric signatures of gravitational
waves in a Gaia DR3 QSO catalogue.

The program is designed for injection-recovery simulations.

For every MC realization:

    1. The QSO sky distribution is kept fixed.
    2. A random GW propagation direction is generated.
    3. A random GW phase is generated.
    4. A plane GW with a specified strain hc is injected.
    5. Correlated Gaussian Gaia-like noise is generated using
       the full 2x2 covariance matrix of every QSO.
    6. The observed proper-motion field is constructed.

For a monochromatic GW:

        h(t) = h_plus e_plus cos(phi)
             + h_cross e_cross sin(phi)

and

        mu_GW ~ 2*pi*f_GW*h_c.

Input catalogue
---------------

CSV file with at least:

    ra
    dec

and, preferably:

    sig_pmra
    sig_pmdec
    corr_pmra_pmdec

All angles in the catalogue are in degrees.

Proper-motion uncertainties are in mas/yr.

Output
------

MC summary:

    realization
    hc
    frequency_hz
    ra_gw_deg
    dec_gw_deg
    phase_rad
    psi_deg
    rms_pmra_gw
    rms_pmdec_gw
    rms_pmra_noise
    rms_pmdec_noise

Optionally, the complete simulated catalogue of every realization
can be written to individual CSV files.

Important
---------

The sky distribution of Gaia QSOs is NOT randomized between
realizations. Only the GW realization and measurement noise are
randomized.

This is the desired setup for an injection-recovery Monte-Carlo
experiment with the actual Gaia DR3 sky distribution.

Author:
    Volodymyr Akhmetov
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


# ============================================================
# Constants
# ============================================================

RAD_TO_MAS = 206264806.24709636

SECONDS_PER_YEAR = 365.25 * 86400.0

DEG_TO_RAD = np.pi / 180.0


# ============================================================
# Coordinate functions
# ============================================================

def radec_to_unit_vectors(ra_deg, dec_deg):
    """
    Convert RA/DEC to Cartesian unit vectors.

    Parameters
    ----------
    ra_deg : ndarray
        Right ascension [deg].

    dec_deg : ndarray
        Declination [deg].

    Returns
    -------
    n : ndarray, shape (N,3)
        Unit vectors.
    """

    ra = np.asarray(ra_deg) * DEG_TO_RAD
    dec = np.asarray(dec_deg) * DEG_TO_RAD

    cos_dec = np.cos(dec)

    n = np.empty(
        (len(ra), 3),
        dtype=np.float64
    )

    n[:, 0] = cos_dec * np.cos(ra)
    n[:, 1] = cos_dec * np.sin(ra)
    n[:, 2] = np.sin(dec)

    return n


def tangent_basis(ra_deg, dec_deg):
    """
    Construct tangent vectors e_alpha and e_delta.

    e_alpha corresponds to increasing alpha-star.

    e_delta corresponds to increasing declination.
    """

    ra = np.asarray(ra_deg) * DEG_TO_RAD
    dec = np.asarray(dec_deg) * DEG_TO_RAD

    e_alpha = np.empty(
        (len(ra), 3),
        dtype=np.float64
    )

    e_delta = np.empty(
        (len(ra), 3),
        dtype=np.float64
    )

    e_alpha[:, 0] = -np.sin(ra)
    e_alpha[:, 1] = np.cos(ra)
    e_alpha[:, 2] = 0.0

    e_delta[:, 0] = (
        -np.sin(dec) * np.cos(ra)
    )

    e_delta[:, 1] = (
        -np.sin(dec) * np.sin(ra)
    )

    e_delta[:, 2] = np.cos(dec)

    return e_alpha, e_delta


# ============================================================
# Random GW direction
# ============================================================

def random_gw_direction(rng):
    """
    Generate an isotropically distributed GW propagation direction.

    RA:

        alpha ~ U(0, 2*pi)

    Declination:

        sin(delta) ~ U(-1,1)

    This guarantees a uniform distribution on the sphere.
    """

    ra = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    sin_dec = rng.uniform(
        -1.0,
        1.0
    )

    dec = np.arcsin(sin_dec)

    return (
        ra / DEG_TO_RAD,
        dec / DEG_TO_RAD
    )


# ============================================================
# GW propagation vector
# ============================================================

def propagation_vector(
        ra_gw_deg,
        dec_gw_deg
):
    """
    GW propagation direction.
    """

    ra = ra_gw_deg * DEG_TO_RAD
    dec = dec_gw_deg * DEG_TO_RAD

    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])


# ============================================================
# Polarization basis
# ============================================================

def polarization_basis(
        ra_gw_deg,
        dec_gw_deg,
        psi_deg
):
    """
    Construct orthonormal polarization basis p and q.
    """

    ra = ra_gw_deg * DEG_TO_RAD
    dec = dec_gw_deg * DEG_TO_RAD
    psi = psi_deg * DEG_TO_RAD

    p0 = np.array([
        -np.sin(ra),
        np.cos(ra),
        0.0
    ])

    q0 = np.array([
        -np.sin(dec) * np.cos(ra),
        -np.sin(dec) * np.sin(ra),
        np.cos(dec)
    ])

    p = (
        np.cos(psi) * p0
        + np.sin(psi) * q0
    )

    q = (
        -np.sin(psi) * p0
        + np.cos(psi) * q0
    )

    return p, q


def polarization_tensors(p, q):
    """
    Construct plus and cross polarization tensors.
    """

    e_plus = (
        np.outer(p, p)
        - np.outer(q, q)
    )

    e_cross = (
        np.outer(p, q)
        + np.outer(q, p)
    )

    return e_plus, e_cross


# ============================================================
# Plane GW astrometric response
# ============================================================

def plane_gw_deflection(
        n,
        k,
        e_plus,
        e_cross,
        h_plus,
        h_cross
):
    """
    Astrometric deflection produced by a plane GW.

    delta n_i =
        1/2 *
        (n_i-k_i)/(1-k.n) *
        n_j n_k h_jk
        -
        1/2 h_ij n_j

    The expression is evaluated for all QSOs simultaneously.
    """

    h = (
        h_plus * e_plus
        +
        h_cross * e_cross
    )

    # n_i h_ij n_j
    hn = np.einsum(
        "ni,ij,nj->n",
        n,
        h,
        n
    )

    # h_ij n_j
    hn_vec = np.einsum(
        "ij,nj->ni",
        h,
        n
    )

    nk = np.dot(
        n,
        k
    )

    denominator = 1.0 - nk

    # Avoid numerical singularity for sources very close
    # to the GW propagation direction.
    denominator = np.maximum(
        denominator,
        1.0e-12
    )

    term1 = (
        0.5
        * (n - k)
        / denominator[:, None]
        * hn[:, None]
    )

    term2 = -0.5 * hn_vec

    return term1 + term2


def deflection_to_proper_motion(
        delta_n,
        e_alpha,
        e_delta,
        frequency_hz
):
    """
    Convert angular deflection to proper motion.

    For a monochromatic GW:

        d/dt cos(2*pi*f*t)
            =
        -2*pi*f*sin(2*pi*f*t)

    Therefore the proper-motion amplitude is proportional to

        2*pi*f*h.

    Output is in mas/yr.
    """

    omega = (
        2.0
        * np.pi
        * frequency_hz
    )

    delta_dot = omega * delta_n

    pmra_rad_s = np.sum(
        delta_dot * e_alpha,
        axis=1
    )

    pmdec_rad_s = np.sum(
        delta_dot * e_delta,
        axis=1
    )

    conversion = (
        RAD_TO_MAS
        * SECONDS_PER_YEAR
    )

    pmra = (
        pmra_rad_s
        * conversion
    )

    pmdec = (
        pmdec_rad_s
        * conversion
    )

    return pmra, pmdec


# ============================================================
# Single MC GW realization
# ============================================================

def generate_gw_realization(
        n,
        e_alpha,
        e_delta,
        hc,
        frequency_hz,
        ra_gw_deg,
        dec_gw_deg,
        phase,
        psi_deg,
        equal_polarization=True
):
    """
    Generate one GW realization.

    Parameters
    ----------
    hc :
        GW strain amplitude.

    frequency_hz :
        GW frequency [Hz].

    ra_gw_deg :
        Random GW RA.

    dec_gw_deg :
        Random GW DEC.

    phase :
        Random initial phase [rad].

    psi_deg :
        Polarization angle.

    equal_polarization :
        If True:

            h_plus = hc/sqrt(2)
            h_cross = hc/sqrt(2)

        This corresponds to equal + and x polarization
        amplitudes before applying the random phase.
    """

    k = propagation_vector(
        ra_gw_deg,
        dec_gw_deg
    )

    p, q = polarization_basis(
        ra_gw_deg,
        dec_gw_deg,
        psi_deg
    )

    e_plus, e_cross = polarization_tensors(
        p,
        q
    )

    if equal_polarization:

        h_plus_amp = (
            hc / np.sqrt(2.0)
        )

        h_cross_amp = (
            hc / np.sqrt(2.0)
        )

    else:

        h_plus_amp = hc
        h_cross_amp = hc

    # --------------------------------------------------------
    # Random phase
    #
    # Both polarization components have the same random
    # realization phase, shifted by 90 degrees.
    # --------------------------------------------------------

    h_plus = (
        h_plus_amp
        * np.cos(phase)
    )

    h_cross = (
        h_cross_amp
        * np.sin(phase)
    )

    delta_n = plane_gw_deflection(
        n,
        k,
        e_plus,
        e_cross,
        h_plus,
        h_cross
    )

    pmra_gw, pmdec_gw = (
        deflection_to_proper_motion(
            delta_n,
            e_alpha,
            e_delta,
            frequency_hz
        )
    )

    return pmra_gw, pmdec_gw


# ============================================================
# Correlated Gaia noise
# ============================================================

def generate_correlated_noise(
        sig_pmra,
        sig_pmdec,
        corr,
        rng
):
    """
    Generate correlated Gaussian noise for all QSOs.

    Covariance:

        C_i =
        [[sigma_ra^2,
          rho sigma_ra sigma_dec],

         [rho sigma_ra sigma_dec,
          sigma_dec^2]]

    Using Cholesky-equivalent representation:

        n_ra  = sigma_ra z1

        n_dec =
            sigma_dec *
            [rho z1 +
             sqrt(1-rho^2) z2]
    """

    n = len(sig_pmra)

    z1 = rng.normal(
        0.0,
        1.0,
        n
    )

    z2 = rng.normal(
        0.0,
        1.0,
        n
    )

    rho = np.clip(
        corr,
        -0.999999,
        0.999999
    )

    noise_ra = (
        sig_pmra
        * z1
    )

    noise_dec = (
        sig_pmdec
        * (
            rho * z1
            +
            np.sqrt(
                1.0 - rho**2
            ) * z2
        )
    )

    return noise_ra, noise_dec


# ============================================================
# Catalogue loading
# ============================================================

def find_column(
        names,
        candidates
):
    """
    Find the first matching column name.
    """

    for candidate in candidates:

        if candidate in names:

            return candidate

    return None


def load_catalogue(filename):
    """
    Load Gaia QSO catalogue.
    """

    print(
        f"Reading catalogue: {filename}"
    )

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )

    names = data.dtype.names

    ra_col = find_column(
        names,
        [
            "ra",
            "RA",
            "ra_deg"
        ]
    )

    dec_col = find_column(
        names,
        [
            "dec",
            "DEC",
            "dec_deg"
        ]
    )

    if ra_col is None:
        raise ValueError(
            "RA column not found."
        )

    if dec_col is None:
        raise ValueError(
            "DEC column not found."
        )

    ra = np.asarray(
        data[ra_col],
        dtype=np.float64
    )

    dec = np.asarray(
        data[dec_col],
        dtype=np.float64
    )

    sig_ra_col = find_column(
        names,
        [
            "sig_pmra",
            "pmra_error",
            "pmra_err"
        ]
    )

    sig_dec_col = find_column(
        names,
        [
            "sig_pmdec",
            "pmdec_error",
            "pmdec_err"
        ]
    )

    corr_col = find_column(
        names,
        [
            "corr_pmra_pmdec",
            "pmra_pmdec_corr",
            "corr"
        ]
    )

    if sig_ra_col is None:

        raise ValueError(
            "sig_pmra column not found."
        )

    if sig_dec_col is None:

        raise ValueError(
            "sig_pmdec column not found."
        )

    if corr_col is None:

        raise ValueError(
            "corr_pmra_pmdec column not found."
        )

    sig_pmra = np.asarray(
        data[sig_ra_col],
        dtype=np.float64
    )

    sig_pmdec = np.asarray(
        data[sig_dec_col],
        dtype=np.float64
    )

    corr = np.asarray(
        data[corr_col],
        dtype=np.float64
    )

    return (
        ra,
        dec,
        sig_pmra,
        sig_pmdec,
        corr
    )


# ============================================================
# Save one realization
# ============================================================

def save_realization(
        filename,
        ra,
        dec,
        pmra_gw,
        pmdec_gw,
        pmra_noise,
        pmdec_noise
):
    """
    Save one complete MC realization.
    """

    pmra_obs = (
        pmra_gw
        + pmra_noise
    )

    pmdec_obs = (
        pmdec_gw
        + pmdec_noise
    )

    arr = np.column_stack([
        ra,
        dec,
        pmra_gw,
        pmdec_gw,
        pmra_noise,
        pmdec_noise,
        pmra_obs,
        pmdec_obs
    ])

    header = (
        "ra,dec,"
        "pmra_gw,pmdec_gw,"
        "pmra_noise,pmdec_noise,"
        "pmra_obs,pmdec_obs"
    )

    np.savetxt(
        filename,
        arr,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.12e"
    )


# ============================================================
# Main MC simulation
# ============================================================

def run_mc(args):

    # --------------------------------------------------------
    # Load catalogue
    # --------------------------------------------------------

    (
        ra,
        dec,
        sig_pmra,
        sig_pmdec,
        corr
    ) = load_catalogue(
        args.input
    )

    n_qso = len(ra)

    print()
    print(
        f"Number of QSO: {n_qso:,}"
    )

    # --------------------------------------------------------
    # Coordinate preparation
    #
    # These quantities are calculated ONLY ONCE.
    #
    # This is important for 1000 MC realizations.
    # --------------------------------------------------------

    print(
        "Preparing sky coordinates..."
    )

    n = radec_to_unit_vectors(
        ra,
        dec
    )

    e_alpha, e_delta = tangent_basis(
        ra,
        dec
    )

    # --------------------------------------------------------
    # RNG
    # --------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    print(
        f"Master random seed: {args.seed}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    if args.save_catalogues:

        os.makedirs(
            args.output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Summary arrays
    # --------------------------------------------------------

    n_mc = args.n_mc

    realization_id = np.arange(
        n_mc,
        dtype=int
    )

    ra_gw_all = np.zeros(
        n_mc
    )

    dec_gw_all = np.zeros(
        n_mc
    )

    phase_all = np.zeros(
        n_mc
    )

    psi_all = np.zeros(
        n_mc
    )

    rms_pmra_gw = np.zeros(
        n_mc
    )

    rms_pmdec_gw = np.zeros(
        n_mc
    )

    rms_noise_ra = np.zeros(
        n_mc
    )

    rms_noise_dec = np.zeros(
        n_mc
    )

    rms_obs_ra = np.zeros(
        n_mc
    )

    rms_obs_dec = np.zeros(
        n_mc
    )

    # --------------------------------------------------------
    # MC loop
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "Starting Monte-Carlo simulations"
    )

    print(
        "=" * 72
    )

    t_start = time.time()

    for imc in range(n_mc):

        t0 = time.time()

        # ----------------------------------------------------
        # Random GW direction
        # ----------------------------------------------------

        ra_gw, dec_gw = (
            random_gw_direction(rng)
        )

        # ----------------------------------------------------
        # Random phase
        # ----------------------------------------------------

        phase = rng.uniform(
            0.0,
            2.0 * np.pi
        )

        # ----------------------------------------------------
        # Polarization angle
        #
        # By default psi is also randomized.
        #
        # If --fixed-psi is specified, psi remains fixed.
        # ----------------------------------------------------

        if args.fixed_psi is None:

            psi = rng.uniform(
                0.0,
                180.0
            )

        else:

            psi = args.fixed_psi

        # ----------------------------------------------------
        # Generate GW signal
        # ----------------------------------------------------

        pmra_gw, pmdec_gw = (
            generate_gw_realization(
                n=n,
                e_alpha=e_alpha,
                e_delta=e_delta,
                hc=args.hc,
                frequency_hz=args.frequency,
                ra_gw_deg=ra_gw,
                dec_gw_deg=dec_gw,
                phase=phase,
                psi_deg=psi,
                equal_polarization=(
                    args.equal_polarization
                )
            )
        )

        # ----------------------------------------------------
        # Generate Gaia noise
        # ----------------------------------------------------

        if args.no_noise:

            pmra_noise = np.zeros(
                n_qso,
                dtype=np.float64
            )

            pmdec_noise = np.zeros(
                n_qso,
                dtype=np.float64
            )

        else:

            pmra_noise, pmdec_noise = (
                generate_correlated_noise(
                    sig_pmra,
                    sig_pmdec,
                    corr,
                    rng
                )
            )

        # ----------------------------------------------------
        # Observed field
        # ----------------------------------------------------

        pmra_obs = (
            pmra_gw
            + pmra_noise
        )

        pmdec_obs = (
            pmdec_gw
            + pmdec_noise
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        ra_gw_all[imc] = ra_gw
        dec_gw_all[imc] = dec_gw

        phase_all[imc] = phase
        psi_all[imc] = psi

        rms_pmra_gw[imc] = np.sqrt(
            np.mean(
                pmra_gw**2
            )
        )

        rms_pmdec_gw[imc] = np.sqrt(
            np.mean(
                pmdec_gw**2
            )
        )

        rms_noise_ra[imc] = np.sqrt(
            np.mean(
                pmra_noise**2
            )
        )

        rms_noise_dec[imc] = np.sqrt(
            np.mean(
                pmdec_noise**2
            )
        )

        rms_obs_ra[imc] = np.sqrt(
            np.mean(
                pmra_obs**2
            )
        )

        rms_obs_dec[imc] = np.sqrt(
            np.mean(
                pmdec_obs**2
            )
        )

        # ----------------------------------------------------
        # Save complete catalogue if requested
        # ----------------------------------------------------

        if args.save_catalogues:

            filename = os.path.join(
                args.output_dir,
                f"mc_{imc:04d}.csv"
            )

            save_realization(
                filename,
                ra,
                dec,
                pmra_gw,
                pmdec_gw,
                pmra_noise,
                pmdec_noise
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = time.time() - t_start
        dt = time.time() - t0

        if imc == 0:

            eta = (
                dt * n_mc
            )

        else:

            eta = (
                elapsed
                / (imc + 1)
                * (n_mc - imc - 1)
            )

        print(
            f"MC {imc + 1:4d}/{n_mc:4d} | "
            f"RA_GW={ra_gw:8.3f} | "
            f"DEC_GW={dec_gw:8.3f} | "
            f"phase={phase:7.3f} | "
            f"time={dt:6.2f}s | "
            f"ETA={eta/60.0:7.2f} min"
        )

    # --------------------------------------------------------
    # Save MC summary
    # --------------------------------------------------------

    summary = np.column_stack([
        realization_id,
        ra_gw_all,
        dec_gw_all,
        phase_all,
        psi_all,
        rms_pmra_gw,
        rms_pmdec_gw,
        rms_noise_ra,
        rms_noise_dec,
        rms_obs_ra,
        rms_obs_dec
    ])

    header = (
        "realization,"
        "ra_gw_deg,"
        "dec_gw_deg,"
        "phase_rad,"
        "psi_deg,"
        "rms_pmra_gw,"
        "rms_pmdec_gw,"
        "rms_pmra_noise,"
        "rms_pmdec_noise,"
        "rms_pmra_obs,"
        "rms_pmdec_obs"
    )

    np.savetxt(
        args.summary,
        summary,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.12e"
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    total_time = (
        time.time()
        - t_start
    )

    print()
    print(
        "=" * 72
    )

    print(
        "Monte-Carlo simulation completed."
    )

    print(
        "=" * 72
    )

    print(
        f"Number of MC realizations : {n_mc}"
    )

    print(
        f"Number of QSO             : {n_qso:,}"
    )

    print(
        f"hc                        : "
        f"{args.hc:.6e}"
    )

    print(
        f"GW frequency              : "
        f"{args.frequency:.6e} Hz"
    )

    print(
        f"Total computation time    : "
        f"{total_time / 60.0:.2f} min"
    )

    print()
    print(
        f"Summary saved to: "
        f"{args.summary}"
    )

    if args.save_catalogues:

        print(
            f"Catalogues saved to: "
            f"{args.output_dir}"
        )

    print()


# ============================================================
# Argument parser
# ============================================================

def create_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Monte-Carlo simulator for astrometric "
            "gravitational-wave injections into Gaia QSO data."
        )
    )

    # --------------------------------------------------------
    # Input/output
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        required=True,
        help="Gaia QSO CSV catalogue."
    )

    parser.add_argument(
        "--summary",
        default="mc_summary.csv",
        help="MC summary output."
    )

    parser.add_argument(
        "--output-dir",
        default="mc_catalogues",
        help="Directory for individual MC catalogues."
    )

    parser.add_argument(
        "--save-catalogues",
        action="store_true",
        help=(
            "Save the complete simulated catalogue "
            "for every MC realization."
        )
    )

    # --------------------------------------------------------
    # MC
    # --------------------------------------------------------

    parser.add_argument(
        "--n-mc",
        type=int,
        default=1000,
        help="Number of MC realizations."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=99999,
        help="Master random seed."
    )

    # --------------------------------------------------------
    # GW
    # --------------------------------------------------------

    parser.add_argument(
        "--hc",
        type=float,
        required=True,
        help="Injected GW strain amplitude."
    )

    parser.add_argument(
        "--frequency",
        type=float,
        required=True,
        help="GW frequency in Hz."
    )

    parser.add_argument(
        "--equal-polarization",
        action="store_true",
        help=(
            "Use equal plus and cross polarization amplitudes."
        )
    )

    parser.add_argument(
        "--fixed-psi",
        type=float,
        default=None,
        help=(
            "Fixed polarization angle in degrees. "
            "If omitted, psi is randomized."
        )
    )

    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Do not add Gaia measurement noise."
    )

    return parser


# ============================================================
# Main
# ============================================================

def main():

    parser = create_parser()

    args = parser.parse_args()

    if args.n_mc < 1:

        raise ValueError(
            "--n-mc must be >= 1"
        )

    if args.hc <= 0:

        raise ValueError(
            "--hc must be positive."
        )

    if args.frequency <= 0:

        raise ValueError(
            "--frequency must be positive."
        )

    print()
    print(
        "=" * 72
    )

    print(
        "MC GW SIMULATOR"
    )

    print(
        "=" * 72
    )

    print(
        f"MC realizations : {args.n_mc}"
    )

    print(
        f"hc              : {args.hc:.6e}"
    )

    print(
        f"frequency       : "
        f"{args.frequency:.6e} Hz"
    )

    print(
        f"frequency       : "
        f"{args.frequency * 1e9:.6f} nHz"
    )

    print(
        f"seed             : {args.seed}"
    )

    print()

    run_mc(args)


if __name__ == "__main__":
    main()
