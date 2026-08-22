#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gw_simulator.py

Simulation of astrometric proper-motion signatures produced by
plane gravitational waves and a stochastic gravitational-wave background.

Designed for Gaia DR3 QSO simulations.

Input catalogue:
    ra, dec
    optional:
        sig_pmra
        sig_pmdec
        corr_pmra_pmdec

The covariance matrix for every source is

        C = [[sig_pmra^2, rho*sig_pmra*sig_pmdec],
             [rho*sig_pmra*sig_pmdec, sig_pmdec^2]]

The program can simulate:

    1. A single monochromatic plane GW.
    2. A stochastic GWB as a sum of independent plane GWs.
    3. Gaia-like correlated proper-motion noise.
    4. Observed proper motions = GW signal + noise.

Output columns:

    ra
    dec
    pmra_gw
    pmdec_gw
    pmra_noise
    pmdec_noise
    pmra_obs
    pmdec_obs

Units:

    RA, DEC:
        degrees

    proper motions:
        mas/yr

    strain:
        dimensionless

    GW frequency:
        Hz

The implementation uses NumPy and optionally Numba.

Author:
    Volodymyr Akhmetov
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ============================================================
# Constants
# ============================================================

RAD_TO_MAS = 206264806.24709636
SECONDS_PER_YEAR = 365.25 * 86400.0

MAS_TO_RAD = 1.0 / RAD_TO_MAS
YR_TO_SEC = 1.0 / SECONDS_PER_YEAR


# ============================================================
# Coordinate transformations
# ============================================================

def radec_to_unit_vectors(ra_deg: np.ndarray,
                           dec_deg: np.ndarray) -> np.ndarray:
    """
    Convert RA/DEC in degrees to Cartesian unit vectors.

    Parameters
    ----------
    ra_deg : array
        Right ascension [deg].
    dec_deg : array
        Declination [deg].

    Returns
    -------
    n : (N, 3) array
        Unit vectors pointing toward sources.
    """

    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    cos_dec = np.cos(dec)

    n = np.empty((len(ra), 3), dtype=np.float64)

    n[:, 0] = cos_dec * np.cos(ra)
    n[:, 1] = cos_dec * np.sin(ra)
    n[:, 2] = np.sin(dec)

    return n


def tangent_basis(ra_deg: np.ndarray,
                  dec_deg: np.ndarray):
    """
    Construct tangent basis vectors e_alpha and e_delta.

    e_alpha corresponds to increasing RA on the sky,
    i.e. the alpha-star direction.

    e_delta corresponds to increasing declination.

    Returns
    -------
    e_alpha : (N,3)
    e_delta : (N,3)
    """

    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    e_alpha = np.empty((len(ra), 3), dtype=np.float64)
    e_delta = np.empty((len(ra), 3), dtype=np.float64)

    e_alpha[:, 0] = -np.sin(ra)
    e_alpha[:, 1] = np.cos(ra)
    e_alpha[:, 2] = 0.0

    e_delta[:, 0] = -np.sin(dec) * np.cos(ra)
    e_delta[:, 1] = -np.sin(dec) * np.sin(ra)
    e_delta[:, 2] = np.cos(dec)

    return e_alpha, e_delta


# ============================================================
# GW propagation and polarization basis
# ============================================================

def propagation_vector(ra_deg: float,
                       dec_deg: float) -> np.ndarray:
    """
    Return GW propagation vector.

    The convention is:

        k = direction FROM the source toward the observer.

    Thus a GW arriving from (ra_gw, dec_gw) has

        k = n(ra_gw, dec_gw).

    """

    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ], dtype=np.float64)


def polarization_basis(ra_gw_deg: float,
                        dec_gw_deg: float,
                        psi_deg: float = 0.0):
    """
    Construct orthonormal GW polarization basis p, q.

    p and q are perpendicular to the propagation vector k.

    psi is the polarization angle.
    """

    ra = np.deg2rad(ra_gw_deg)
    dec = np.deg2rad(dec_gw_deg)
    psi = np.deg2rad(psi_deg)

    # Basis associated with increasing RA
    p0 = np.array([
        -np.sin(ra),
        np.cos(ra),
        0.0
    ])

    # Basis associated with increasing DEC
    q0 = np.array([
        -np.sin(dec) * np.cos(ra),
        -np.sin(dec) * np.sin(ra),
        np.cos(dec)
    ])

    # Rotate by polarization angle psi
    p = np.cos(psi) * p0 + np.sin(psi) * q0
    q = -np.sin(psi) * p0 + np.cos(psi) * q0

    return p, q


def polarization_tensors(p: np.ndarray,
                         q: np.ndarray):
    """
    Construct plus and cross polarization tensors.

        e_plus  = p⊗p - q⊗q
        e_cross = p⊗q + q⊗p
    """

    e_plus = np.outer(p, p) - np.outer(q, q)
    e_cross = np.outer(p, q) + np.outer(q, p)

    return e_plus, e_cross


# ============================================================
# Plane GW astrometric response
# ============================================================

def plane_gw_deflection(
        n: np.ndarray,
        k: np.ndarray,
        e_plus: np.ndarray,
        e_cross: np.ndarray,
        h_plus: float,
        h_cross: float
):
    """
    Astrometric deflection produced by a plane GW.

    The instantaneous angular deflection is

        delta n_i =
            1/2 * (n_i - k_i)/(1 - k.n)
                * n_j n_k h_jk
            - 1/2 * h_ij n_j

    where h_ij is the TT GW metric perturbation.

    Parameters
    ----------
    n :
        Source unit vectors, shape (N,3)

    k :
        GW propagation direction, shape (3,)

    e_plus, e_cross :
        Polarization tensors.

    h_plus, h_cross :
        Instantaneous GW strain amplitudes.

    Returns
    -------
    delta_n : (N,3)
        Angular deflection in radians.
    """

    h = h_plus * e_plus + h_cross * e_cross

    # n_i n_j h_ij
    hn = np.einsum("ni,ij,nj->n", n, h, n)

    # h_ij n_j
    hn_vec = np.einsum("ij,nj->ni", h, n)

    dot = np.dot(n, k)

    denominator = 1.0 - dot

    # Avoid numerical singularity near GW source direction.
    denominator = np.where(
        np.abs(denominator) < 1e-12,
        np.sign(denominator) * 1e-12 + (denominator == 0.0) * 1e-12,
        denominator
    )

    term1 = 0.5 * ((n - k) / denominator[:, None]) * hn[:, None]

    term2 = -0.5 * hn_vec

    return term1 + term2


def deflection_to_proper_motion(
        delta_n: np.ndarray,
        e_alpha: np.ndarray,
        e_delta: np.ndarray,
        frequency_hz: float
):
    """
    Convert GW angular deflection to proper motion.

    For a monochromatic GW,

        h(t) ~ cos(2 pi f t)

    and therefore

        d(delta n)/dt ~ 2 pi f delta_n.

    The result is converted from rad/s to mas/yr.
    """

    omega = 2.0 * np.pi * frequency_hz

    delta_dot = omega * delta_n

    pm_alpha_rad_s = np.sum(delta_dot * e_alpha, axis=1)
    pm_delta_rad_s = np.sum(delta_dot * e_delta, axis=1)

    conversion = RAD_TO_MAS * SECONDS_PER_YEAR

    pm_alpha = pm_alpha_rad_s * conversion
    pm_delta = pm_delta_rad_s * conversion

    return pm_alpha, pm_delta


# ============================================================
# Single plane GW
# ============================================================

def simulate_plane_gw(
        ra_deg: np.ndarray,
        dec_deg: np.ndarray,
        hc: float,
        frequency_hz: float,
        ra_gw_deg: float,
        dec_gw_deg: float,
        psi_deg: float = 0.0,
        phase: float = 0.0,
        plus_fraction: float = 1.0 / np.sqrt(2.0),
        cross_fraction: float = 1.0 / np.sqrt(2.0)
):
    """
    Simulate a single monochromatic plane GW.

    Parameters
    ----------
    hc :
        GW strain amplitude.

    frequency_hz :
        GW frequency [Hz].

    ra_gw_deg, dec_gw_deg :
        GW arrival direction [deg].

    psi_deg :
        Polarization angle.

    phase :
        Initial GW phase [rad].

    plus_fraction :
        Fraction of hc assigned to plus polarization.

    cross_fraction :
        Fraction of hc assigned to cross polarization.

    Returns
    -------
    pmra_gw, pmdec_gw :
        Proper-motion GW signal [mas/yr].
    """

    n = radec_to_unit_vectors(ra_deg, dec_deg)
    e_alpha, e_delta = tangent_basis(ra_deg, dec_deg)

    k = propagation_vector(ra_gw_deg, dec_gw_deg)

    p, q = polarization_basis(
        ra_gw_deg,
        dec_gw_deg,
        psi_deg
    )

    e_plus, e_cross = polarization_tensors(p, q)

    # Monochromatic GW at phase t
    h_plus = hc * plus_fraction * np.cos(phase)

    h_cross = hc * cross_fraction * np.sin(phase)

    delta_n = plane_gw_deflection(
        n,
        k,
        e_plus,
        e_cross,
        h_plus,
        h_cross
    )

    pmra, pmdec = deflection_to_proper_motion(
        delta_n,
        e_alpha,
        e_delta,
        frequency_hz
    )

    return pmra, pmdec


# ============================================================
# Covariance handling
# ============================================================

def build_covariance(
        sig_pmra: np.ndarray,
        sig_pmdec: np.ndarray,
        corr: np.ndarray
):
    """
    Build covariance matrices.

    Returns
    -------
    C : (N,2,2)
    """

    cov = corr * sig_pmra * sig_pmdec

    C = np.empty((len(sig_pmra), 2, 2), dtype=np.float64)

    C[:, 0, 0] = sig_pmra ** 2
    C[:, 0, 1] = cov
    C[:, 1, 0] = cov
    C[:, 1, 1] = sig_pmdec ** 2

    return C


def draw_correlated_noise(
        sig_pmra: np.ndarray,
        sig_pmdec: np.ndarray,
        corr: np.ndarray,
        rng: np.random.Generator
):
    """
    Generate correlated Gaussian proper-motion noise.

    Each QSO gets a 2D Gaussian random vector with covariance

        [[sigma_ra^2, rho sigma_ra sigma_dec],
         [rho sigma_ra sigma_dec, sigma_dec^2]]
    """

    n = len(sig_pmra)

    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)

    rho = np.clip(corr, -0.999999, 0.999999)

    noise_ra = sig_pmra * z1

    noise_dec = (
        sig_pmdec *
        (
            rho * z1
            + np.sqrt(1.0 - rho ** 2) * z2
        )
    )

    return noise_ra, noise_dec


# ============================================================
# Stochastic GWB
# ============================================================

def simulate_stochastic_gwb(
        ra_deg: np.ndarray,
        dec_deg: np.ndarray,
        hc: float,
        frequency_hz: float,
        n_waves: int = 1000,
        seed: int = 99999,
        polarization: str = "random"
):
    """
    Simulate a stochastic gravitational-wave background.

    The background is represented as a sum of independent plane GWs
    with random sky directions, phases and polarization states.

    Important:
        To preserve approximately fixed total RMS strain, the amplitude
        of every individual plane wave is scaled by

            hc / sqrt(n_waves).

    Parameters
    ----------
    hc :
        Total characteristic strain amplitude.

    frequency_hz :
        GW frequency [Hz].

    n_waves :
        Number of plane waves.

    seed :
        RNG seed.

    polarization :
        "random" or "equal".

    Returns
    -------
    pmra_gw, pmdec_gw
    """

    rng = np.random.default_rng(seed)

    n = radec_to_unit_vectors(ra_deg, dec_deg)
    e_alpha, e_delta = tangent_basis(ra_deg, dec_deg)

    n_sources = len(ra_deg)

    pmra_total = np.zeros(n_sources, dtype=np.float64)
    pmdec_total = np.zeros(n_sources, dtype=np.float64)

    # Each plane wave contributes approximately hc/sqrt(N)
    wave_amplitude = hc / np.sqrt(float(n_waves))

    # Generate all random parameters first.
    ra_waves = rng.uniform(0.0, 360.0, n_waves)

    sin_dec = rng.uniform(-1.0, 1.0, n_waves)
    dec_waves = np.rad2deg(np.arcsin(sin_dec))

    psi_waves = rng.uniform(0.0, 180.0, n_waves)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_waves)

    if polarization == "random":

        pol_angles = rng.uniform(
            0.0,
            2.0 * np.pi,
            n_waves
        )

    else:

        pol_angles = np.zeros(n_waves)

    for iw in range(n_waves):

        k = propagation_vector(
            ra_waves[iw],
            dec_waves[iw]
        )

        p, q = polarization_basis(
            ra_waves[iw],
            dec_waves[iw],
            psi_waves[iw]
        )

        e_plus, e_cross = polarization_tensors(p, q)

        phase = phases[iw]

        if polarization == "random":

            fp = np.cos(pol_angles[iw])
            fc = np.sin(pol_angles[iw])

        else:

            fp = 1.0 / np.sqrt(2.0)
            fc = 1.0 / np.sqrt(2.0)

        h_plus = (
            wave_amplitude
            * fp
            * np.cos(phase)
        )

        h_cross = (
            wave_amplitude
            * fc
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

        pmra, pmdec = deflection_to_proper_motion(
            delta_n,
            e_alpha,
            e_delta,
            frequency_hz
        )

        pmra_total += pmra
        pmdec_total += pmdec

    return pmra_total, pmdec_total


# ============================================================
# Catalogue I/O
# ============================================================

def detect_column(data, names):
    """
    Find first existing column from a list of possible names.
    """

    for name in names:

        if name in data.dtype.names:
            return name

    return None


def load_catalogue(filename: str):
    """
    Load Gaia-like QSO catalogue.

    Required:
        ra
        dec

    Optional:
        sig_pmra
        sig_pmdec
        corr_pmra_pmdec

    Accepted aliases:

        RA:
            ra, RA, ra_deg

        DEC:
            dec, DEC, dec_deg

        sigma pmra:
            sig_pmra
            pmra_error
            pmra_err

        sigma pmdec:
            sig_pmdec
            pmdec_error
            pmdec_err

        correlation:
            corr_pmra_pmdec
            pmra_pmdec_corr
            corr
    """

    print(f"Reading catalogue: {filename}")

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )

    ra_col = detect_column(
        data,
        ["ra", "RA", "ra_deg"]
    )

    dec_col = detect_column(
        data,
        ["dec", "DEC", "dec_deg"]
    )

    if ra_col is None or dec_col is None:

        raise ValueError(
            "Input catalogue must contain RA and DEC columns."
        )

    ra = np.asarray(data[ra_col], dtype=np.float64)
    dec = np.asarray(data[dec_col], dtype=np.float64)

    sig_ra_col = detect_column(
        data,
        [
            "sig_pmra",
            "pmra_error",
            "pmra_err"
        ]
    )

    sig_dec_col = detect_column(
        data,
        [
            "sig_pmdec",
            "pmdec_error",
            "pmdec_err"
        ]
    )

    corr_col = detect_column(
        data,
        [
            "corr_pmra_pmdec",
            "pmra_pmdec_corr",
            "corr"
        ]
    )

    if sig_ra_col is not None:

        sig_pmra = np.asarray(
            data[sig_ra_col],
            dtype=np.float64
        )

    else:

        sig_pmra = None

    if sig_dec_col is not None:

        sig_pmdec = np.asarray(
            data[sig_dec_col],
            dtype=np.float64
        )

    else:

        sig_pmdec = None

    if corr_col is not None:

        corr = np.asarray(
            data[corr_col],
            dtype=np.float64
        )

    else:

        corr = None

    return (
        ra,
        dec,
        sig_pmra,
        sig_pmdec,
        corr
    )


# ============================================================
# CSV output
# ============================================================

def save_catalogue(
        filename,
        ra,
        dec,
        pmra_gw,
        pmdec_gw,
        pmra_noise,
        pmdec_noise
):
    """
    Save simulated catalogue.
    """

    pmra_obs = pmra_gw + pmra_noise
    pmdec_obs = pmdec_gw + pmdec_noise

    print(f"Writing output: {filename}")

    header = (
        "ra,dec,"
        "pmra_gw,pmdec_gw,"
        "pmra_noise,pmdec_noise,"
        "pmra_obs,pmdec_obs"
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

    np.savetxt(
        filename,
        arr,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.12e"
    )


# ============================================================
# Argument parser
# ============================================================

def create_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Simulate astrometric proper-motion signatures "
            "of gravitational waves for Gaia QSO catalogues."
        )
    )

    # --------------------------------------------------------
    # Input / output
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        required=True,
        help="Input QSO CSV catalogue."
    )

    parser.add_argument(
        "--output",
        default="gw_simulated.csv",
        help="Output CSV catalogue."
    )

    # --------------------------------------------------------
    # Simulation mode
    # --------------------------------------------------------

    parser.add_argument(
        "--mode",
        choices=[
            "plane",
            "stochastic"
        ],
        default="plane",
        help="GW simulation mode."
    )

    # --------------------------------------------------------
    # GW parameters
    # --------------------------------------------------------

    parser.add_argument(
        "--hc",
        type=float,
        required=True,
        help="GW strain amplitude."
    )

    parser.add_argument(
        "--frequency",
        type=float,
        required=True,
        help="GW frequency in Hz."
    )

    # --------------------------------------------------------
    # Plane GW direction
    # --------------------------------------------------------

    parser.add_argument(
        "--ra-gw",
        type=float,
        default=45.0,
        help="GW RA in degrees."
    )

    parser.add_argument(
        "--dec-gw",
        type=float,
        default=45.0,
        help="GW DEC in degrees."
    )

    parser.add_argument(
        "--psi",
        type=float,
        default=0.0,
        help="GW polarization angle in degrees."
    )

    parser.add_argument(
        "--phase",
        type=float,
        default=0.0,
        help="Initial GW phase in radians."
    )

    # --------------------------------------------------------
    # Stochastic GWB
    # --------------------------------------------------------

    parser.add_argument(
        "--n-waves",
        type=int,
        default=1000,
        help="Number of plane waves for stochastic GWB."
    )

    parser.add_argument(
        "--polarization",
        choices=[
            "random",
            "equal"
        ],
        default="random",
        help="Polarization model for stochastic GWB."
    )

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Do not add Gaia-like proper-motion noise."
    )

    parser.add_argument(
        "--sigma-pmra",
        type=float,
        default=None,
        help=(
            "Constant sigma_pmra [mas/yr] if catalogue "
            "does not contain uncertainties."
        )
    )

    parser.add_argument(
        "--sigma-pmdec",
        type=float,
        default=None,
        help=(
            "Constant sigma_pmdec [mas/yr] if catalogue "
            "does not contain uncertainties."
        )
    )

    parser.add_argument(
        "--corr",
        type=float,
        default=0.0,
        help=(
            "Constant pmra-pmdec correlation if catalogue "
            "does not contain correlations."
        )
    )

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=99999,
        help="Random seed."
    )

    return parser


# ============================================================
# Main simulation
# ============================================================

def main():

    parser = create_parser()
    args = parser.parse_args()

    print()
    print("=" * 70)
    print(" Gaia Astrometric Gravitational-Wave Simulator")
    print("=" * 70)
    print()

    print(f"NumPy version: {np.__version__}")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print()

    # --------------------------------------------------------
    # Read catalogue
    # --------------------------------------------------------

    (
        ra,
        dec,
        sig_pmra,
        sig_pmdec,
        corr
    ) = load_catalogue(args.input)

    n_sources = len(ra)

    print(f"Number of QSO: {n_sources:,}")

    # --------------------------------------------------------
    # Check coordinates
    # --------------------------------------------------------

    if np.any(~np.isfinite(ra)):
        raise ValueError("RA contains invalid values.")

    if np.any(~np.isfinite(dec)):
        raise ValueError("DEC contains invalid values.")

    if np.any(dec < -90.0) or np.any(dec > 90.0):

        raise ValueError(
            "DEC must be in [-90, +90] degrees."
        )

    # --------------------------------------------------------
    # Noise parameters
    # --------------------------------------------------------

    if sig_pmra is None:

        if args.sigma_pmra is None:

            print(
                "WARNING: sigma_pmra is not present in "
                "the catalogue."
            )

            print(
                "Using sigma_pmra = 0.1 mas/yr."
            )

            sig_pmra = np.full(
                n_sources,
                0.1,
                dtype=np.float64
            )

        else:

            sig_pmra = np.full(
                n_sources,
                args.sigma_pmra,
                dtype=np.float64
            )

    if sig_pmdec is None:

        if args.sigma_pmdec is None:

            print(
                "WARNING: sigma_pmdec is not present "
                "in the catalogue."
            )

            print(
                "Using sigma_pmdec = 0.1 mas/yr."
            )

            sig_pmdec = np.full(
                n_sources,
                0.1,
                dtype=np.float64
            )

        else:

            sig_pmdec = np.full(
                n_sources,
                args.sigma_pmdec,
                dtype=np.float64
            )

    if corr is None:

        corr = np.full(
            n_sources,
            args.corr,
            dtype=np.float64
        )

    corr = np.clip(corr, -0.999999, 0.999999)

    # --------------------------------------------------------
    # Print simulation parameters
    # --------------------------------------------------------

    print()
    print("Simulation parameters")
    print("-" * 70)

    print(f"Mode             : {args.mode}")
    print(f"hc               : {args.hc:.6e}")
    print(f"frequency        : {args.frequency:.6e} Hz")
    print(
        f"frequency        : "
        f"{args.frequency * 1.0e9:.6f} nHz"
    )

    if args.mode == "plane":

        print(
            f"GW direction     : "
            f"RA={args.ra_gw:.3f} deg, "
            f"DEC={args.dec_gw:.3f} deg"
        )

        print(
            f"polarization psi : "
            f"{args.psi:.3f} deg"
        )

        print(
            f"phase            : "
            f"{args.phase:.6f} rad"
        )

    else:

        print(
            f"number of waves  : "
            f"{args.n_waves}"
        )

        print(
            f"polarization     : "
            f"{args.polarization}"
        )

    print(f"random seed      : {args.seed}")

    # --------------------------------------------------------
    # GW signal
    # --------------------------------------------------------

    print()
    print("Generating GW signal...")

    if args.mode == "plane":

        pmra_gw, pmdec_gw = simulate_plane_gw(
            ra_deg=ra,
            dec_deg=dec,
            hc=args.hc,
            frequency_hz=args.frequency,
            ra_gw_deg=args.ra_gw,
            dec_gw_deg=args.dec_gw,
            psi_deg=args.psi,
            phase=args.phase
        )

    elif args.mode == "stochastic":

        pmra_gw, pmdec_gw = simulate_stochastic_gwb(
            ra_deg=ra,
            dec_deg=dec,
            hc=args.hc,
            frequency_hz=args.frequency,
            n_waves=args.n_waves,
            seed=args.seed,
            polarization=args.polarization
        )

    else:

        raise RuntimeError(
            f"Unknown simulation mode: {args.mode}"
        )

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    if args.no_noise:

        print("Noise generation: disabled.")

        pmra_noise = np.zeros(
            n_sources,
            dtype=np.float64
        )

        pmdec_noise = np.zeros(
            n_sources,
            dtype=np.float64
        )

    else:

        print("Generating correlated Gaia-like noise...")

        rng = np.random.default_rng(args.seed)

        (
            pmra_noise,
            pmdec_noise
        ) = draw_correlated_noise(
            sig_pmra,
            sig_pmdec,
            corr,
            rng
        )

    # --------------------------------------------------------
    # Observed signal
    # --------------------------------------------------------

    pmra_obs = pmra_gw + pmra_noise
    pmdec_obs = pmdec_gw + pmdec_noise

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("Simulation statistics")
    print("-" * 70)

    print(
        f"GW pmRA RMS      : "
        f"{np.std(pmra_gw):.6e} mas/yr"
    )

    print(
        f"GW pmDEC RMS     : "
        f"{np.std(pmdec_gw):.6e} mas/yr"
    )

    print(
        f"Noise pmRA RMS   : "
        f"{np.std(pmra_noise):.6e} mas/yr"
    )

    print(
        f"Noise pmDEC RMS  : "
        f"{np.std(pmdec_noise):.6e} mas/yr"
    )

    print(
        f"Observed pmRA RMS: "
        f"{np.std(pmra_obs):.6e} mas/yr"
    )

    print(
        f"Observed pmDEC RMS: "
        f"{np.std(pmdec_obs):.6e} mas/yr"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_catalogue(
        args.output,
        ra,
        dec,
        pmra_gw,
        pmdec_gw,
        pmra_noise,
        pmdec_noise
    )

    print()
    print("=" * 70)
    print("Simulation finished successfully.")
    print("=" * 70)
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
