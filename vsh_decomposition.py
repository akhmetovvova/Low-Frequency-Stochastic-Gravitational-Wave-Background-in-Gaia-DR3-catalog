#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vsh_decomposition.py

Vector Spherical Harmonic decomposition of an astrometric
proper-motion field.

Input CSV:
    ra, dec, pmra, pmdec

Optional:
    sig_pmra
    sig_pmdec
    corr_pmra_pmdec

The proper motions are assumed to be in mas/yr.

The field is decomposed as

    V(n) = sum_lm [
              a_lm^E Y_lm^E(n)
            + a_lm^B Y_lm^B(n)
          ]

where

    Y_lm^E = (1/sqrt(l(l+1))) grad Y_lm

    Y_lm^B = (1/sqrt(l(l+1))) n x grad Y_lm

The code uses complex spherical harmonics internally.

Power:

    P_l^E = sum_m |a_lm^E|^2

    P_l^B = sum_m |a_lm^B|^2

    P_l   = P_l^E + P_l^B

The output coefficients contain real and imaginary parts.

Author:
    Volodymyr Akhmetov
"""

from __future__ import annotations

import argparse
import numpy as np

from scipy.special import sph_harm_y


# ============================================================
# Constants
# ============================================================

DEG_TO_RAD = np.pi / 180.0


# ============================================================
# VSH basis
# ============================================================

def scalar_spherical_harmonic(l, m, theta, phi):
    """
    Complex scalar spherical harmonic Y_lm.

    scipy convention:

        theta = polar angle [0, pi]
        phi   = azimuth [0, 2pi]

    """

    # scipy.special.sph_harm_y(n, m, theta, phi)
    return sph_harm_y(l, m, theta, phi)


def vsh_basis(
        l,
        m,
        ra,
        dec
):
    """
    Calculate E and B vector spherical harmonics.

    Parameters
    ----------
    l, m :
        Harmonic degree/order.

    ra, dec :
        radians.

    Returns
    -------
    E_ra, E_dec :
        E-mode vector spherical harmonic components.

    B_ra, B_dec :
        B-mode vector spherical harmonic components.
    """

    theta = np.pi / 2.0 - dec
    phi = ra

    # --------------------------------------------------------
    # Scalar harmonic
    # --------------------------------------------------------

    Y = scalar_spherical_harmonic(
        l,
        m,
        theta,
        phi
    )

    # --------------------------------------------------------
    # Numerical derivative in theta.
    #
    # This is intentionally implemented using a high-order
    # central finite difference. For the typical Gaia
    # application lmax ~ 5-10 this is sufficiently accurate.
    # --------------------------------------------------------

    h = 1.0e-6

    theta_p = theta + h
    theta_m = theta - h

    Y_p = scalar_spherical_harmonic(
        l,
        m,
        theta_p,
        phi
    )

    Y_m = scalar_spherical_harmonic(
        l,
        m,
        theta_m,
        phi
    )

    dY_dtheta = (Y_p - Y_m) / (2.0 * h)

    # derivative wrt phi
    dY_dphi = 1j * m * Y

    norm = np.sqrt(l * (l + 1.0))

    # --------------------------------------------------------
    # Vector spherical harmonics
    #
    # e_theta  -> decreasing DEC
    # e_phi    -> increasing RA
    #
    # Y_E = [dY/dtheta e_theta
    #        + 1/sin(theta) dY/dphi e_phi] / norm
    #
    # Y_B = [-1/sin(theta) dY/dphi e_theta
    #        + dY/dtheta e_phi] / norm
    # --------------------------------------------------------

    sin_theta = np.sin(theta)

    # Avoid numerical problems near poles
    sin_theta = np.maximum(
        np.abs(sin_theta),
        1.0e-12
    ) * np.sign(
        sin_theta + 1.0e-30
    )

    E_theta = dY_dtheta / norm

    E_phi = dY_dphi / (
        norm * sin_theta
    )

    B_theta = -dY_dphi / (
        norm * sin_theta
    )

    B_phi = dY_dtheta / norm

    # --------------------------------------------------------
    # Convert theta component to DEC component.
    #
    # e_theta points toward decreasing DEC:
    #
    # e_theta = -e_delta
    #
    # Therefore:
    #
    # V_dec = -V_theta
    # --------------------------------------------------------

    E_dec = -E_theta
    E_ra = E_phi

    B_dec = -B_theta
    B_ra = B_phi

    return E_ra, E_dec, B_ra, B_dec


# ============================================================
# Data loading
# ============================================================

def load_catalogue(filename):

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True
    )

    names = data.dtype.names

    required = [
        "ra",
        "dec",
        "pmra",
        "pmdec"
    ]

    for name in required:

        if name not in names:

            raise ValueError(
                f"Missing column: {name}"
            )

    ra = np.asarray(
        data["ra"],
        dtype=float
    )

    dec = np.asarray(
        data["dec"],
        dtype=float
    )

    pmra = np.asarray(
        data["pmra"],
        dtype=float
    )

    pmdec = np.asarray(
        data["pmdec"],
        dtype=float
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    if "sig_pmra" in names:

        sig_pmra = np.asarray(
            data["sig_pmra"],
            dtype=float
        )

    else:

        sig_pmra = np.ones_like(pmra)

    if "sig_pmdec" in names:

        sig_pmdec = np.asarray(
            data["sig_pmdec"],
            dtype=float
        )

    else:

        sig_pmdec = np.ones_like(pmdec)

    if "corr_pmra_pmdec" in names:

        corr = np.asarray(
            data["corr_pmra_pmdec"],
            dtype=float
        )

    else:

        corr = np.zeros_like(pmra)

    return (
        ra,
        dec,
        pmra,
        pmdec,
        sig_pmra,
        sig_pmdec,
        corr
    )


# ============================================================
# Build covariance matrix
# ============================================================

def build_covariance(
        sig_ra,
        sig_dec,
        corr
):

    n = len(sig_ra)

    C = np.zeros(
        (2*n, 2*n),
        dtype=float
    )

    for i in range(n):

        c = (
            corr[i]
            * sig_ra[i]
            * sig_dec[i]
        )

        C[2*i, 2*i] = (
            sig_ra[i] ** 2
        )

        C[2*i+1, 2*i+1] = (
            sig_dec[i] ** 2
        )

        C[2*i, 2*i+1] = c
        C[2*i+1, 2*i] = c

    return C


# ============================================================
# Weighted VSH decomposition
# ============================================================

def build_design_matrix(
        ra,
        dec,
        lmax
):
    """
    Construct VSH design matrix.

    For every (l,m) two columns are generated:

        E_lm
        B_lm

    The data vector is

        [pmra_1, pmdec_1,
         pmra_2, pmdec_2, ...]

    """

    n = len(ra)

    columns = []
    labels = []

    for l in range(1, lmax + 1):

        for m in range(-l, l + 1):

            (
                E_ra,
                E_dec,
                B_ra,
                B_dec
            ) = vsh_basis(
                l,
                m,
                ra,
                dec
            )

            col_E = np.empty(
                2*n,
                dtype=complex
            )

            col_B = np.empty(
                2*n,
                dtype=complex
            )

            col_E[0::2] = E_ra
            col_E[1::2] = E_dec

            col_B[0::2] = B_ra
            col_B[1::2] = B_dec

            columns.append(col_E)
            labels.append(
                (l, m, "E")
            )

            columns.append(col_B)
            labels.append(
                (l, m, "B")
            )

    A = np.column_stack(columns)

    return A, labels


def weighted_least_squares(
        A,
        y,
        covariance
):
    """
    Solve

        y = A x + noise

    using full covariance matrix.

    x =
        (A^H C^-1 A)^-1
        A^H C^-1 y
    """

    Cinv = np.linalg.inv(covariance)

    AH = A.conj().T

    normal = (
        AH
        @ Cinv
        @ A
    )

    rhs = (
        AH
        @ Cinv
        @ y
    )

    cov_x = np.linalg.inv(normal)

    x = cov_x @ rhs

    return x, cov_x


# ============================================================
# Power spectrum
# ============================================================

def calculate_power(
        coefficients,
        labels
):

    power_E = {}
    power_B = {}

    for coeff, label in zip(
        coefficients,
        labels
    ):

        l, m, mode = label

        value = np.abs(coeff) ** 2

        if mode == "E":

            power_E[l] = (
                power_E.get(l, 0.0)
                + value
            )

        else:

            power_B[l] = (
                power_B.get(l, 0.0)
                + value
            )

    power_total = {}

    for l in range(
        1,
        max(power_E.keys()) + 1
    ):

        power_total[l] = (
            power_E.get(l, 0.0)
            + power_B.get(l, 0.0)
        )

    return (
        power_E,
        power_B,
        power_total
    )


# ============================================================
# Output coefficients
# ============================================================

def save_coefficients(
        filename,
        coefficients,
        labels,
        covariance
):

    rows = []

    for i, (coeff, label) in enumerate(
        zip(coefficients, labels)
    ):

        l, m, mode = label

        sigma = np.sqrt(
            max(
                covariance[i, i].real,
                0.0
            )
        )

        rows.append([
            l,
            m,
            mode,
            coeff.real,
            coeff.imag,
            np.abs(coeff),
            sigma
        ])

    rows = np.asarray(
        rows,
        dtype=object
    )

    header = (
        "l,m,mode,"
        "real,imag,amplitude,sigma"
    )

    np.savetxt(
        filename,
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt="%s"
    )


def save_power_spectrum(
        filename,
        power_E,
        power_B,
        power_total
):

    lmax = max(power_total.keys())

    rows = []

    for l in range(
        1,
        lmax + 1
    ):

        rows.append([
            l,
            power_E.get(l, 0.0),
            power_B.get(l, 0.0),
            power_total.get(l, 0.0)
        ])

    rows = np.asarray(
        rows
    )

    header = (
        "l,power_E,power_B,power_total"
    )

    np.savetxt(
        filename,
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt="%d,%.12e,%.12e,%.12e"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Vector Spherical Harmonic decomposition "
            "of Gaia proper-motion field."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file."
    )

    parser.add_argument(
        "--lmax",
        type=int,
        default=5,
        help="Maximum harmonic degree."
    )

    parser.add_argument(
        "--coefficients",
        default="vsh_coefficients.csv",
        help="Output coefficient file."
    )

    parser.add_argument(
        "--power",
        default="vsh_power.csv",
        help="Output power spectrum."
    )

    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Perform unweighted least squares."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Read catalogue
    # --------------------------------------------------------

    (
        ra_deg,
        dec_deg,
        pmra,
        pmdec,
        sig_ra,
        sig_dec,
        corr
    ) = load_catalogue(
        args.input
    )

    n = len(ra_deg)

    print(
        f"Number of sources: {n:,}"
    )

    print(
        f"VSH lmax: {args.lmax}"
    )

    # --------------------------------------------------------
    # Convert coordinates
    # --------------------------------------------------------

    ra = ra_deg * DEG_TO_RAD
    dec = dec_deg * DEG_TO_RAD

    # --------------------------------------------------------
    # Data vector
    # --------------------------------------------------------

    y = np.empty(
        2*n,
        dtype=float
    )

    y[0::2] = pmra
    y[1::2] = pmdec

    # --------------------------------------------------------
    # Design matrix
    # --------------------------------------------------------

    print(
        "Building VSH design matrix..."
    )

    A, labels = build_design_matrix(
        ra,
        dec,
        args.lmax
    )

    print(
        f"Number of VSH coefficients: "
        f"{A.shape[1]}"
    )

    # --------------------------------------------------------
    # Weighted/unweighted solution
    # --------------------------------------------------------

    if args.unweighted:

        print(
            "Using unweighted least squares."
        )

        normal = (
            A.conj().T
            @ A
        )

        rhs = (
            A.conj().T
            @ y
        )

        cov_x = np.linalg.inv(
            normal
        )

        coefficients = (
            cov_x @ rhs
        )

    else:

        print(
            "Using covariance-weighted "
            "least squares."
        )

        # ----------------------------------------------------
        # Important:
        #
        # For very large Gaia catalogues we should NOT build
        # a 2N x 2N covariance matrix.
        #
        # Since each source has an independent 2x2 covariance
        # matrix, the weighting can be performed source by
        # source.
        # ----------------------------------------------------

        ncoeff = A.shape[1]

        normal = np.zeros(
            (ncoeff, ncoeff),
            dtype=complex
        )

        rhs = np.zeros(
            ncoeff,
            dtype=complex
        )

        for i in range(n):

            Ai = A[
                2*i:2*i+2,
                :
            ]

            yi = y[
                2*i:2*i+2
            ]

            rho = corr[i]

            C = np.array([
                [
                    sig_ra[i]**2,
                    rho
                    * sig_ra[i]
                    * sig_dec[i]
                ],
                [
                    rho
                    * sig_ra[i]
                    * sig_dec[i],
                    sig_dec[i]**2
                ]
            ])

            Ci = np.linalg.inv(C)

            normal += (
                Ai.conj().T
                @ Ci
                @ Ai
            )

            rhs += (
                Ai.conj().T
                @ Ci
                @ yi
            )

        cov_x = np.linalg.inv(
            normal
        )

        coefficients = (
            cov_x @ rhs
        )

    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    (
        power_E,
        power_B,
        power_total
    ) = calculate_power(
        coefficients,
        labels
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VSH POWER SPECTRUM"
    )
    print("=" * 70)

    print(
        f"{'l':>3} "
        f"{'P_E':>18} "
        f"{'P_B':>18} "
        f"{'P_total':>18}"
    )

    for l in range(
        1,
        args.lmax + 1
    ):

        print(
            f"{l:3d} "
            f"{power_E.get(l, 0.0):18.8e} "
            f"{power_B.get(l, 0.0):18.8e} "
            f"{power_total.get(l, 0.0):18.8e}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print(
        f"Saving coefficients: "
        f"{args.coefficients}"
    )

    save_coefficients(
        args.coefficients,
        coefficients,
        labels,
        cov_x
    )

    print(
        f"Saving power spectrum: "
        f"{args.power}"
    )

    save_power_spectrum(
        args.power,
        power_E,
        power_B,
        power_total
    )

    print()
    print(
        "VSH decomposition completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
