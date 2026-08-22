#!/usr/bin/env python3
"""
Fast HDC calculation for quasar proper motions.

Input CSV columns:
ra, dec, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr

Units:
ra, dec             : degree
pmra, pmdec         : mas/yr
errors              : mas/yr
corr                : dimensionless

The code:
  * uses the full 2x2 covariance matrix of each source;
  * projects the covariance onto the pair-dependent parallel/perpendicular
    basis;
  * calculates four HDC components;
  * calculates pair-dependent inverse-variance weights;
  * accumulates pairs on the fly (never stores N(N-1)/2 pairs);
  * uses Numba parallel loops.

Important:
The formal error returned below describes the weighted mean under the
assumption that pair measurements are independent. Since HDC pairs share
sources, bins are in general correlated. A jackknife/bootstrap covariance
should be used for a fully empirical uncertainty/covariance estimate.
"""

import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------

def read_catalog(filename):
    df = pd.read_csv(filename)

    required = [
        "ra", "dec", "pmra", "pmdec",
        "pmra_error", "pmdec_error", "pmra_pmdec_corr"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    ra = np.deg2rad(df["ra"].to_numpy(dtype=np.float64))
    dec = np.deg2rad(df["dec"].to_numpy(dtype=np.float64))

    pmra = df["pmra"].to_numpy(dtype=np.float64)
    pmdec = df["pmdec"].to_numpy(dtype=np.float64)

    sra = df["pmra_error"].to_numpy(dtype=np.float64)
    sdec = df["pmdec_error"].to_numpy(dtype=np.float64)
    rho = df["pmra_pmdec_corr"].to_numpy(dtype=np.float64)

    # Full covariance matrices.
    c00 = sra * sra
    c11 = sdec * sdec
    c01 = rho * sra * sdec

    # Unit position vectors.
    cosd = np.cos(dec)
    xyz = np.empty((len(df), 3), dtype=np.float64)
    xyz[:, 0] = cosd * np.cos(ra)
    xyz[:, 1] = cosd * np.sin(ra)
    xyz[:, 2] = np.sin(dec)

    # Local tangent vectors.
    # e_alpha = (-sin alpha, cos alpha, 0)
    # e_delta = (-sin delta cos alpha, -sin delta sin alpha, cos delta)
    ea = np.empty_like(xyz)
    ed = np.empty_like(xyz)

    ea[:, 0] = -np.sin(ra)
    ea[:, 1] =  np.cos(ra)
    ea[:, 2] = 0.0

    ed[:, 0] = -np.sin(dec) * np.cos(ra)
    ed[:, 1] = -np.sin(dec) * np.sin(ra)
    ed[:, 2] =  np.cos(dec)

    mu = np.column_stack((pmra, pmdec))
    cov = np.zeros((len(df), 2, 2), dtype=np.float64)
    cov[:, 0, 0] = c00
    cov[:, 0, 1] = c01
    cov[:, 1, 0] = c01
    cov[:, 1, 1] = c11

    return xyz, ea, ed, mu, cov


# ----------------------------------------------------------------------
# HDC kernel
# ----------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def hdc_kernel(xyz, ea, ed, mu, cov,
               theta_edges, weighted=True):

    n = xyz.shape[0]
    nb = len(theta_edges) - 1

    # Components:
    # 0 = parallel-parallel
    # 1 = perpendicular-perpendicular
    # 2 = parallel-perpendicular
    # 3 = perpendicular-parallel
    sum_w = np.zeros((4, nb))
    sum_wc = np.zeros((4, nb))
    sum_wc2 = np.zeros((4, nb))

    # Parallel accumulation over source i.
    # Each thread gets its own temporary arrays.
    tmp_w = np.zeros((n, 4, nb))
    tmp_wc = np.zeros((n, 4, nb))
    tmp_wc2 = np.zeros((n, 4, nb))

    for i in prange(n - 1):
        ri = xyz[i]
        eai = ea[i]
        edi = ed[i]

        for j in range(i + 1, n):

            rj = xyz[j]

            # Great-circle angle.
            dot = ri[0]*rj[0] + ri[1]*rj[1] + ri[2]*rj[2]
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0

            theta = np.arccos(dot)

            # Find angular bin.
            # Linear scan is acceptable for moderate nb.
            # For many bins this can be replaced by searchsorted.
            b = -1
            for k in range(nb):
                if theta >= theta_edges[k] and theta < theta_edges[k+1]:
                    b = k
                    break

            if b < 0:
                continue

            # Tangent direction at i toward j:
            # t_i = r_j - cos(theta) r_i
            st = np.sin(theta)
            if st < 1.0e-12:
                continue

            ti0 = (rj[0] - dot*ri[0]) / st
            ti1 = (rj[1] - dot*ri[1]) / st
            ti2 = (rj[2] - dot*ri[2]) / st

            # Tangent direction at j toward i.
            tj0 = (ri[0] - dot*rj[0]) / st
            tj1 = (ri[1] - dot*rj[1]) / st
            tj2 = (ri[2] - dot*rj[2]) / st

            # Perpendicular directions.
            # n_i = r_i x t_i
            epi0 = ri[1]*ti2 - ri[2]*ti1
            epi1 = ri[2]*ti0 - ri[0]*ti2
            epi2 = ri[0]*ti1 - ri[1]*ti0

            epj0 = rj[1]*tj2 - rj[2]*tj1
            epj1 = rj[2]*tj0 - rj[0]*tj2
            epj2 = rj[0]*tj1 - rj[1]*tj0

            # Project proper motions.
            mpi = eai[0]*mu[i,0] + edi[0]*mu[i,1]
            # Above mpi is not yet parallel; calculate explicitly.
            mui_p = ti0*(eai[0]*mu[i,0] + edi[0]*mu[i,1]) \
                  + ti1*(eai[1]*mu[i,0] + edi[1]*mu[i,1]) \
                  + ti2*(eai[2]*mu[i,0] + edi[2]*mu[i,1])

            mui_x = epi0*(eai[0]*mu[i,0] + edi[0]*mu[i,1]) \
                  + epi1*(eai[1]*mu[i,0] + edi[1]*mu[i,1]) \
                  + epi2*(eai[2]*mu[i,0] + edi[2]*mu[i,1])

            muj_p = tj0*(ea[j,0]*mu[j,0] + ed[j,0]*mu[j,1]) \
                  + tj1*(ea[j,1]*mu[j,0] + ed[j,1]*mu[j,1]) \
                  + tj2*(ea[j,2]*mu[j,0] + ed[j,2]*mu[j,1])

            muj_x = epj0*(ea[j,0]*mu[j,0] + ed[j,0]*mu[j,1]) \
                  + epj1*(ea[j,1]*mu[j,0] + ed[j,1]*mu[j,1]) \
                  + epj2*(ea[j,2]*mu[j,0] + ed[j,2]*mu[j,1])

            # Projection coefficients:
            # q = (q_alpha, q_delta) such that projected mu = q . mu_local
            qi_p0 = ti0*eai[0] + ti1*eai[1] + ti2*eai[2]
            qi_p1 = ti0*edi[0] + ti1*edi[1] + ti2*edi[2]
            qi_x0 = epi0*eai[0] + epi1*eai[1] + epi2*eai[2]
            qi_x1 = epi0*edi[0] + epi1*edi[1] + epi2*edi[2]

            qj_p0 = tj0*ea[j,0] + tj1*ea[j,1] + tj2*ea[j,2]
            qj_p1 = tj0*ed[j,0] + tj1*ed[j,1] + tj2*ed[j,2]
            qj_x0 = epj0*ea[j,0] + epj1*ea[j,1] + epj2*ea[j,2]
            qj_x1 = epj0*ed[j,0] + epj1*ed[j,1] + epj2*ed[j,2]

            # Projected variances.
            ci00 = cov[i,0,0]
            ci01 = cov[i,0,1]
            ci11 = cov[i,1,1]

            cj00 = cov[j,0,0]
            cj01 = cov[j,0,1]
            cj11 = cov[j,1,1]

            vip = qi_p0*qi_p0*ci00 + 2.0*qi_p0*qi_p1*ci01 \
                + qi_p1*qi_p1*ci11
            vix = qi_x0*qi_x0*ci00 + 2.0*qi_x0*qi_x1*ci01 \
                + qi_x1*qi_x1*ci11

            vjp = qj_p0*qj_p0*cj00 + 2.0*qj_p0*qj_p1*cj01 \
                + qj_p1*qj_p1*cj11
            vjx = qj_x0*qj_x0*cj00 + 2.0*qj_x0*qj_x1*cj01 \
                + qj_x1*qj_x1*cj11

            # Four products.
            c0 = mui_p*muj_p
            c1 = mui_x*muj_x
            c2 = mui_p*muj_x
            c3 = mui_x*muj_p

            # Product variances.
            # Var(XY) for independent Gaussian X,Y:
            # mu_X^2 Var(Y) + mu_Y^2 Var(X) + Var(X)Var(Y)
            v0 = mui_p*mui_p*vjp + muj_p*muj_p*vip + vip*vjp
            v1 = mui_x*mui_x*vjx + muj_x*muj_x*vix + vix*vjx
            v2 = mui_p*mui_p*vjx + muj_x*muj_x*vip + vip*vjx
            v3 = mui_x*mui_x*vjp + muj_p*muj_p*vix + vix*vjp

            vals = (c0, c1, c2, c3)
            vars_ = (v0, v1, v2, v3)

            for a in range(4):
                va = vars_[a]
                if weighted:
                    if va <= 0.0 or not np.isfinite(va):
                        continue
                    w = 1.0 / va
                else:
                    w = 1.0

                tmp_w[i,a,b] += w
                tmp_wc[i,a,b] += w*vals[a]
                tmp_wc2[i,a,b] += w*vals[a]*vals[a]

    # Reduce thread-independent source accumulators.
    for i in range(n):
        for a in range(4):
            for b in range(nb):
                sum_w[a,b] += tmp_w[i,a,b]
                sum_wc[a,b] += tmp_wc[i,a,b]
                sum_wc2[a,b] += tmp_wc2[i,a,b]

    return sum_w, sum_wc, sum_wc2


# ----------------------------------------------------------------------
# Convert accumulated pairs to Gamma and errors
# ----------------------------------------------------------------------

def finalize_hdc(sum_w, sum_wc, sum_wc2):
    """
    Gamma = sum(w C) / sum(w)

    Formal variance of the weighted mean:
        1 / sum(w)

    Also returns the weighted sample variance of pair values.
    The latter is useful as a diagnostic, but it does NOT account for
    correlations between pairs sharing the same source.
    """
    gamma = np.full_like(sum_w, np.nan)
    err_formal = np.full_like(sum_w, np.nan)
    scatter = np.full_like(sum_w, np.nan)

    good = sum_w > 0

    gamma[good] = sum_wc[good] / sum_w[good]

    # If w = 1/Var(C), then variance of weighted mean is 1/sum(w).
    err_formal[good] = 1.0 / np.sqrt(sum_w[good])

    # Weighted second central moment.
    second = np.zeros_like(sum_w)
    second[good] = sum_wc2[good] / sum_w[good]
    scatter[good] = np.sqrt(
        np.maximum(second[good] - gamma[good]**2, 0.0)
    )

    return gamma, err_formal, scatter


# ----------------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------------

def calculate_hdc(filename,
                  n_bins=40,
                  theta_min_deg=0.0,
                  theta_max_deg=180.0,
                  n_threads=16,
                  weighted=True):

    set_num_threads(n_threads)

    xyz, ea, ed, mu, cov = read_catalog(filename)

    edges = np.linspace(
        np.deg2rad(theta_min_deg),
        np.deg2rad(theta_max_deg),
        n_bins + 1
    )

    sum_w, sum_wc, sum_wc2 = hdc_kernel(
        xyz, ea, ed, mu, cov, edges, weighted
    )

    gamma, err, scatter = finalize_hdc(
        sum_w, sum_wc, sum_wc2
    )

    theta = 0.5 * (edges[:-1] + edges[1:])

    return {
        "theta_deg": np.rad2deg(theta),
        "Gamma_pp": gamma[0],
        "Gamma_xx": gamma[1],
        "Gamma_px": gamma[2],
        "Gamma_xp": gamma[3],
        "err_pp": err[0],
        "err_xx": err[1],
        "err_px": err[2],
        "err_xp": err[3],
        "scatter_pp": scatter[0],
        "scatter_xx": scatter[1],
        "scatter_px": scatter[2],
        "scatter_xp": scatter[3],
        "weight_pp": sum_w[0],
        "weight_xx": sum_w[1],
        "weight_px": sum_w[2],
        "weight_xp": sum_w[3],
    }


def save_hdc(result, filename):
    cols = [
        result["theta_deg"],
        result["Gamma_pp"], result["err_pp"],
        result["Gamma_xx"], result["err_xx"],
        result["Gamma_px"], result["err_px"],
        result["Gamma_xp"], result["err_xp"],
        result["weight_pp"], result["weight_xx"],
        result["weight_px"], result["weight_xp"],
    ]

    header = (
        "theta_deg "
        "Gamma_parallel_parallel err "
        "Gamma_perp_perp err "
        "Gamma_parallel_perp err "
        "Gamma_perp_parallel err "
        "weight_pp weight_xx weight_px weight_xp"
    )

    np.savetxt(filename, np.column_stack(cols), header=header)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("-o", "--output", default="hdc.dat")
    parser.add_argument("-n", "--bins", type=int, default=40)
    parser.add_argument("-t", "--threads", type=int, default=16)
    parser.add_argument("--unweighted", action="store_true")

    args = parser.parse_args()

    result = calculate_hdc(
        args.input_csv,
        n_bins=args.bins,
        n_threads=args.threads,
        weighted=not args.unweighted
    )

    save_hdc(result, args.output)

    print("HDC calculation finished.")
    print("Results:", args.output)
