# ===============================================================
# helpers.py
# Supporting mathematical utilities for Polyhedral Gravitation
#
# Purpose:
#   Provides numerically stable helper functions used by the
#   potential, acceleration, and tensor formulations:
#     - log_term   : safe log((ra+rb+rab)/(ra+rb-rab))
#     - arctan_term: stable arctangent kernel
#     - solid_angle_tri: signed solid angle of a triangle
#
# Notes:
#   - All routines are vectorized over batches of points.
#   - Algorithms are identical to the analytical formulation;
#     only organized for reuse.
# ===============================================================

import numpy as np

__all__ = ["log_term", "arctan_term", "solid_angle_tri"]

def log_term(ra: np.ndarray, rb: np.ndarray, rab: np.ndarray, eps: float) -> np.ndarray:
    """
    Safe log: ln((ra + rb + rab)/(ra + rb - rab)).
    Vectorized over points (rows) and faces/edges (cols).
    """
    sum_ab = ra + rb
    rab_b = rab[None, :]
    num = sum_ab + rab_b
    den = sum_ab - rab_b
    ok = (rab_b > eps) & (den > eps)
    out = np.zeros_like(ra)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.log(np.where(ok, num / den, 1.0))
    out[ok] = val[ok]
    return out

def arctan_term(numerator, det_a, det_b, dz2, dot_ab, r, eps):
    """
    Vectorized atan2 equivalent used by the face-summation formulation.
    """
    out = np.zeros_like(r)
    ok = r > eps
    denom = -((det_b * det_a) + dz2 * dot_ab)
    with np.errstate(divide='ignore', invalid='ignore'):
        out[ok] = np.arctan2(numerator[ok], denom[ok] / r[ok])
    return out

def solid_angle_tri(P, v1, v2, v3, eps):
    """
    Signed solid angle of triangle (v1,v2,v3) as seen from P (batched).
    P: (B,3), v*: (F,3) — returns (B,F)
    """
    r1 = v1[None, :, :] - P[:, None, :]
    r2 = v2[None, :, :] - P[:, None, :]
    r3 = v3[None, :, :] - P[:, None, :]
    r1n = np.linalg.norm(r1, axis=-1)
    r2n = np.linalg.norm(r2, axis=-1)
    r3n = np.linalg.norm(r3, axis=-1)
    triple = np.einsum('bfi,bfi->bf', np.cross(r2, r3), r1, optimize=True)
    denom = (r1n * r2n * r3n
             + np.einsum('bfi,bfi->bf', r1, r2, optimize=True) * r3n
             + np.einsum('bfi,bfi->bf', r2, r3, optimize=True) * r1n
             + np.einsum('bfi,bfi->bf', r3, r1, optimize=True) * r2n)
    return 2.0 * np.arctan2(triple, denom + eps)
