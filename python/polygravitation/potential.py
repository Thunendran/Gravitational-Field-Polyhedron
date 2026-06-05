# ===============================================================
# potential.py
# Gravitational potential of a homogeneous polyhedron
#
# Purpose:
#   Implements the face-summation formulation for the scalar
#   potential U at one or more evaluation points.
#
# Design:
#   - Uses only the *potential-specific* precomputations:
#       edge lengths (L_ij, L_jk, L_ki),
#       safe inverses, and three dot products.
#   - Parallelizes over point chunks via the base thread pool.
#
# Public:
#   class PotentialOps (mixin):
#       - potential(points, block_size=4096)
#
# Notes:
#   - Math/algorithms are unchanged from the analytical model.
#   - Returns shape (N,) or scalar for a single point.
# ===============================================================

import numpy as np
import concurrent.futures
from .helpers import log_term, arctan_term

class PotentialOps:
    # ---- chunk worker ----
    def _potential_chunk(self, P: np.ndarray) -> np.ndarray:
        out_chunk = np.zeros(P.shape[0], dtype=np.float64)

        Vi, Vj, Vk = self.Vi, self.Vj, self.Vk
        nhat = self.n_hat
        nraw = self.n_raw_norm
        inv_L_ij, inv_L_jk, inv_L_ki = self.inv_L_ij, self.inv_L_jk, self.inv_L_ki
        L_ij, L_jk, L_ki = self.L_ij, self.L_jk, self.L_ki
        dot_ij_ki, dot_jk_ij, dot_ki_jk = self.dot_ij_ki, self.dot_jk_ij, self.dot_ki_jk
        eps = self.eps

        Pi = Vi[None, :, :] - P[:, None, :]
        Pj = Vj[None, :, :] - P[:, None, :]
        Pk = Vk[None, :, :] - P[:, None, :]

        r_i = np.linalg.norm(Pi, axis=2)
        r_j = np.linalg.norm(Pj, axis=2)
        r_k = np.linalg.norm(Pk, axis=2)

        diff_z = np.einsum('bnj,nj->bn', Pi, nhat, optimize=True)
        dz2 = diff_z * diff_z

        det_ij = np.einsum('bnj,nj->bn', np.cross(Pi, Pj), nhat, optimize=True)
        det_jk = np.einsum('bnj,nj->bn', np.cross(Pj, Pk), nhat, optimize=True)
        det_ki = np.einsum('bnj,nj->bn', np.cross(Pk, Pi), nhat, optimize=True)

        L12 = log_term(r_i, r_j, L_ij, eps) * inv_L_ij[None, :]
        L23 = log_term(r_j, r_k, L_jk, eps) * inv_L_jk[None, :]
        L31 = log_term(r_k, r_i, L_ki, eps) * inv_L_ki[None, :]

        for M in (L12, L23, L31):
            bad = ~np.isfinite(M)
            if bad.any():
                M[bad] = 0.0

        numerator = diff_z * nraw[None, :]
        S1 = arctan_term(numerator, det_ki, det_ij, dz2, dot_ij_ki[None, :], r_i, eps)
        S2 = arctan_term(numerator, det_ij, det_jk, dz2, dot_jk_ij[None, :], r_j, eps)
        S3 = arctan_term(numerator, det_jk, det_ki, dz2, dot_ki_jk[None, :], r_k, eps)

        term1 = diff_z * (det_ij * L12 + det_jk * L23 + det_ki * L31)
        term2 = dz2 * (S1 + S2 + S3 - np.sign(diff_z) * np.pi)
        total = 0.5 * np.sum(term1 - term2, axis=1)

        out_chunk[:] = -self.G * self.rho * total
        return out_chunk

    # ---- public API ----
    def potential(self, points: np.ndarray, block_size: int = 100) -> np.ndarray:
        """
        Compute the gravitational potential at given evaluation points.
        
        Automatically switches between serial and multi-threaded modes:
        - If number of points <= block_size → run serially (single chunk)
        - Else → use thread pool for parallel block processing

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, 3) containing evaluation points.
        block_size : int, optional
            Number of points per processing chunk (default = 10).
            If N <= block_size, the call runs serially to avoid thread overhead.

        Returns
        -------
        np.ndarray
            Array of gravitational potentials at each point (float64).
            Returns a scalar if a single point is passed.
        """
        self._ensure_potential_precomp()

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        N = pts.shape[0]
        out = np.zeros(N, dtype=np.float64)

        # ---- Adaptive: small workloads use direct serial evaluation ----
        if N <= block_size:
            try:
                out[:] = self._potential_chunk(pts)
            except Exception as exc:
                print(f"Direct potential computation failed: {exc}")
                out[:] = np.nan
            return out[0] if N == 1 else out

        # ---- Parallel multi-threaded block evaluation ----
        if not self.pool:
            raise RuntimeError("Thread pool is not running. Was close() called?")

        futures = {}
        for s, e in self._iter_chunks(N, block_size):
            future = self.pool.submit(self._potential_chunk, pts[s:e])
            futures[future] = (s, e)

        for f in concurrent.futures.as_completed(futures):
            s, e = futures[f]
            try:
                out[s:e] = f.result()
            except Exception as exc:
                print(f"Chunk {s}:{e} potential failed: {exc}")
                out[s:e] = np.nan

        return out[0] if N == 1 else out

