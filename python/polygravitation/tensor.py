# ===============================================================
# tensor.py
# Gravity gradient tensor (Hessian of potential)
#
# Purpose:
#   Implements the dyadic second-derivative (Werner–Scheeres-style)
#   formulation:
#     Γ(P) = G*rho [ Σ_e L_e(P) E_e  -  Σ_f ω_f(P) F_f ]
#
# Design:
#   - Reuses the *accel/tensor* precomputations (face/edge dyads).
#   - Parallelizes over point chunks via the base thread pool.
#
# Public:
#   class TensorOps (mixin):
#       - gravity_tensor(points, block_size=4096)
#
# Notes:
#   - Algorithms unchanged; returns (N,3,3) or (3,3).
# ===============================================================

import numpy as np
import concurrent.futures
from .helpers import log_term, solid_angle_tri

class TensorOps:
    def _tensor_chunk(self, P: np.ndarray) -> np.ndarray:
        out_chunk = np.zeros((P.shape[0], 3, 3), dtype=np.float64)

        Vi, Vj, Vk = self.Vi, self.Vj, self.Vk
        Fdyads = self.Fdyads
        edge_u, edge_v = self.edge_u, self.edge_v
        edge_E = self.edge_E
        edge_len = self.edge_len
        eps = self.eps

        omega_f = solid_angle_tri(P, Vi, Vj, Vk, eps)
        r_i_norm = np.linalg.norm(edge_u[None, :, :] - P[:, None, :], axis=2)
        r_j_norm = np.linalg.norm(edge_v[None, :, :] - P[:, None, :], axis=2)
        log_e = log_term(r_i_norm, r_j_norm, edge_len, eps)

        face_sum = np.einsum('bf,fij->bij', omega_f, Fdyads, optimize=True)
        edge_sum = np.einsum('be,eij->bij', log_e,    edge_E, optimize=True)

        out_chunk[:] = self.G * self.rho * (edge_sum - face_sum)
        return out_chunk

    def gravity_tensor(self, points: np.ndarray, block_size: int = 100) -> np.ndarray:
        """
        Compute the gravitational tensor (second derivatives of potential)
        at the given evaluation points.

        Automatically switches between serial and multi-threaded modes:
        - If number of points <= block_size → run serially (single chunk)
        - Else → use thread pool for parallel block processing

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, 3) containing evaluation points.
        block_size : int, optional
            Number of points per processing chunk (default = 100).
            If N <= block_size, runs serially to avoid threading overhead.

        Returns
        -------
        np.ndarray
            Array of shape (N, 3, 3) containing the gravitational tensor Γ.
            Returns a single 3×3 matrix if N == 1.
        """
        self._ensure_accel_tensor_precomp()

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        N = pts.shape[0]
        out = np.zeros((N, 3, 3), dtype=np.float64)

        # ---- Adaptive: small workloads use direct serial evaluation ----
        if N <= block_size:
            try:
                out[:] = self._tensor_chunk(pts)
            except Exception as exc:
                print(f"Direct tensor computation failed: {exc}")
                out[:] = np.nan
            return out[0] if N == 1 else out

        # ---- Parallel multi-threaded block evaluation ----
        if not self.pool:
            raise RuntimeError("Thread pool is not running. Was close() called?")

        futures = {}
        for s, e in self._iter_chunks(N, block_size):
            future = self.pool.submit(self._tensor_chunk, pts[s:e])
            futures[future] = (s, e)

        for f in concurrent.futures.as_completed(futures):
            s, e = futures[f]
            try:
                out[s:e] = f.result()
            except Exception as exc:
                print(f"Chunk {s}:{e} tensor failed: {exc}")
                out[s:e] = np.nan

        return out[0] if N == 1 else out

