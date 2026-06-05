# ===============================================================
# acceleration.py
# Gravitational acceleration (vector field) of a polyhedron
#
# Purpose:
#   Implements the dyadic acceleration formulation:
#     g(P) = G*rho [ Σ_f ω_f(P) (F_f r_f)  -  Σ_e L_e(P) (E_e r_u) ]
#
# Design:
#   - Triggers the *accel/tensor* precomputations only:
#       face dyads (F_f) and edge dyads (E_e) + topology.
#   - Shared with tensor (no duplication).
#   - Parallelizes over point chunks via the base thread pool.
#
# Public:
#   class AccelOps (mixin):
#       - acceleration(points, block_size=4096)
#
# Notes:
#   - Algorithms are unchanged; organization only.
#   - Returns shape (N,3) or (3,) for a single point.
# ===============================================================

import numpy as np
import concurrent.futures
from .helpers import log_term, solid_angle_tri

class AccelOps:
    def _acceleration_chunk(self, P: np.ndarray) -> np.ndarray:
        out_chunk = np.zeros(P.shape, dtype=np.float64)

        Vi, Vj, Vk = self.Vi, self.Vj, self.Vk
        Fdyads = self.Fdyads
        eps = self.eps
        edge_u, edge_v = self.edge_u, self.edge_v
        edge_E = self.edge_E
        edge_len = self.edge_len

        # faces
        r_f = Vi[None, :, :] - P[:, None, :]
        omega_f = solid_angle_tri(P, Vi, Vj, Vk, eps)
        face_vectors = np.einsum('fij,bfj->bfi', Fdyads, r_f, optimize=True)
        B = np.einsum('bf,bfi->bi', omega_f, face_vectors, optimize=True)

        # edges
        r_u = edge_u[None, :, :] - P[:, None, :]
        r_v = edge_v[None, :, :] - P[:, None, :]
        r_i_norm = np.linalg.norm(r_u, axis=2)
        r_j_norm = np.linalg.norm(r_v, axis=2)
        log_e = log_term(r_i_norm, r_j_norm, edge_len, eps)
        edge_vectors = np.einsum('eij,bej->bei', edge_E, r_u, optimize=True)
        A = np.einsum('be,bei->bi', log_e, edge_vectors, optimize=True)

        out_chunk[:] = self.G * self.rho * (B - A)
        return out_chunk

    def acceleration(self, points: np.ndarray, block_size: int = 100) -> np.ndarray:
        """
        Compute gravitational acceleration at given evaluation points.

        Automatically switches between serial and multi-threaded modes:
        - If number of points <= block_size → run serially (single chunk)
        - Else → use thread pool for parallel block processing

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, 3) containing evaluation points.
        block_size : int, optional
            Number of points per processing chunk (default = 100).
            If N <= block_size, runs in serial mode to avoid threading overhead.

        Returns
        -------
        np.ndarray
            Array of accelerations (shape Nx3). Returns a single 3-vector if N == 1.
        """
        self._ensure_accel_tensor_precomp()

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        N = pts.shape[0]
        out = np.zeros_like(pts, dtype=np.float64)

        # ---- Adaptive: small workloads use direct serial evaluation ----
        if N <= block_size:
            try:
                out[:] = self._acceleration_chunk(pts)
            except Exception as exc:
                print(f"Direct acceleration computation failed: {exc}")
                out[:] = np.nan
            return out[0] if N == 1 else out

        # ---- Parallel multi-threaded block evaluation ----
        if not self.pool:
            raise RuntimeError("Thread pool is not running. Was close() called?")

        futures = {}
        for s, e in self._iter_chunks(N, block_size):
            future = self.pool.submit(self._acceleration_chunk, pts[s:e])
            futures[future] = (s, e)

        for f in concurrent.futures.as_completed(futures):
            s, e = futures[f]
            try:
                out[s:e] = f.result()
            except Exception as exc:
                print(f"Chunk {s}:{e} acceleration failed: {exc}")
                out[s:e] = np.nan

        return out[0] if N == 1 else out

