# ===============================================================
# base.py
# Core geometry container + selective precomputations + threading
#
# Purpose:
#   Holds mesh data and common geometry (faces, normals, triplets),
#   and exposes *selective* precomputation gates:
#     - _ensure_potential_precomp()      -> edges/lengths/dot products
#     - _ensure_accel_tensor_precomp()   -> face dyads + edge dyads
#
# Design:
#   - Algorithms are unchanged; only precompute what each method needs.
#   - A single ThreadPoolExecutor is used to parallelize over
#     *point chunks* (no nested/block-level threading).
#
# Public:
#   class PolyhedronBase:
#       - stores V, F, Vi/Vj/Vk, n_hat, etc.
#       - manages thread pool and chunk scheduling
#
# Notes:
#   - Float64 machine precision throughout.
#   - Face orientation (optional) via centroid test.
#   - Common data are C-contiguous numpy arrays for speed.
# ===============================================================

import os
import numpy as np
import concurrent.futures

__all__ = ["PolyhedronBase"]

class PolyhedronBase:
    def __init__(self,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 G: float = 1.0,
                 density: float = 1.0,
                 eps: float = 0.0,
                 orient_faces: bool = True,
                 n_threads: int | None = None):
        # Physical
        self.G   = float(G)
        self.rho = float(density)
        self.eps = float(eps)

        # Mesh (float64/int32, C-contiguous)
        self.V = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
        self.F = np.ascontiguousarray(np.asarray(faces,    dtype=np.int32))

        # Common precomputation (faces, normals, etc.)
        self._precompute_common(orient_faces)

        # Flags for selective precomputation
        self._have_potential = False
        self._have_accel     = False   # accel + tensor share this (edge dyads)
        self._have_tensor    = False   # (no extra beyond accel)

        # Thread pool (point-chunk parallelism)
        if n_threads is None:
            n_threads = os.cpu_count() or 1
        self.n_threads = int(n_threads)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads)

    # ---------- lifecycle ----------
    def __del__(self):
        if hasattr(self, "pool") and self.pool:
            self.pool.shutdown(wait=False)

    def close(self):
        if hasattr(self, "pool") and self.pool:
            self.pool.shutdown(wait=True)
            self.pool = None

    # ---------- chunk scheduling ----------
    def _iter_chunks(self, N: int, block_size: int) :
        """
        Split N points into ~n_threads chunks (or user block_size), avoiding too-small blocks.
        """
        # ensure at least one chunk per thread
        num_chunks = max(self.n_threads, min(N, (N + block_size - 1) // block_size))
        eff_bs = max(1, (N + num_chunks - 1) // num_chunks)
        for s in range(0, N, eff_bs):
            e = min(s + eff_bs, N)
            yield s, e

    # ---------- common precomputation ----------
    def _precompute_common(self, orient_faces: bool):
        F = self.F
        V = self.V
        Fi, Fj, Fk = F[:,0], F[:,1], F[:,2]
        v0 = V[Fi]; v1 = V[Fj]; v2 = V[Fk]

        normals_raw = np.cross(v1 - v0, v2 - v0)

        if orient_faces:
            centroid = V.mean(axis=0)
            face_centers = (v0 + v1 + v2) / 3.0
            flip = (np.einsum('ij,ij->i', normals_raw, face_centers - centroid) < 0.0)
            if np.any(flip):
                self.F[flip] = self.F[flip][:, [0, 2, 1]]
                Fi, Fj, Fk = self.F[:,0], self.F[:,1], self.F[:,2]
                v0 = V[Fi]; v1 = V[Fj]; v2 = V[Fk]
                normals_raw = np.cross(v1 - v0, v2 - v0)

        # Store face vertex triplets
        self.Vi = np.ascontiguousarray(V[self.F[:,0]])
        self.Vj = np.ascontiguousarray(V[self.F[:,1]])
        self.Vk = np.ascontiguousarray(V[self.F[:,2]])

        # Normals
        self.n_raw_norm = np.linalg.norm(normals_raw, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            self.n_hat = np.divide(
                normals_raw, self.n_raw_norm[:, None],
                out=np.zeros_like(normals_raw),
                where=self.n_raw_norm[:, None] > self.eps
            )
        # Basic contiguity
        self.n_hat = np.ascontiguousarray(self.n_hat)

    # ---------- selective precomputations ----------
    def _ensure_potential_precomp(self):
        if self._have_potential:
            return
        # edges (for potential formulas)
        e_ij = self.Vj - self.Vi
        e_jk = self.Vk - self.Vj
        e_ki = self.Vi - self.Vk

        self.L_ij = np.linalg.norm(e_ij, axis=1)
        self.L_jk = np.linalg.norm(e_jk, axis=1)
        self.L_ki = np.linalg.norm(e_ki, axis=1)

        # safe inverses
        self.inv_L_ij = np.zeros_like(self.L_ij)
        self.inv_L_jk = np.zeros_like(self.L_jk)
        self.inv_L_ki = np.zeros_like(self.L_ki)
        nz_ij = self.L_ij > self.eps
        nz_jk = self.L_jk > self.eps
        nz_ki = self.L_ki > self.eps
        self.inv_L_ij[nz_ij] = 1.0 / self.L_ij[nz_ij]
        self.inv_L_jk[nz_jk] = 1.0 / self.L_jk[nz_jk]
        self.inv_L_ki[nz_ki] = 1.0 / self.L_ki[nz_ki]

        # dot products
        self.dot_ij_ki = np.einsum('ij,ij->i', e_ij, e_ki)
        self.dot_jk_ij = np.einsum('ij,ij->i', e_jk, e_ij)
        self.dot_ki_jk = np.einsum('ij,ij->i', e_ki, e_jk)

        self._have_potential = True

    def _ensure_accel_tensor_precomp(self):
        if self._have_accel:
            return
        # Face dyads
        self.Fdyads = np.einsum('fi,fj->fij', self.n_hat, self.n_hat, optimize=True)

        # Edge topology & dyads
        edge_to_faces, edge_map = {}, {}
        for f_idx, (a,b,c) in enumerate(self.F):
            for u, v in ((a,b), (b,c), (c,a)):
                key = (u, v) if u < v else (v, u)
                edge_to_faces.setdefault(key, []).append(f_idx)
                edge_map[key] = (u, v)

        edge_i, edge_j, edge_len, edge_E = [], [], [], []
        V = self.V; nhat = self.n_hat
        for key, faces in edge_to_faces.items():
            if len(faces) != 2:
                continue
            u, v = edge_map[key]
            fa, fb = faces[0], faces[1]
            na, nb = nhat[fa], nhat[fb]

            evec = V[v] - V[u]
            L = np.linalg.norm(evec)
            if L <= self.eps:
                continue
            ehat = evec / L

            a,b,c = self.F[fa]
            off = next(t for t in (a,b,c) if t not in key)
            face_a = np.array([a,b,c], dtype=np.int32)
            off_pos = np.where(face_a == off)[0][0]
            after_off = face_a[(off_pos + 1) % 3]

            tA = np.cross(na, ehat)
            tB = np.cross(nb, ehat)
            if after_off == u:
                tA = -tA
            else:
                tB = -tB
            Ee = np.outer(na, tA) + np.outer(nb, tB)

            edge_i.append(u); edge_j.append(v); edge_len.append(L); edge_E.append(Ee)

        self.edge_i = np.asarray(edge_i, dtype=np.int32)
        self.edge_j = np.asarray(edge_j, dtype=np.int32)
        self.edge_len = np.asarray(edge_len, dtype=np.float64)
        self.edge_E   = np.ascontiguousarray(np.asarray(edge_E, dtype=np.float64))
        self.edge_u   = np.ascontiguousarray(self.V[self.edge_i])
        self.edge_v   = np.ascontiguousarray(self.V[self.edge_j])

        self._have_accel = True
        self._have_tensor = True  # tensor uses the same dyads
