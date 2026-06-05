# ===============================================================
# core.py
# User-facing class combining base + potential/accel/tensor ops
#
# Purpose:
#   Provide a single, convenient API:
#       PolyhedronGravitation(V, F, G=1.0, density=1.0, eps=0.0,
#                             orient_faces=True, n_threads=None)
#       .potential(points), .acceleration(points), .gravity_tensor(points)
#
# Design:
#   - Multiple-inheritance mixins:
#       PotentialOps + AccelOps + TensorOps + PolyhedronBase
#   - Selective precomputations:
#       potential() -> only potential invariants
#       acceleration()/gravity_tensor() -> dyads/topology
#   - Thread pool schedules *point chunks* (no nested/block threads).
#
# Notes:
#   - Float64 machine precision throughout.
#   - Algorithms and formulas are identical to the analytical model;
#     only code organization and precompute gating differ.
# ===============================================================

from .base import PolyhedronBase
from .potential import PotentialOps
from .acceleration import AccelOps
from .tensor import TensorOps

class PolyhedronGravitation(PotentialOps, AccelOps, TensorOps, PolyhedronBase):
    """
    Float64 machine-precision analytical polyhedron gravity with:
      • selective precomputations (potential vs accel/tensor),
      • shared common geometry,
      • point-chunk parallelism across a thread pool.

    Usage:
        model = PolyhedronGravitation(V, F, G=1.0, density=1.0, eps=0.0, orient_faces=True, n_threads=None)
        U = model.potential(points)
        g = model.acceleration(points)
        Γ = model.gravity_tensor(points)
        model.close()
    """
    pass
