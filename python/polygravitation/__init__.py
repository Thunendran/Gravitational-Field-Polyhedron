# ===============================================================
# polygrav/__init__.py
#
# Package init for the machine-precision (float64) analytical
# polyhedron gravity library.
#
# Public API:
#   from polygrav import PolyhedronGravitation
#
# Notes:
#   - Keeps the top-level surface minimal and stable.
#   - Implementation lives in core.py and submodules.
# ===============================================================

from .core import PolyhedronGravitation

__all__ = ["PolyhedronGravitation"]
