# ===============================================================
# Polyhedron_Gravitation_Big.jl
# Main Interface Module (BigFloat Precision)
#
# The Gravitational Field of a Homogeneous Polyhedron
# Authors: Thunendran Periyandy, Michael Bevis
# (c) 2025, The Ohio State University / Sabaragamuwa University of Sri Lanka
#
# Purpose:
#   Integrates all BigFloat submodules of the PolyGravitation framework,
#   providing a unified interface for computing the gravitational
#   potential, acceleration, and gradient tensor of a homogeneous polyhedron
#   with arbitrary-precision arithmetic.
#
# Description:
#   This BigFloat version mirrors the standard (Float64) implementation,
#   offering extremely high precision suitable for sensitivity analysis,
#   stability verification, and validation of analytical results.
# ===============================================================

module Polyhedron_Gravitation

include(joinpath(@__DIR__, "PolyhedronBase.jl"))
include(joinpath(@__DIR__, "PolyhedronPotential.jl"))
include(joinpath(@__DIR__, "PolyhedronAcceleration.jl"))
include(joinpath(@__DIR__, "PolyhedronTensor.jl"))

using .PolyhedronBase
using .PolyhedronPotential
using .PolyhedronAcceleration
using .PolyhedronTensor

export PolyhedronGravity, build_polyhedron, potential, acceleration, gravity_tensor

end # module

