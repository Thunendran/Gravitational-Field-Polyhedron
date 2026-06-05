# ===============================================================
# PolyhedronHelpers.jl
# Supporting mathematical utilities for Polyhedral Gravitation
#
# Authors: Thunendran Periyandy, Michael Bevis
# Description:
#   Provides numerically stable helper functions for the evaluation
#   of the gravitational potential, acceleration, and tensor terms.
#   These include logarithmic, arctangent, and solid-angle routines.
# ===============================================================

using LinearAlgebra
using StaticArrays

@inline _sv3(a::Real, b::Real, c::Real) = SVector{3,Float64}(Float64(a), Float64(b), Float64(c))

# Safe log term: ln((ra + rb + rab) / (ra + rb - rab))
@inline function logterm(ra::Float64, rb::Float64, rab::Float64, eps::Float64)
    sumab = ra + rb
    num = sumab + rab
    den = sumab - rab
    (rab > eps && den > eps) ? log(num/den) : 0.0
end

# Arctangent term (single-point variant), mirrors arctan2 in Python
@inline function arctanterm(numerator::Float64, det_a::Float64, det_b::Float64,
                            dz2::Float64, dot_ab::Float64, r::Float64, eps::Float64)
    if r <= eps
        return 0.0
    end
    denom = -((det_b * det_a) + dz2 * dot_ab) / r
    return atan(numerator, denom)
end

# Signed solid angle for a triangle at a single point
@inline function solid_angle_tri(P::SVector{3,Float64},
                                 v1::SVector{3,Float64},
                                 v2::SVector{3,Float64},
                                 v3::SVector{3,Float64},
                                 eps::Float64)
    r1 = v1 - P; r2 = v2 - P; r3 = v3 - P
    r1n = norm(r1); r2n = norm(r2); r3n = norm(r3)
    triple = dot(cross(r2, r3), r1)
    denom = (r1n*r2n*r3n +
             dot(r1, r2)*r3n +
             dot(r2, r3)*r1n +
             dot(r3, r1)*r2n) + eps
    return 2.0 * atan(triple, denom)
end
