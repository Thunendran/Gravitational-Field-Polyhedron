# ===============================================================
# PolyhedronHelpers_Big.jl
# Supporting mathematical utilities for Polyhedral Gravitation (BigFloat)
#
# Authors: Thunendran Periyandy, Michael Bevis
# Description:
#   High-precision helper functions for potential, acceleration,
#   and tensor evaluation in BigFloat arithmetic.
# ===============================================================

using LinearAlgebra
using StaticArrays

# Construct SVector{3,BigFloat}
@inline _sv3(a::Real, b::Real, c::Real) = SVector{3,BigFloat}(BigFloat(a), BigFloat(b), BigFloat(c))

# ---------------------------------------------------------------
# Safe logarithmic term (BigFloat)
# ln((ra + rb + rab) / (ra + rb - rab))
# ---------------------------------------------------------------
@inline function logterm(ra::BigFloat, rb::BigFloat, rab::BigFloat, eps::BigFloat)
    sumab = ra + rb
    num = sumab + rab
    den = sumab - rab
    (rab > eps && den > eps) ? log(num / den) : BigFloat(0)
end

# ---------------------------------------------------------------
# Arctangent term (BigFloat)
# Equivalent to atan2(numerator, denom)
# ---------------------------------------------------------------
@inline function arctanterm(numerator::BigFloat, det_a::BigFloat, det_b::BigFloat,
                            dz2::BigFloat, dot_ab::BigFloat, r::BigFloat, eps::BigFloat)
    if r <= eps
        return BigFloat(0)
    end
    denom = -((det_b * det_a) + dz2 * dot_ab) / r
    return atan(numerator, denom)
end

# ---------------------------------------------------------------
# Signed solid angle for a triangle at a single point (BigFloat)
# ---------------------------------------------------------------
@inline function solid_angle_tri(P::SVector{3,BigFloat},
                                 v1::SVector{3,BigFloat},
                                 v2::SVector{3,BigFloat},
                                 v3::SVector{3,BigFloat},
                                 eps::BigFloat)
    r1 = v1 - P
    r2 = v2 - P
    r3 = v3 - P

    r1n = norm(r1)
    r2n = norm(r2)
    r3n = norm(r3)

    triple = dot(cross(r2, r3), r1)
    denom = (r1n * r2n * r3n +
             dot(r1, r2) * r3n +
             dot(r2, r3) * r1n +
             dot(r3, r1) * r2n) + eps

    return BigFloat(2) * atan(triple, denom)
end
