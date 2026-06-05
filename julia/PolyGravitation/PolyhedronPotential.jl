# ===============================================================
# PolyhedronPotential.jl
# Module: PolyhedronPotential
#
# Purpose:
#   Computes the gravitational potential of a homogeneous polyhedron.
#
# Description:
#   Implements the face-summation potential formulation described in
#   Periyandy and Bevis (2025)
#   The function potential() evaluates the scalar field at one or more
#   observation points using precomputed geometric quantities from
#   the PolyhedronGravity structure.
# ===============================================================

module PolyhedronPotential

using ..PolyhedronBase
using LinearAlgebra, StaticArrays, Base.Threads
include("PolyhedronHelpers.jl")

export potential

function potential(pg::PolyhedronGravity, points::AbstractVecOrMat{<:Real})
    Pts = (ndims(points) == 1) ? reshape(Float64.(collect(points)), 1, 3) : Float64.(Matrix(points))
    Np = size(Pts, 1)
    out = zeros(Float64, Np)

    F = length(pg.Vi)
    G_rho = pg.G * pg.rho
    eps = pg.eps

    Threads.@threads for i in 1:Np
        pb = @view Pts[i, :]
        P = SVector{3,Float64}(pb[1], pb[2], pb[3])

        sumb = 0.0
        @inbounds for f in 1:F
            Vi = pg.Vi[f]; Vj = pg.Vj[f]; Vk = pg.Vk[f]; nh = pg.n_hat[f]

            Pi = Vi - P; Pj = Vj - P; Pk = Vk - P
            ri = norm(Pi); rj = norm(Pj); rk = norm(Pk)

            diffz = dot(Pi, nh); dz2 = diffz * diffz
            det_ij = dot(cross(Pi, Pj), nh)
            det_jk = dot(cross(Pj, Pk), nh)
            det_ki = dot(cross(Pk, Pi), nh)

            L12 = logterm(ri, rj, pg.L_ij[f], eps) * pg.inv_L_ij[f]
            L23 = logterm(rj, rk, pg.L_jk[f], eps) * pg.inv_L_jk[f]
            L31 = logterm(rk, ri, pg.L_ki[f], eps) * pg.inv_L_ki[f]

            numerator = diffz * pg.n_raw_norm[f]
            S1 = arctanterm(numerator, det_ki, det_ij, dz2, pg.dot_ij_ki[f], ri, eps)
            S2 = arctanterm(numerator, det_ij, det_jk, dz2, pg.dot_jk_ij[f], rj, eps)
            S3 = arctanterm(numerator, det_jk, det_ki, dz2, pg.dot_ki_jk[f], rk, eps)

            term1 = diffz * (det_ij * L12 + det_jk * L23 + det_ki * L31)
            term2 = dz2 * (S1 + S2 + S3 - sign(diffz) * π)
            sumb += 0.5 * (term1 - term2)
        end

        out[i] = -G_rho * sumb
    end

    return (Np == 1) ? out[1] : out
end

end # module
