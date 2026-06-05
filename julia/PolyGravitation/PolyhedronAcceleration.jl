# ===============================================================
# PolyhedronAcceleration.jl
# Module: PolyhedronAcceleration
#
# Purpose:
#   Computes the gravitational acceleration vector field
#   of a homogeneous polyhedron.
#
# Description:
#   Implements the dyadic formulation of the acceleration
#   (gradient of potential) as described by Periyandy and Bevis (2025).
#   The method uses precomputed geometric dyads and solid-angle
#   functions to produce stable and continuous acceleration
#   values across interior, boundary, and exterior regions.
# ===============================================================
# ===============================================================
# PolyhedronAcceleration.jl
# Module: PolyhedronAcceleration
#
# Acceleration via dyadic formulation:
#   g(P) = G*rho [ Σ_f ω_f(P) (F_f r_f)  -  Σ_e L_e(P) (E_e r_u) ]
# where r_f = Vi - P and r_u = Vu - P (edge u endpoint).
# ===============================================================

module PolyhedronAcceleration

using ..PolyhedronBase
using LinearAlgebra, StaticArrays, Base.Threads
include("PolyhedronHelpers.jl")

export acceleration

# multiply a 3x3 dyad (row-major flattened) by a vector
@inline function dyad_times_vec_rowmajor(D::NTuple{9,Float64}, r::SVector{3,Float64})
    D11,D12,D13,D21,D22,D23,D31,D32,D33 = D
    return SVector{3,Float64}(
        D11*r[1] + D12*r[2] + D13*r[3],
        D21*r[1] + D22*r[2] + D23*r[3],
        D31*r[1] + D32*r[2] + D33*r[3]
    )
end

function acceleration(pg::PolyhedronGravity, points::AbstractVecOrMat{<:Real})
    Pts = (ndims(points) == 1) ? reshape(Float64.(collect(points)), 1, 3) : Float64.(Matrix(points))
    Np = size(Pts, 1)
    out = zeros(Float64, Np, 3)

    Nf = length(pg.Vi)
    Ne = length(pg.edge_len)
    G_rho = pg.G * pg.rho
    eps = pg.eps

    Threads.@threads for i in 1:Np
        pb = @view Pts[i, :]
        P = SVector{3,Float64}(pb[1], pb[2], pb[3])

        # Face contribution: Σ ω_f * (F_f * r_f)
        B = SVector{3,Float64}(0.0, 0.0, 0.0)
        @inbounds for f in 1:Nf
            ω = solid_angle_tri(P, pg.Vi[f], pg.Vj[f], pg.Vk[f], eps)
            rf = pg.Vi[f] - P
            # F_f dyad (row-major) times rf
            D = ntuple(j->pg.Fdyads[f,j], 9)
            Fr = dyad_times_vec_rowmajor(D, rf)
            B += ω * Fr
        end

        # Edge contribution: Σ log_e * (E_e * r_u)
        A = SVector{3,Float64}(0.0, 0.0, 0.0)
        @inbounds for e in 1:Ne
            Vu = pg.edge_u[e]
            rv = Vu - P
            rj = pg.edge_v[e] - P
            ri_norm = norm(rv); rj_norm = norm(rj)
            loge = logterm(ri_norm, rj_norm, pg.edge_len[e], eps)
            Ee = ntuple(j->pg.edge_E[e,j], 9)
            Evr = dyad_times_vec_rowmajor(Ee, rv)
            A += loge * Evr
        end

        g = G_rho * (B - A)
        out[i,1] = g[1]; out[i,2] = g[2]; out[i,3] = g[3]
    end

    return (Np == 1) ? vec(out[1, :]) : out
end

end # module
