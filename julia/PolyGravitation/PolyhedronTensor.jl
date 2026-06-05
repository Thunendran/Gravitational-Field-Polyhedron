# ===============================================================
# PolyhedronTensor.jl
# Module: PolyhedronTensor
#
# Purpose:
#   Computes the gravitational gradient tensor of a homogeneous polyhedron.
#
# Description:
#   Implements the dyadic second-derivative formulation (Werner–Scheeres)
#   in a numerically stable form consistent with the work of
#   Periyandy and Bevis (2025). Each tensor component represents
#   the spatial derivative of the acceleration field, forming the
#   3×3 Hessian matrix of the gravitational potential.
# ===============================================================

# ===============================================================
# PolyhedronTensor.jl
# Module: PolyhedronTensor
#
# Gravity gradient tensor via dyadic formulation:
#   Γ(P) = G*rho [ Σ_e L_e(P) E_e  -  Σ_f ω_f(P) F_f ]
# ===============================================================

module PolyhedronTensor

using ..PolyhedronBase
using LinearAlgebra, StaticArrays, Base.Threads
include("PolyhedronHelpers.jl")

export gravity_tensor

# add two 3×3 using row-major 9-tuple
@inline function add_rowmajor!(acc::NTuple{9,Float64}, add::NTuple{9,Float64})
    return ntuple(k->acc[k] + add[k], 9)
end

function gravity_tensor(pg::PolyhedronGravity, points::AbstractVecOrMat{<:Real})
    Pts = (ndims(points) == 1) ? reshape(Float64.(collect(points)), 1, 3) : Float64.(Matrix(points))
    Np = size(Pts, 1)
    out = zeros(Float64, Np, 3, 3)

    Nf = length(pg.Vi)
    Ne = length(pg.edge_len)
    G_rho = pg.G * pg.rho
    eps = pg.eps

    Threads.@threads for i in 1:Np
        pb = @view Pts[i, :]
        P = SVector{3,Float64}(pb[1], pb[2], pb[3])

        # sums in row-major 9-tuple form
        face_sum = ntuple(_->0.0, 9)
        edge_sum = ntuple(_->0.0, 9)

        # faces
        @inbounds for f in 1:Nf
            ω = solid_angle_tri(P, pg.Vi[f], pg.Vj[f], pg.Vk[f], eps)
            Ff = ntuple(j->pg.Fdyads[f,j], 9)
            face_sum = ntuple(k->face_sum[k] + ω * Ff[k], 9)
        end

        # edges
        @inbounds for e in 1:Ne
            Vu = pg.edge_u[e]; Vv = pg.edge_v[e]
            ru = norm(Vu - P); rv = norm(Vv - P)
            loge = logterm(ru, rv, pg.edge_len[e], eps)
            Ee = ntuple(j->pg.edge_E[e,j], 9)
            edge_sum = ntuple(k->edge_sum[k] + loge * Ee[k], 9)
        end

        # Γ = G*rho (edge_sum - face_sum)
        Γ = ntuple(k->G_rho * (edge_sum[k] - face_sum[k]), 9)
        out[i,1,1]=Γ[1]; out[i,1,2]=Γ[2]; out[i,1,3]=Γ[3]
        out[i,2,1]=Γ[4]; out[i,2,2]=Γ[5]; out[i,2,3]=Γ[6]
        out[i,3,1]=Γ[7]; out[i,3,2]=Γ[8]; out[i,3,3]=Γ[9]
    end

    return (Np == 1) ? dropdims(out; dims=1) : out
end

end # module

