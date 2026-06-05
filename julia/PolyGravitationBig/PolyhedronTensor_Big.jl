# ===============================================================
# PolyhedronTensor_Big.jl
# Module: PolyhedronTensorBig
#
# Purpose:
#   Computes the gravitational gradient tensor of a homogeneous
#   polyhedron using BigFloat arithmetic.
#
# Description:
#   Implements the dyadic second-derivative formulation
#   (Werner–Scheeres) in a numerically stable BigFloat form,
#   consistent with Periyandy & Bevis (2025).
#   Each tensor component represents the spatial derivative
#   of the acceleration field, forming the 3×3 Hessian matrix
#   of the gravitational potential.
# ===============================================================

module PolyhedronTensorBig

using ..PolyhedronBaseBig
using LinearAlgebra, StaticArrays, Base.Threads
include("PolyhedronHelpers_Big.jl")

export gravity_tensor_big

# ---------------------------------------------------------------
# Helper: add two 3×3 matrices represented as row-major 9-tuples
# ---------------------------------------------------------------
@inline function add_rowmajor!(acc::NTuple{9,BigFloat}, add::NTuple{9,BigFloat})
    return ntuple(k -> acc[k] + add[k], 9)
end

# ---------------------------------------------------------------
# Compute gravity gradient tensor field (BigFloat)
# ---------------------------------------------------------------
function gravity_tensor_big(pg::PolyhedronGravityBig, points::AbstractVecOrMat{<:Real})
    # Normalize input into N×3 BigFloat matrix
    Pts = (ndims(points) == 1) ? reshape(BigFloat.(collect(points)), 1, 3) : BigFloat.(Matrix(points))
    Np = size(Pts, 1)
    out = zeros(BigFloat, Np, 3, 3)

    Nf = length(pg.Vi)
    Ne = length(pg.edge_len)
    G_rho = pg.G * pg.rho
    eps = pg.eps

    Threads.@threads for i in 1:Np
        pb = @view Pts[i, :]
        P = SVector{3,BigFloat}(pb[1], pb[2], pb[3])

        # Initialize face and edge accumulators
        face_sum = ntuple(_ -> BigFloat(0), 9)
        edge_sum = ntuple(_ -> BigFloat(0), 9)

        # -------------------------------------------------------
        # Face contributions:  Σ_f ω_f * F_f
        # -------------------------------------------------------
        @inbounds for f in 1:Nf
            ω = solid_angle_tri(P, pg.Vi[f], pg.Vj[f], pg.Vk[f], eps)
            Ff = ntuple(j -> pg.Fdyads[f, j], 9)
            face_sum = ntuple(k -> face_sum[k] + ω * Ff[k], 9)
        end

        # -------------------------------------------------------
        # Edge contributions:  Σ_e L_e * E_e
        # -------------------------------------------------------
        @inbounds for e in 1:Ne
            Vu = pg.edge_u[e]
            Vv = pg.edge_v[e]
            ru = norm(Vu - P)
            rv = norm(Vv - P)
            loge = logterm(ru, rv, pg.edge_len[e], eps)
            Ee = ntuple(j -> pg.edge_E[e, j], 9)
            edge_sum = ntuple(k -> edge_sum[k] + loge * Ee[k], 9)
        end

        # -------------------------------------------------------
        # Final tensor: Γ = Gρ (edge_sum - face_sum)
        # -------------------------------------------------------
        Γ = ntuple(k -> G_rho * (edge_sum[k] - face_sum[k]), 9)

        out[i,1,1] = Γ[1]; out[i,1,2] = Γ[2]; out[i,1,3] = Γ[3]
        out[i,2,1] = Γ[4]; out[i,2,2] = Γ[5]; out[i,2,3] = Γ[6]
        out[i,3,1] = Γ[7]; out[i,3,2] = Γ[8]; out[i,3,3] = Γ[9]
    end

    return (Np == 1) ? dropdims(out; dims=1) : out
end

end # module
