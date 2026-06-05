# ===============================================================
# PolyhedronBase_Big.jl
# Module: PolyhedronBaseBig
#
# The Gravitational Field of a Homogeneous Polyhedron (BigFloat)
# Authors: Thunendran Periyandy, Michael Bevis
#
# Purpose:
#   High-precision version of PolyhedronBase.jl using BigFloat
#   for all arithmetic and data structures.
#
# Notes:
#   - Compatible with all existing examples
#   - Adds optional `mode` argument:
#       mode = :potential     → compute face geometry only
#       mode = :acceleration  → compute faces + Fdyads
#       mode = :tensor/full   → compute everything (default)
# ===============================================================

module PolyhedronBaseBig

using LinearAlgebra
using StaticArrays
using Base.Threads

include("PolyhedronHelpers_Big.jl")

export PolyhedronGravityBig, build_polyhedron_big

mutable struct PolyhedronGravityBig
    # physical
    G::BigFloat
    rho::BigFloat
    eps::BigFloat

    # mesh
    F::Matrix{Int}
    V::Vector{SVector{3,BigFloat}}

    # per-face vertices and normals
    Vi::Vector{SVector{3,BigFloat}}
    Vj::Vector{SVector{3,BigFloat}}
    Vk::Vector{SVector{3,BigFloat}}
    n_hat::Vector{SVector{3,BigFloat}}
    n_raw_norm::Vector{BigFloat}

    # potential invariants
    L_ij::Vector{BigFloat}
    L_jk::Vector{BigFloat}
    L_ki::Vector{BigFloat}
    inv_L_ij::Vector{BigFloat}
    inv_L_jk::Vector{BigFloat}
    inv_L_ki::Vector{BigFloat}
    dot_ij_ki::Vector{BigFloat}
    dot_jk_ij::Vector{BigFloat}
    dot_ki_jk::Vector{BigFloat}

    # face dyads (Nf×9 row-major)
    Fdyads::Matrix{BigFloat}

    # edge topology and dyads
    edge_i::Vector{Int}
    edge_j::Vector{Int}
    edge_len::Vector{BigFloat}
    edge_E::Matrix{BigFloat}
    edge_u::Vector{SVector{3,BigFloat}}
    edge_v::Vector{SVector{3,BigFloat}}
end

# ---------------------------------------------------------------
# build_polyhedron_big
# ---------------------------------------------------------------
function build_polyhedron_big(vertices::Matrix{<:Real}, faces::Matrix{<:Integer};
                              G::Real=1.0, rho::Real=1.0, eps::Real=0.0,
                              orient_faces::Bool=true, mode::Symbol=:full)

    Gb, rhob, epsb = BigFloat(G), BigFloat(rho), BigFloat(eps)

    @assert size(vertices,2) == 3 "vertices must be N×3"
    Nv = size(vertices,1)
    V = [SVector{3,BigFloat}(BigFloat(vertices[i,1]),
                              BigFloat(vertices[i,2]),
                              BigFloat(vertices[i,3])) for i in 1:Nv]

    @assert size(faces,2) == 3 "faces must be M×3"
    F = Matrix{Int}(faces)
    if minimum(F) == 0
        F .+= 1
    end
    Nf = size(F,1)

    Vi = Vector{SVector{3,BigFloat}}(undef, Nf)
    Vj = similar(Vi); Vk = similar(Vi)
    normals_raw = Vector{SVector{3,BigFloat}}(undef, Nf)
    n_raw_norm = zeros(BigFloat, Nf)

    for f in 1:Nf
        i1,i2,i3 = F[f,1], F[f,2], F[f,3]
        Vi[f] = V[i1]; Vj[f] = V[i2]; Vk[f] = V[i3]
        n = cross(Vj[f] - Vi[f], Vk[f] - Vi[f])
        normals_raw[f] = n
        n_raw_norm[f] = norm(n)
    end

    if orient_faces
        centroid = reduce(+, V) / BigFloat(length(V))
        flip = falses(Nf)
        for f in 1:Nf
            fc = (Vi[f] + Vj[f] + Vk[f]) / BigFloat(3)
            if dot(normals_raw[f], fc - centroid) < 0
                flip[f] = true
            end
        end
        if any(flip)
            for f in findall(flip)
                F[f,2], F[f,3] = F[f,3], F[f,2]
            end
            for f in 1:Nf
                i1,i2,i3 = F[f,1], F[f,2], F[f,3]
                Vi[f] = V[i1]; Vj[f] = V[i2]; Vk[f] = V[i3]
                n = cross(Vj[f] - Vi[f], Vk[f] - Vi[f])
                normals_raw[f] = n
                n_raw_norm[f] = norm(n)
            end
        end
    end

    n_hat = Vector{SVector{3,BigFloat}}(undef, Nf)
    for f in 1:Nf
        nrm = n_raw_norm[f]
        n_hat[f] = (nrm > epsb) ? (normals_raw[f] / nrm) : SVector{3,BigFloat}(0,0,0)
    end

    e_ij = [Vj[f] - Vi[f] for f in 1:Nf]
    e_jk = [Vk[f] - Vj[f] for f in 1:Nf]
    e_ki = [Vi[f] - Vk[f] for f in 1:Nf]

    L_ij = [norm(e_ij[f]) for f in 1:Nf]
    L_jk = [norm(e_jk[f]) for f in 1:Nf]
    L_ki = [norm(e_ki[f]) for f in 1:Nf]

    inv_L_ij = [L_ij[f] > epsb ? inv(L_ij[f]) : BigFloat(0) for f in 1:Nf]
    inv_L_jk = [L_jk[f] > epsb ? inv(L_jk[f]) : BigFloat(0) for f in 1:Nf]
    inv_L_ki = [L_ki[f] > epsb ? inv(L_ki[f]) : BigFloat(0) for f in 1:Nf]

    dot_ij_ki = [dot(e_ij[f], e_ki[f]) for f in 1:Nf]
    dot_jk_ij = [dot(e_jk[f], e_ij[f]) for f in 1:Nf]
    dot_ki_jk = [dot(e_ki[f], e_jk[f]) for f in 1:Nf]

    # ------------------------------------------------------------
    # Conditional precomputation (same as Float version)
    # ------------------------------------------------------------

    if mode == :potential
        Fdyads = zeros(BigFloat, 0, 0)
        edge_i = Int[]; edge_j = Int[]; edge_len = BigFloat[]
        edge_E_mat = zeros(BigFloat, 0, 9)
        edge_u = SVector{3,BigFloat}[]; edge_v = SVector{3,BigFloat}[]

    elseif mode == :acceleration
        # Fdyads only
        Fdyads = zeros(BigFloat, Nf, 9)
        for f in 1:Nf
            nx, ny, nz = n_hat[f]
            Fdyads[f,1] = nx*nx; Fdyads[f,2] = nx*ny; Fdyads[f,3] = nx*nz
            Fdyads[f,4] = ny*nx; Fdyads[f,5] = ny*ny; Fdyads[f,6] = ny*nz
            Fdyads[f,7] = nz*nx; Fdyads[f,8] = nz*ny; Fdyads[f,9] = nz*nz
        end
        edge_i = Int[]; edge_j = Int[]; edge_len = BigFloat[]
        edge_E_mat = zeros(BigFloat, 0, 9)
        edge_u = SVector{3,BigFloat}[]; edge_v = SVector{3,BigFloat}[]

    else
        # Full computation (tensor / default)
        Fdyads = zeros(BigFloat, Nf, 9)
        for f in 1:Nf
            nx, ny, nz = n_hat[f]
            Fdyads[f,1] = nx*nx; Fdyads[f,2] = nx*ny; Fdyads[f,3] = nx*nz
            Fdyads[f,4] = ny*nx; Fdyads[f,5] = ny*ny; Fdyads[f,6] = ny*nz
            Fdyads[f,7] = nz*nx; Fdyads[f,8] = nz*ny; Fdyads[f,9] = nz*nz
        end

        edge_to_faces = Dict{Tuple{Int,Int}, Vector{Int}}()
        edge_map = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()

        for f in 1:Nf
            a,b,c = F[f,1], F[f,2], F[f,3]
            for (u,v) in ((a,b),(b,c),(c,a))
                key = u < v ? (u,v) : (v,u)
                push!(get!(edge_to_faces, key, Int[]), f)
                edge_map[key] = (u,v)
            end
        end

        edge_i = Int[]; edge_j = Int[]
        edge_len = BigFloat[]; edge_E = BigFloat[]
        edge_u = SVector{3,BigFloat}[]; edge_v = SVector{3,BigFloat}[]

        for (key, faces4edge) in edge_to_faces
            length(faces4edge) == 2 || continue
            u,v = edge_map[key]
            fa, fb = faces4edge[1], faces4edge[2]
            na, nb = n_hat[fa], n_hat[fb]

            evec = V[v] - V[u]
            L = norm(evec)
            L > epsb || continue
            ehat = evec / L

            a,b,c = F[fa,1], F[fa,2], F[fa,3]
            off = (a != u && a != v) ? a : ((b != u && b != v) ? b : c)
            face_a = (a,b,c)
            off_idx = (off == a) ? 1 : ((off == b) ? 2 : 3)
            after_off = face_a[(off_idx % 3) + 1]

            tA = cross(na, ehat)
            tB = cross(nb, ehat)
            if after_off == u
                tA = -tA
            else
                tB = -tB
            end
            Ee = na * transpose(tA) + nb * transpose(tB)

            push!(edge_i, u); push!(edge_j, v); push!(edge_len, L)
            push!(edge_u, V[u]); push!(edge_v, V[v])
            push!(edge_E, Ee[1,1], Ee[1,2], Ee[1,3],
                          Ee[2,1], Ee[2,2], Ee[2,3],
                          Ee[3,1], Ee[3,2], Ee[3,3])
        end
        edge_E_mat = reshape(collect(edge_E), 9, length(edge_len))'
    end

    return PolyhedronGravityBig(Gb, rhob, epsb, F, V,
                                Vi, Vj, Vk, n_hat, n_raw_norm,
                                L_ij, L_jk, L_ki, inv_L_ij, inv_L_jk, inv_L_ki,
                                dot_ij_ki, dot_jk_ij, dot_ki_jk,
                                Fdyads,
                                edge_i, edge_j, edge_len, edge_E_mat,
                                edge_u, edge_v)
end

end # module
