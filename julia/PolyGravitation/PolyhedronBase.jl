# ===============================================================
# PolyhedronBase.jl
# Module: PolyhedronBase
#
# The Gravitational Field of a Homogeneous Polyhedron
# Authors: Thunendran Periyandy, Michael Bevis
#
# Purpose:
#   Defines data structures and precomputations for a constant-density
#   polyhedron used to evaluate potential, acceleration, and tensor.
#
# Notes:
#   - Compatible with all existing examples
#   - Adds optional `mode` argument to control precomputations:
#       mode = :potential     → compute face geometry only
#       mode = :acceleration  → compute faces + Fdyads
#       mode = :tensor/full   → compute everything (default)
# ===============================================================

module PolyhedronBase

using LinearAlgebra
using StaticArrays
using Base.Threads

include("PolyhedronHelpers.jl")

export PolyhedronGravity, build_polyhedron

mutable struct PolyhedronGravity
    # physical
    G::Float64
    rho::Float64
    eps::Float64

    # mesh (faces are M×3, vertices are N×3; 1-based indices)
    F::Matrix{Int}
    V::Vector{SVector{3,Float64}}

    # per-face vertex triplets and normals
    Vi::Vector{SVector{3,Float64}}
    Vj::Vector{SVector{3,Float64}}
    Vk::Vector{SVector{3,Float64}}
    n_hat::Vector{SVector{3,Float64}}
    n_raw_norm::Vector{Float64}

    # potential invariants (edge lengths and dots)
    L_ij::Vector{Float64}
    L_jk::Vector{Float64}
    L_ki::Vector{Float64}
    inv_L_ij::Vector{Float64}
    inv_L_jk::Vector{Float64}
    inv_L_ki::Vector{Float64}
    dot_ij_ki::Vector{Float64}
    dot_jk_ij::Vector{Float64}
    dot_ki_jk::Vector{Float64}

    # face dyads (Nf × 3 × 3)
    Fdyads::Matrix{Float64}  # stored as Nf×9 (row-major) for cache locality

    # edge topology and dyads
    edge_i::Vector{Int}
    edge_j::Vector{Int}
    edge_len::Vector{Float64}
    edge_E::Matrix{Float64}  # Ne×9 (row-major)
    edge_u::Vector{SVector{3,Float64}} # endpoints (u)
    edge_v::Vector{SVector{3,Float64}} # endpoints (v)
end

# ---------------------------------------------------------------
# build_polyhedron
# ---------------------------------------------------------------
function build_polyhedron(vertices::Matrix{<:Real}, faces::Matrix{<:Integer};
                          G::Real=1.0, rho::Real=1.0, eps::Real=0.0,
                          orient_faces::Bool=true, mode::Symbol=:full)

    Gf, rhof, epsf = Float64(G), Float64(rho), Float64(eps)

    # vertices -> SVectors
    @assert size(vertices,2) == 3 "vertices must be N×3"
    Nv = size(vertices,1)
    V  = [SVector{3,Float64}(Float64(vertices[i,1]),
                              Float64(vertices[i,2]),
                              Float64(vertices[i,3])) for i in 1:Nv]

    # faces matrix (M×3), 1-based
    @assert size(faces,2) == 3 "faces must be M×3"
    F = Matrix{Int}(faces)
    if minimum(F) == 0
        F .+= 1
    end
    Nf = size(F,1)

    # Face vertices
    Vi = Vector{SVector{3,Float64}}(undef, Nf)
    Vj = similar(Vi); Vk = similar(Vi)

    # Raw normals
    normals_raw = Vector{SVector{3,Float64}}(undef, Nf)
    n_raw_norm  = zeros(Float64, Nf)

    for f in 1:Nf
        i1,i2,i3 = F[f,1], F[f,2], F[f,3]
        Vi[f] = V[i1]; Vj[f] = V[i2]; Vk[f] = V[i3]
        n = cross(Vj[f] - Vi[f], Vk[f] - Vi[f])
        normals_raw[f] = n
        n_raw_norm[f] = norm(n)
    end

    # Optional outward orientation via centroid test
    if orient_faces
        centroid = reduce(+, V) / length(V)
        flip = falses(Nf)
        for f in 1:Nf
            fc = (Vi[f] + Vj[f] + Vk[f]) / 3.0
            if dot(normals_raw[f], fc - centroid) < 0.0
                flip[f] = true
            end
        end
        if any(flip)
            for f in findall(flip)
                F[f,2], F[f,3] = F[f,3], F[f,2]
            end
            # rebuild with oriented faces
            for f in 1:Nf
                i1,i2,i3 = F[f,1], F[f,2], F[f,3]
                Vi[f] = V[i1]; Vj[f] = V[i2]; Vk[f] = V[i3]
                n = cross(Vj[f] - Vi[f], Vk[f] - Vi[f])
                normals_raw[f] = n
                n_raw_norm[f] = norm(n)
            end
        end
    end

    # Unit normals (safe)
    n_hat = Vector{SVector{3,Float64}}(undef, Nf)
    for f in 1:Nf
        nrm = n_raw_norm[f]
        n_hat[f] = (nrm > epsf) ? (normals_raw[f] / nrm) : SVector{3,Float64}(0.0,0.0,0.0)
    end

    # Edge vectors per face (for potential invariants)
    e_ij = [Vj[f] - Vi[f] for f in 1:Nf]
    e_jk = [Vk[f] - Vj[f] for f in 1:Nf]
    e_ki = [Vi[f] - Vk[f] for f in 1:Nf]

    L_ij = [norm(e_ij[f]) for f in 1:Nf]
    L_jk = [norm(e_jk[f]) for f in 1:Nf]
    L_ki = [norm(e_ki[f]) for f in 1:Nf]

    inv_L_ij = [L_ij[f] > epsf ? 1.0 / L_ij[f] : 0.0 for f in 1:Nf]
    inv_L_jk = [L_jk[f] > epsf ? 1.0 / L_jk[f] : 0.0 for f in 1:Nf]
    inv_L_ki = [L_ki[f] > epsf ? 1.0 / L_ki[f] : 0.0 for f in 1:Nf]

    dot_ij_ki = [dot(e_ij[f], e_ki[f]) for f in 1:Nf]
    dot_jk_ij = [dot(e_jk[f], e_ij[f]) for f in 1:Nf]
    dot_ki_jk = [dot(e_ki[f], e_jk[f]) for f in 1:Nf]

    # ------------------------------------------------------------
    # Conditional precomputation based on mode
    # ------------------------------------------------------------

    # Face dyads
    if mode == :potential
        Fdyads = zeros(Float64, 0, 0)
        edge_i = Int[]; edge_j = Int[]; edge_len = Float64[]
        edge_E_mat = zeros(Float64, 0, 9)
        edge_u = SVector{3,Float64}[]; edge_v = SVector{3,Float64}[]

    elseif mode == :acceleration
        # compute Fdyads only
        Fdyads = zeros(Float64, Nf, 9)
        for f in 1:Nf
            nx, ny, nz = n_hat[f]
            Fdyads[f,1] = nx*nx; Fdyads[f,2] = nx*ny; Fdyads[f,3] = nx*nz
            Fdyads[f,4] = ny*nx; Fdyads[f,5] = ny*ny; Fdyads[f,6] = ny*nz
            Fdyads[f,7] = nz*nx; Fdyads[f,8] = nz*ny; Fdyads[f,9] = nz*nz
        end
        edge_i = Int[]; edge_j = Int[]; edge_len = Float64[]
        edge_E_mat = zeros(Float64, 0, 9)
        edge_u = SVector{3,Float64}[]; edge_v = SVector{3,Float64}[]

    else
        # Full computation (tensor or default)
        Fdyads = zeros(Float64, Nf, 9)
        for f in 1:Nf
            nx, ny, nz = n_hat[f]
            Fdyads[f,1] = nx*nx; Fdyads[f,2] = nx*ny; Fdyads[f,3] = nx*nz
            Fdyads[f,4] = ny*nx; Fdyads[f,5] = ny*ny; Fdyads[f,6] = ny*nz
            Fdyads[f,7] = nz*nx; Fdyads[f,8] = nz*ny; Fdyads[f,9] = nz*nz
        end

        # Edge topology and dyads
        edge_to_faces = Dict{Tuple{Int,Int}, Vector{Int}}()
        edge_map      = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
        for f in 1:Nf
            a,b,c = F[f,1], F[f,2], F[f,3]
            for (u,v) in ((a,b), (b,c), (c,a))
                key = u < v ? (u,v) : (v,u)
                push!(get!(edge_to_faces, key, Int[]), f)
                edge_map[key] = (u,v)
            end
        end

        edge_i = Int[]; edge_j = Int[]; edge_len = Float64[]; edge_E = Float64[]
        edge_u = SVector{3,Float64}[]; edge_v = SVector{3,Float64}[]

        for (key, faces4edge) in edge_to_faces
            length(faces4edge) == 2 || continue  # skip boundary edges
            u,v = edge_map[key]
            fa, fb = faces4edge[1], faces4edge[2]
            na, nb = n_hat[fa], n_hat[fb]

            evec = V[v] - V[u]
            L = norm(evec)
            L > epsf || continue
            ehat = evec / L

            # orientation logic to choose signs of tA and tB
            a,b,c = F[fa,1], F[fa,2], F[fa,3]
            off = (a != u && a != v) ? a : ((b != u && b != v) ? b : c)
            face_a = (a,b,c); off_idx = (off == a) ? 1 : ((off == b) ? 2 : 3)
            after_off = face_a[(off_idx % 3) + 1]

            tA = cross(na, ehat)
            tB = cross(nb, ehat)
            if after_off == u
                tA = -tA
            else
                tB = -tB
            end
            Ee = na * transpose(tA) + nb * transpose(tB) # 3×3

            # store
            push!(edge_i, u); push!(edge_j, v); push!(edge_len, L)
            push!(edge_u, V[u]); push!(edge_v, V[v])
            # flatten row-major
            push!(edge_E, Ee[1,1], Ee[1,2], Ee[1,3],
                          Ee[2,1], Ee[2,2], Ee[2,3],
                          Ee[3,1], Ee[3,2], Ee[3,3])
        end
        edge_E_mat = reshape(collect(edge_E), 9, length(edge_len))' # Ne×9
    end

    # ------------------------------------------------------------
    return PolyhedronGravity(Gf, rhof, epsf, F, V,
                             Vi, Vj, Vk, n_hat, n_raw_norm,
                             L_ij, L_jk, L_ki, inv_L_ij, inv_L_jk, inv_L_ki,
                             dot_ij_ki, dot_jk_ij, dot_ki_jk,
                             Fdyads,
                             edge_i, edge_j, edge_len, edge_E_mat,
                             edge_u, edge_v)
end

end # module
