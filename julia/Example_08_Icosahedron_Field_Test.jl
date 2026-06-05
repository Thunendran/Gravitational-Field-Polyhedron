###############################################################
# Example 08: Icosahedron Gravitational Field Validation
# Tests potential, acceleration, and tensor at specific points
# defined by rational and shifted coordinates.
###############################################################

using LinearAlgebra
using Printf

# Import the Polyhedron Gravitation model
include(joinpath("PolyGravitation", "Polyhedron_Gravitation.jl"))
using .Polyhedron_Gravitation

###############################################################
# 1) Helper Functions
###############################################################

normalize_vec(v::AbstractVector{<:Real}) = v ./ norm(v)

function format_point_as_fraction(p::Vector{Float64})
    coords = String[]
    for c in p
        r = rationalize(c; tol=1e-10)
        if denominator(r) == 1
            push!(coords, string(numerator(r)))
        else
            push!(coords, string(numerator(r)) * "/" * string(denominator(r)))
        end
    end
    return "(" * join(coords, ", ") * ")"
end

format_point_as_float(p::Vector{Float64}) =
    @sprintf("(%0.10e, %0.10e, %0.10e)", p...)

###############################################################
# 2) Create Transformed Icosahedron
###############################################################
function create_transformed_icosahedron()
    φ = (1 + sqrt(5)) / 2
    V_std = [
        [-1,  φ, 0], [ 1,  φ, 0], [-1, -φ, 0], [ 1, -φ, 0],
        [0, -1,  φ], [0,  1,  φ], [0, -1, -φ], [0,  1, -φ],
        [ φ, 0, -1], [ φ, 0,  1], [-φ, 0, -1], [-φ, 0,  1]
    ] |> x -> reduce(vcat, permutedims.(x))

    # convert to 1-based indexing correctly
    F_raw = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ]
    F = [f .+ 1 for f in F_raw]  # broadcast correctly
    F = reduce(vcat, permutedims.(F))

    # Define source and target faces
    s1, s2, s3 = eachrow(V_std)[[1, 12, 6]]
    t1 = [0.0, 0.0, 0.0]
    t2 = [1.0, 1.0, 0.0]
    t3 = [1.0, 0.0, 1.0]

    scale = norm(t2 - t1) / norm(s2 - s1)
    u_s = normalize_vec(s2 - s1)
    w_s = normalize_vec(cross(u_s, s3 - s1))
    v_s = cross(w_s, u_s)
    A = hcat(u_s, v_s, w_s)

    u_t = normalize_vec(t2 - t1)
    w_t = normalize_vec(cross(u_t, t3 - t1))
    v_t = cross(w_t, u_t)
    B = hcat(u_t, v_t, w_t)

    R = B * A'
    V_trans = [((R * (p - s1)) * scale) + t1 for p in eachrow(V_std)]
    V_mat = reduce(vcat, permutedims.(V_trans))

    return V_mat, Matrix{Int64}(F)
end

###############################################################
# 3) Build Icosahedron Model
###############################################################
G = 1.0
ρ = 1.0
ϵ = 1e-10

V, F = create_transformed_icosahedron()
println("Generated an icosahedron with $(size(V,1)) vertices and $(size(F,1)) faces.")
println("One face is defined by integer coordinates: (0,0,0), (1,1,0), (1,0,1).\n")

model = build_polyhedron(V, F; G=G, rho=ρ, eps=ϵ, orient_faces=true)

###############################################################
# 4) Define Test Points
###############################################################
v1 = [0.0, 0.0, 0.0]
v2 = [1.0, 1.0, 0.0]
v3 = [1.0, 0.0, 1.0]
face_center = (v1 + v2 + v3) ./ 3.0

test_points = Dict(
    "Vertex"          => v2,
    "On Edge"         => (v1 + v2) ./ 2.0,
    "On Face"         => face_center,
    "On Extended Edge"=> v1 + 2.0 * (v2 - v1),
    "On Extended Face"=> face_center + [1.0, 0.0, 0.0],
    "Interior"        => face_center ./ 2.0,
    "Exterior"        => [2.0, 2.0, 2.0]
)

###############################################################
# 5) Output Setup
###############################################################
out_path = "icosahedron_field_results.txt"
open(out_path, "w") do io
    header = @sprintf("%-25s | %-35s | %-25s | %-65s | %-100s",
        "Point ID", "Test Point (x,y,z)", "Potential", "Acceleration (gx,gy,gz)", "Tensor (xx,xy,xz,yy,yz,zz)")
    println(io, "-"^length(header))
    println(io, header)
    println(io, "-"^length(header))

    for (name, p_orig) in test_points
        p_shift = if name == "On Edge"
            p_orig + ϵ * normalize_vec(v2 - v1)
        elseif name == "On Face"
            p_orig + ϵ * normalize_vec(v2 - v1)
        else
            p_orig .+ ϵ
        end

        for (p, label, fmtfun) in [(p_orig, "$name (original)", format_point_as_fraction),
                                   (p_shift, "$name (shifted)", format_point_as_float)]
            try
                Φ = potential(model, p)
                g = acceleration(model, p)
                Γ = gravity_tensor(model, p)

                p_str = fmtfun(p)
                Φ_str = @sprintf("%.14e", Φ)
                g_str = @sprintf("(%.14e, %.14e, %.14e)", g...)
                g_xx, g_xy, g_xz = Γ[1,1], Γ[1,2], Γ[1,3]
                g_yy, g_yz, g_zz = Γ[2,2], Γ[2,3], Γ[3,3]
                Γ_str = @sprintf("(%.14e, %.14e, %.14e, %.14e, %.14e, %.14e)",
                                 g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)

                println(io, @sprintf("%-25s | %-35s | %-25s | %-65s | %-100s",
                                     label, p_str, Φ_str, g_str, Γ_str))
            catch e
                p_str = fmtfun(p)
                println(io, @sprintf("%-25s | %-35s | Calculation failed: %s", label, p_str, e))
            end
        end
    end

    println(io, "-"^length(header))
end

println("Results saved to icosahedron_field_results.txt ")
